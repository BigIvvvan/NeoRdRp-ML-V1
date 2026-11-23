"""Single-GPU trainer with exact resume state and calibration-only thresholds."""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from .config import TrainConfig
from .data import PreparedSplit, collate, move_batch, plan_batches
from .losses import multi_task_loss
from .metrics import (
    boundary_prediction_metrics,
    detection_metrics,
    family_metrics,
    fit_endpoint_interval_offsets,
    fit_temperature,
    interval_prediction_metrics,
    selection_score,
    threshold_at_fpr,
)
from .model import NeoRdRpV1


@dataclass
class PredictionBundle:
    detection_logits: np.ndarray
    detection_labels: np.ndarray
    family_logits: np.ndarray
    family_labels: np.ndarray
    start_predictions: np.ndarray
    end_predictions: np.ndarray
    start_expectations: np.ndarray
    end_expectations: np.ndarray
    start_labels: np.ndarray
    end_labels: np.ndarray
    lengths: np.ndarray


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".pt")
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _strip_state_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in state.items():
        for prefix in ("module.", "model."):
            key = key.removeprefix(prefix)
        result[key] = value
    return result


def _load_initial_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    allowed_missing_prefixes: tuple[str, ...],
) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" in payload:
        state = payload["model_state"]
    elif "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    state = _strip_state_prefixes(state)
    aliases = {
        "protein_head.weight": "detection.3.weight",
        "protein_head.bias": "detection.3.bias",
    }
    state = {aliases.get(key, key): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    disallowed_missing = [
        key
        for key in missing
        if not key.startswith(tuple(allowed_missing_prefixes))
    ]
    if unexpected or disallowed_missing:
        raise ValueError(
            "Initial checkpoint is incompatible with the V1 model: "
            f"missing={disallowed_missing[:8]}, unexpected={list(unexpected)[:8]}"
        )
    print(
        json.dumps(
            {
                "init_from": str(checkpoint_path),
                "mapped_tensors": len(state) - len(unexpected),
                "missing_tensors": len(missing),
                "unexpected_tensors": len(unexpected),
            }
        ),
        flush=True,
    )


class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        splits: dict[str, PreparedSplit],
        family_names: list[str],
        resume_path: str | Path | None = None,
    ):
        self.config = config
        self.splits = splits
        self.family_names = family_names
        self.root = config.resolve_root()
        self.device = torch.device(
            "cuda" if config.train.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        if config.train.device == "cuda" and self.device.type != "cuda":
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.set_num_threads(config.train.num_threads)
        self._seed_all(config.seed)
        self.model = NeoRdRpV1(config.model, len(family_names)).to(self.device)
        if config.train.init_from:
            _load_initial_weights(
                self.model,
                self.root / config.train.init_from,
                config.train.init_allow_missing_prefixes,
            )
        self.optimizer = self._optimizer()
        if resume_path is not None:
            resolved_resume = Path(resume_path).expanduser().resolve()
            self.run_dir = resolved_resume.parent
            self.resume_path = resolved_resume
        else:
            self.run_dir = self.root / config.train.output_dir / f"{config.name}_s{config.seed}"
            self.run_dir.mkdir(parents=True, exist_ok=False)
            self.resume_path = None
        config_path = self.run_dir / "config.frozen.yaml"
        if not config_path.exists():
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "config_hash": config.hash,
                        "config": dataclasses.asdict(config),
                    },
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.global_step = 0
        self.best_score = -float("inf")
        self.epoch_without_improvement = 0
        self.best_metrics: dict[str, float] = {}
        self.calibration: dict[str, float] = {}
        self.sample_rng: np.random.Generator | None = None
        self.resume_sample_rng_state: dict[str, Any] | None = None

    def _seed_all(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _optimizer(self) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim == 1 or name.endswith(".bias"):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.config.optim.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
        )

    def _amp_context(self):
        if self.device.type != "cuda" or self.config.train.amp == "none":
            return torch.autocast(device_type=self.device.type, enabled=False)
        dtype = torch.bfloat16 if self.config.train.amp == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _lr(self, step: int, max_steps: int) -> float:
        warmup = max(1, int(max_steps * self.config.optim.warmup_fraction))
        if step < warmup:
            return self.config.optim.lr * (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.optim.min_lr + cosine * (
            self.config.optim.lr - self.config.optim.min_lr
        )

    def _set_lr(self, step: int, max_steps: int) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = self._lr(step, max_steps)

    def _log(self, payload: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    def _checkpoint_payload(self, epoch: int, batch_cursor: int) -> dict[str, Any]:
        numpy_state = np.random.get_state()
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": dataclasses.asdict(self.config),
            "config_hash": self.config.hash,
            "family_names": self.family_names,
            "epoch": epoch,
            "batch_cursor": batch_cursor,
            "global_step": self.global_step,
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "epoch_without_improvement": self.epoch_without_improvement,
            "calibration": self.calibration,
            "sample_rng_state": self.sample_rng.bit_generator.state
            if self.sample_rng is not None
            else None,
            "rng": {
                "python": random.getstate(),
                "numpy": {
                    "name": numpy_state[0],
                    "keys": numpy_state[1],
                    "pos": numpy_state[2],
                    "has_gauss": numpy_state[3],
                    "cached_gaussian": numpy_state[4],
                },
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }
        return payload

    def _save_last(self, epoch: int, batch_cursor: int) -> None:
        _atomic_torch_save(self._checkpoint_payload(epoch, batch_cursor), self.run_dir / "last.pt")

    def _save_best(self, epoch: int, batch_cursor: int) -> None:
        payload = self._checkpoint_payload(epoch, batch_cursor)
        payload["dev_metrics"] = self.best_metrics
        _atomic_torch_save(payload, self.run_dir / "best.pt")

    def _save_epoch(self, epoch: int, batch_cursor: int, metrics: dict[str, float]) -> None:
        payload = self._checkpoint_payload(epoch, batch_cursor)
        payload["dev_metrics"] = metrics
        _atomic_torch_save(payload, self.run_dir / f"epoch_{epoch + 1:02d}.pt")

    def resume(self, checkpoint_path: str | Path) -> tuple[int, int]:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("config_hash") != self.config.hash and not self._resume_compatible(
            payload
        ):
            raise ValueError("Resume configuration differs from checkpoint")
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.global_step = int(payload["global_step"])
        self.best_score = float(payload["best_score"])
        self.best_metrics = payload.get("best_metrics", {})
        self.epoch_without_improvement = int(payload.get("epoch_without_improvement", 0))
        self.calibration = payload.get("calibration", {})
        self.resume_sample_rng_state = payload.get("sample_rng_state")
        state = payload["rng"]
        random.setstate(state["python"])
        numpy_state = state["numpy"]
        np.random.set_state(
            (
                numpy_state["name"],
                numpy_state["keys"],
                numpy_state["pos"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state["cuda"]:
            torch.cuda.set_rng_state_all(state["cuda"])
        return int(payload["epoch"]), int(payload["batch_cursor"])

    def _resume_compatible(self, payload: dict[str, Any]) -> bool:
        """Allow evaluation resource changes while preserving the trained recipe."""
        saved = payload.get("config")
        if not isinstance(saved, dict):
            return False
        current = dataclasses.asdict(self.config)
        ignored = {
            "data": {"drop_missing_modalities"},
            "train": {
                "batch_size",
                "grad_accum",
                "max_tokens",
                "eval_batch_size",
                "eval_max_tokens",
            },
        }
        for section in ("name", "seed", "data", "model", "loss", "optim", "train"):
            left = saved.get(section)
            right = current.get(section)
            if isinstance(left, dict) and isinstance(right, dict):
                skip = ignored.get(section, set())
                left = {key: value for key, value in left.items() if key not in skip}
                right = {key: value for key, value in right.items() if key not in skip}
            if left != right:
                return False
        return True

    def _predict(self, split: PreparedSplit, max_length: int) -> PredictionBundle:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        indices = list(range(len(split)))
        detection_logits, detection_labels = [], []
        family_logits, family_labels = [], []
        start_predictions, end_predictions = [], []
        start_expectations, end_expectations = [], []
        start_labels, end_labels = [], []
        lengths = []
        with torch.inference_mode():
            for batch_indices in plan_batches(
                split,
                indices,
                self.config.train.eval_batch_size,
                self.config.train.eval_max_tokens,
                max_length,
                training=False,
            ):
                samples = [split.sample(index, max_length, False) for index in batch_indices]
                batch = move_batch(collate(samples), self.device)
                with self._amp_context():
                    output = self.model(batch)
                detection_logits.append(output["detection_logits"].float().cpu().numpy())
                detection_labels.append(batch["detection_labels"].cpu().numpy())
                family_logits.append(output["family_logits"].float().cpu().numpy())
                family_labels.append(batch["family_labels"].cpu().numpy())
                valid_mask = batch["attention_mask"].bool()
                start_masked = output["start_logits"].float().masked_fill(~valid_mask, -1e4)
                end_masked = output["end_logits"].float().masked_fill(~valid_mask, -1e4)
                positions = torch.arange(valid_mask.shape[1], device=valid_mask.device).float()
                start_predictions.append(
                    output.get("predicted_start", start_masked.argmax(dim=1))
                    .long()
                    .cpu()
                    .numpy()
                )
                end_predictions.append(
                    output.get("predicted_end", end_masked.argmax(dim=1))
                    .long()
                    .cpu()
                    .numpy()
                )
                start_expectations.append(
                    output.get(
                        "predicted_start_expectation",
                        (start_masked.softmax(dim=1) * positions).sum(dim=1),
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
                end_expectations.append(
                    output.get(
                        "predicted_end_expectation",
                        (end_masked.softmax(dim=1) * positions).sum(dim=1),
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
                start_labels.append(batch["start_labels"].cpu().numpy())
                end_labels.append(batch["end_labels"].cpu().numpy())
                lengths.append(valid_mask.sum(dim=1).cpu().numpy())
        return PredictionBundle(
            detection_logits=np.concatenate(detection_logits),
            detection_labels=np.concatenate(detection_labels),
            family_logits=np.concatenate(family_logits),
            family_labels=np.concatenate(family_labels),
            start_predictions=np.concatenate(start_predictions),
            end_predictions=np.concatenate(end_predictions),
            start_expectations=np.concatenate(start_expectations),
            end_expectations=np.concatenate(end_expectations),
            start_labels=np.concatenate(start_labels),
            end_labels=np.concatenate(end_labels),
            lengths=np.concatenate(lengths),
        )

    def evaluate(self, split_name: str, max_length: int | None = None) -> dict[str, float]:
        split = self.splits[split_name]
        predictions = self._predict(
            split, max_length if max_length is not None else self.config.data.eval_max_length
        )
        probabilities = 1.0 / (1.0 + np.exp(-predictions.detection_logits))
        detection_valid = predictions.detection_labels >= 0
        detection_labels = predictions.detection_labels[detection_valid]
        detection_logits = predictions.detection_logits[detection_valid]
        detection_probabilities = probabilities[detection_valid]
        negatives = detection_probabilities[detection_labels == 0]
        if len(negatives):
            threshold_01 = threshold_at_fpr(negatives, 1e-3)
            threshold_1 = threshold_at_fpr(negatives, 1e-2)
        else:
            threshold_01 = threshold_1 = 0.5
        metrics = detection_metrics(
            detection_labels,
            detection_logits,
            threshold_01,
            threshold_1,
        )
        family_mask = predictions.family_labels >= 0
        metrics.update(
            family_metrics(
                predictions.family_logits[family_mask],
                predictions.family_labels[family_mask],
            )
        )
        start_points = self._apply_endpoint_point_calibration(
            predictions.start_predictions, predictions.start_expectations, "start"
        )
        end_points = self._apply_endpoint_point_calibration(
            predictions.end_predictions, predictions.end_expectations, "end"
        )
        boundary = boundary_prediction_metrics(
            start_points,
            end_points,
            predictions.start_labels,
            predictions.end_labels,
        )
        metrics.update(boundary)
        metrics.update(
            interval_prediction_metrics(
                start_points,
                end_points,
                predictions.start_labels,
                predictions.end_labels,
                predictions.lengths,
            )
        )
        metrics["boundary_mae_mean"] = 0.5 * (
            metrics["start_mae"] + metrics["end_mae"]
        )
        metrics["boundary_minus_mae"] = -metrics["boundary_mae_mean"]
        metrics["detection_auprc_p10"] = metrics["auprc_p10"]
        metrics["detection_recall_fpr_1e_3"] = metrics["recall_fpr_1e_3"]
        metrics["family_macro_f1"] = metrics["macro_f1"]
        metrics["boundary_both_e10"] = metrics["both_e10"]
        metrics["selection_score"] = selection_score(
            metrics, dict(self.config.train.selection)
        )
        return metrics

    def _apply_endpoint_point_calibration(
        self,
        predicted: np.ndarray,
        expected: np.ndarray,
        prefix: str,
    ) -> np.ndarray:
        method = self.calibration.get(f"{prefix}_decode_method", "argmax")
        values = predicted if method == "argmax" else np.rint(expected)
        bias = float(self.calibration.get(f"{prefix}_bias", 0.0))
        return values - bias

    def _calibrate(self) -> dict[str, float]:
        predictions = self._predict(self.splits["calibration"], self.config.data.eval_max_length)
        probabilities = 1.0 / (1.0 + np.exp(-predictions.detection_logits))
        calibration = {
            "detection_threshold_fpr_1e_3": threshold_at_fpr(
                probabilities[predictions.detection_labels == 0], 1e-3
            ),
            "detection_threshold_fpr_1e_2": threshold_at_fpr(
                probabilities[predictions.detection_labels == 0], 1e-2
            ),
        }
        family_mask = predictions.family_labels >= 0
        calibration["family_temperature"] = fit_temperature(
            predictions.family_logits[family_mask], predictions.family_labels[family_mask]
        )
        scaled = predictions.family_logits / calibration["family_temperature"]
        scaled -= scaled.max(axis=1, keepdims=True)
        family_probability = np.exp(scaled)
        family_probability /= family_probability.sum(axis=1, keepdims=True)
        calibration["family_abstain_threshold"] = float(
            np.quantile(family_probability[family_mask].max(axis=1), 0.2)
        )
        start_valid = predictions.start_labels >= 0
        end_valid = predictions.end_labels >= 0
        point_calibration = self._calibrate_endpoint_points(
                predictions.start_predictions,
                predictions.start_expectations,
                predictions.end_predictions,
                predictions.end_expectations,
                predictions.start_labels,
                predictions.end_labels,
                start_valid,
                end_valid,
        )
        calibration.update(point_calibration)
        calibrated_start = self._apply_calibration_values(
            predictions.start_predictions,
            predictions.start_expectations,
            point_calibration,
            "start",
        )
        calibrated_end = self._apply_calibration_values(
            predictions.end_predictions,
            predictions.end_expectations,
            point_calibration,
            "end",
        )
        offsets = fit_endpoint_interval_offsets(
            calibrated_start[start_valid] - predictions.start_labels[start_valid],
            calibrated_end[end_valid] - predictions.end_labels[end_valid],
            coverage=0.9,
        )
        calibration.update(offsets)
        return calibration

    @staticmethod
    def _apply_calibration_values(
        predicted: np.ndarray,
        expected: np.ndarray,
        calibration: dict[str, float | str],
        prefix: str,
    ) -> np.ndarray:
        method = calibration[f"{prefix}_decode_method"]
        values = predicted if method == "argmax" else np.rint(expected)
        return values - float(calibration[f"{prefix}_bias"])

    def _calibrate_endpoint_points(
        self,
        start_predictions: np.ndarray,
        start_expectations: np.ndarray,
        end_predictions: np.ndarray,
        end_expectations: np.ndarray,
        start_labels: np.ndarray,
        end_labels: np.ndarray,
        start_valid: np.ndarray,
        end_valid: np.ndarray,
    ) -> dict[str, float | str]:
        rng = np.random.default_rng(self.config.seed)

        def fit_choice(predicted: np.ndarray, labels: np.ndarray, valid: np.ndarray, prefix: str):
            indices = np.flatnonzero(valid)
            if len(indices) < 4:
                return {
                    f"{prefix}_decode_method": "argmax",
                    f"{prefix}_bias": 0.0,
                    f"{prefix}_calibration_mae": float("nan"),
                }
            rng.shuffle(indices)
            halves = [indices[::2], indices[1::2]]
            choices = []
            for method, values in (
                ("argmax", predicted),
                ("soft", np.rint(start_expectations if prefix == "start" else end_expectations)),
            ):
                errors = []
                for fit, check in (halves, halves[::-1]):
                    bias = float(np.median(values[fit] - labels[fit]))
                    corrected = values[check] - bias
                    errors.append(float(np.abs(corrected - labels[check]).mean()))
                choices.append((float(np.mean(errors)), method))
            _, method = min(choices)
            values = predicted if method == "argmax" else np.rint(
                start_expectations if prefix == "start" else end_expectations
            )
            bias = float(np.median(values[valid] - labels[valid]))
            mae = float(np.abs(values[valid] - bias - labels[valid]).mean())
            return {
                f"{prefix}_decode_method": method,
                f"{prefix}_bias": bias,
                f"{prefix}_calibration_mae": mae,
            }

        result: dict[str, float | str] = {}
        result.update(fit_choice(start_predictions, start_labels, start_valid, "start"))
        result.update(fit_choice(end_predictions, end_labels, end_valid, "end"))
        return result

    def train(self, resume: str | None = None, limit_steps: int | None = None) -> None:
        start_epoch = 0
        start_batch = 0
        if resume:
            start_epoch, start_batch = self.resume(resume)
        total_rows = len(self.splits["train"])
        batch_plans = []
        for epoch in range(self.config.train.epochs):
            plan_rng = np.random.default_rng(self.config.seed + epoch)
            batch_plans.append(
                plan_batches(
                    self.splits["train"],
                    range(total_rows),
                    self.config.train.batch_size,
                    self.config.train.max_tokens,
                    self.config.data.max_length,
                    training=True,
                    rng=plan_rng,
                )
            )
        steps_by_epoch = [
            math.ceil(len(batches) / self.config.train.grad_accum)
            for batches in batch_plans
        ]
        max_steps = sum(steps_by_epoch)
        self._log(
            {
                "event": "train_plan",
                "max_steps": max_steps,
                "steps_by_epoch": steps_by_epoch,
                "batches_by_epoch": [len(batches) for batches in batch_plans],
                "grad_accum": self.config.train.grad_accum,
            }
        )
        started = time.monotonic()
        last_epoch = start_epoch
        for epoch in range(start_epoch, self.config.train.epochs):
            last_epoch = epoch
            if epoch == start_epoch and self.resume_sample_rng_state is not None:
                self.sample_rng = np.random.default_rng()
                self.sample_rng.bit_generator.state = self.resume_sample_rng_state
            else:
                self.sample_rng = np.random.default_rng(
                    self.config.seed + epoch + 1_000_003
                )
            batches = batch_plans[epoch]
            self.model.train()
            epoch_batch_start = start_batch if epoch == start_epoch else 0
            for batch_index in range(
                epoch_batch_start,
                len(batches),
                self.config.train.grad_accum,
            ):
                group = batches[
                    batch_index : batch_index + self.config.train.grad_accum
                ]
                if not group:
                    break
                batch_cursor = min(
                    batch_index + self.config.train.grad_accum, len(batches)
                )
                total_samples = sum(len(item) for item in group)
                self.optimizer.zero_grad(set_to_none=True)
                aggregate: dict[str, float] = {}
                for batch_indices in group:
                    samples = [
                        self.splits["train"].sample(
                            index,
                            self.config.data.max_length,
                            True,
                            self.sample_rng,
                        )
                        for index in batch_indices
                    ]
                    batch = move_batch(collate(samples), self.device)
                    with self._amp_context():
                        output = self.model(batch)
                        loss, parts = multi_task_loss(output, batch, self.config)
                    scaled_loss = loss * (len(samples) / total_samples)
                    scaled_loss.backward()
                    for key, value in parts.items():
                        aggregate[key] = aggregate.get(key, 0.0) + value * len(samples)
                if not math.isfinite(float(aggregate.get("loss", float("inf")))):
                    self.optimizer.zero_grad(set_to_none=True)
                    self._log(
                        {
                            "event": "nonfinite_skip",
                            "epoch": epoch,
                            "batch": batch_index,
                        }
                    )
                    continue
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optim.grad_clip
                )
                self._set_lr(self.global_step, max_steps)
                self.optimizer.step()
                self.global_step += 1
                if self.global_step % 50 == 0 and self.device.type == "cuda":
                    torch.cuda.empty_cache()
                if self.global_step % self.config.train.log_every == 0:
                    averaged = {
                        key: value / max(total_samples, 1)
                        for key, value in aggregate.items()
                        if key != "loss"
                    }
                    self._log(
                        {
                            "event": "train",
                            "epoch": epoch,
                            "batch": batch_index + len(group) - 1,
                            "batch_cursor": batch_cursor,
                            "step": self.global_step,
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "grad_norm": float(gradient_norm),
                            "seconds": round(time.monotonic() - started, 1),
                            **averaged,
                        }
                    )
                if self.global_step % self.config.train.save_every == 0:
                    self._save_last(epoch, batch_cursor)
                if limit_steps is not None and self.global_step >= limit_steps:
                    self._save_last(epoch, batch_cursor)
                    return
                start_batch = 0

            try:
                del batch, output, loss, parts, gradient_norm, samples
            except NameError:
                pass
            self.optimizer.zero_grad(set_to_none=True)
            if self.device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
            dev = self.evaluate("dev")
            dev["event"] = "eval"
            dev["epoch"] = epoch
            dev["step"] = self.global_step
            self._log(dev)
            improved = dev["selection_score"] > self.best_score + self.config.train.min_delta
            self._save_epoch(epoch, len(batches), dev)
            if improved:
                self.best_score = dev["selection_score"]
                self.best_metrics = dev
                self.epoch_without_improvement = 0
                self._save_best(epoch, len(batches))
            else:
                self.epoch_without_improvement += 1
            self._save_last(epoch, len(batches))
            if (
                self.config.train.patience
                and self.epoch_without_improvement >= self.config.train.patience
            ):
                self._log({"event": "early_stop", "epoch": epoch, "step": self.global_step})
                break

        self._save_last(last_epoch, 0)
        best_path = self.run_dir / "best.pt"
        if best_path.exists():
            best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(best_payload["model_state"])
        self.calibration = self._calibrate()
        _write_json(self.run_dir / "calibration.json", self.calibration)
        if best_path.exists():
            best_payload["calibration"] = self.calibration
            _atomic_torch_save(best_payload, best_path)
        calibrated_dev = self.evaluate("dev")
        _write_json(
            self.run_dir / "summary.json",
            {
                "config_hash": self.config.hash,
                "global_step": self.global_step,
                "best_score": self.best_score,
                "best_metrics": self.best_metrics,
                "calibration": self.calibration,
                "calibrated_dev": calibrated_dev,
                "wall_seconds": round(time.monotonic() - started, 1),
                "selected_checkpoint": str(best_path) if best_path.exists() else "last.pt",
                "data": {
                    split: self.splits[split].modality_coverage() for split in self.splits
                },
            },
        )
