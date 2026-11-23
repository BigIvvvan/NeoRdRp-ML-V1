"""End-to-end inference from a pooled feature bundle to V1 predictions."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ModelConfig
from .model import PAD_ID, NeoRdRpV1, tokenize

DEFAULT_REPO_ID = "BigIvvvan/NeoRdRp-ML-V1"
DEFAULT_MODEL_FILE = "models/neordrp_ml_v1.pt"


@dataclass(frozen=True)
class Prediction:
    protein_id: str
    sequence_length: int
    window_start: int
    window_end: int
    rdrp_probability: float
    rdrp_called: bool
    family: str
    family_probability: float
    family_abstained: bool
    qstart: int
    qend: int
    start_interval_low: int
    start_interval_high: int
    end_interval_low: int
    end_interval_high: int
    checkpoint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Read a small or medium protein FASTA without an extra Biopython dependency."""
    records: list[tuple[str, str]] = []
    identifier: str | None = None
    fragments: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    records.append((identifier, "".join(fragments).upper()))
                identifier = line[1:].split(maxsplit=1)[0]
                if not identifier:
                    raise ValueError(f"Empty FASTA identifier in {path}")
                fragments = []
            elif identifier is None:
                raise ValueError(f"Sequence data precedes the first FASTA header in {path}")
            else:
                fragments.append(line.replace(" ", ""))
    if identifier is not None:
        records.append((identifier, "".join(fragments).upper()))
    if not records:
        raise ValueError(f"No protein records found in {path}")
    ids = [record[0] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("FASTA identifiers must be unique")
    return records


def _resolve_checkpoint(
    checkpoint: str | Path | None,
    repo_id: str,
    model_file: str,
) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "Install huggingface-hub or provide --checkpoint explicitly"
            ) from exc
        token = os.environ.get("HF_TOKEN")
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=model_file,
                token=token,
            )
        ).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _center_window(length: int, maximum: int) -> tuple[int, int]:
    if length <= maximum:
        return 0, length
    start = (length - maximum) // 2
    return start, start + maximum


def _load_features(
    path: str | Path,
    protein_ids: list[str],
    esm_dim: int,
    protrek_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    try:
        feature_ids = payload["protein_ids"].astype(str).tolist()
        if feature_ids != protein_ids:
            raise ValueError("Feature bundle and FASTA identifiers/order differ")
        esm = np.asarray(payload["esm"], dtype=np.float32)
        protrek = np.asarray(payload["protrek"], dtype=np.float32)
        esm_mask = np.asarray(payload["esm_mask"], dtype=bool)
        protrek_mask = np.asarray(payload["protrek_mask"], dtype=bool)
    finally:
        payload.close()
    expected = (len(protein_ids), esm_dim)
    if esm.shape != expected or protrek.shape != (len(protein_ids), protrek_dim):
        raise ValueError(
            f"Feature dimensions are incompatible: ESM {esm.shape}, "
            f"ProTrek {protrek.shape}"
        )
    if esm_mask.shape != (len(protein_ids),) or protrek_mask.shape != (
        len(protein_ids),
    ):
        raise ValueError("Feature masks are incompatible with the identifier list")
    for name, values, mask in (
        ("ESM-C", esm, esm_mask),
        ("ProTrek", protrek, protrek_mask),
    ):
        if not np.isfinite(values[mask]).all():
            raise ValueError(f"{name} contains non-finite values")
    if not (esm_mask | protrek_mask).all():
        raise ValueError("Every sequence must have at least one protein-language model feature")
    return esm, esm_mask, protrek, protrek_mask


class NeoRdRpV1Predictor:
    """Load a frozen V1 checkpoint and predict one or more protein sequences."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str = "auto",
        repo_id: str = DEFAULT_REPO_ID,
        model_file: str = DEFAULT_MODEL_FILE,
        expected_sha256: str | None = None,
        max_length: int = 4096,
    ):
        self.checkpoint_path = _resolve_checkpoint(checkpoint, repo_id, model_file)
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        if expected_sha256 and self.checkpoint_sha256 != expected_sha256:
            raise ValueError(
                f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, "
                f"received {self.checkpoint_sha256}"
            )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(device)
        self.max_length = int(max_length)

        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.model_config = ModelConfig.from_dict(payload["config"]["model"])
        self.family_names = [str(value) for value in payload["family_names"]]
        self.calibration = dict(payload.get("calibration", {}))
        self.model = NeoRdRpV1(self.model_config, len(self.family_names))
        incompatible = self.model.load_state_dict(payload["model_state"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(f"Checkpoint is incompatible: {incompatible}")
        self.model.to(self.device).eval()

    def predict(
        self,
        sequences: Iterable[tuple[str, str]],
        features: str | Path,
    ) -> list[Prediction]:
        records = [(str(identifier), str(sequence).strip().upper()) for identifier, sequence in sequences]
        if not records:
            raise ValueError("No sequences were supplied")
        protein_ids = [record[0] for record in records]
        if len(protein_ids) != len(set(protein_ids)):
            raise ValueError("Protein identifiers must be unique")
        esm, esm_mask, protrek, protrek_mask = _load_features(
            features,
            protein_ids,
            self.model_config.esm_dim,
            self.model_config.protrek_dim,
        )

        lengths = np.asarray([len(sequence) for _, sequence in records], dtype=np.int64)
        maximum = min(self.max_length, int(lengths.max()))
        windows = [_center_window(int(length), maximum) for length in lengths]
        input_ids = torch.full((len(records), maximum), PAD_ID, dtype=torch.long)
        attention_mask = torch.zeros((len(records), maximum), dtype=torch.bool)
        for row, ((_, sequence), (start, end)) in enumerate(zip(records, windows)):
            token_ids = torch.tensor(tokenize(sequence[start:end]), dtype=torch.long)
            input_ids[row, : len(token_ids)] = token_ids
            attention_mask[row, : len(token_ids)] = True
        batch = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "esm": torch.from_numpy(esm).to(self.device),
            "esm_mask": torch.from_numpy(esm_mask).to(self.device),
            "protrek": torch.from_numpy(protrek).to(self.device),
            "protrek_mask": torch.from_numpy(protrek_mask).to(self.device),
        }
        with torch.inference_mode():
            output = self.model(batch)
        probabilities = torch.sigmoid(output["detection_logits"].float()).cpu().numpy()
        family_logits = output["family_logits"].float().cpu() / float(
            self.calibration.get("family_temperature", 1.0)
        )
        family_probabilities = family_logits.softmax(dim=-1).numpy()
        start_logits = output["start_logits"].float().masked_fill(
            ~batch["attention_mask"], -1e4
        )
        end_logits = output["end_logits"].float().masked_fill(
            ~batch["attention_mask"], -1e4
        )
        positions = torch.arange(maximum, device=self.device).float()
        start_argmax = start_logits.argmax(dim=1).cpu().numpy()
        end_argmax = end_logits.argmax(dim=1).cpu().numpy()
        start_expectation = (
            start_logits.softmax(dim=1) * positions
        ).sum(dim=1).cpu().numpy()
        end_expectation = (end_logits.softmax(dim=1) * positions).sum(dim=1).cpu().numpy()

        threshold = float(self.calibration["detection_threshold_fpr_1e_3"])
        abstain_threshold = float(self.calibration["family_abstain_threshold"])
        predictions: list[Prediction] = []
        for row, (protein_id, _) in enumerate(records):
            start = int(
                self._decode_endpoint(
                    start_argmax[row], start_expectation[row], "start", windows[row][0]
                )
            )
            end = int(
                self._decode_endpoint(
                    end_argmax[row], end_expectation[row], "end", windows[row][0]
                )
            )
            start = min(max(1, start), int(lengths[row]))
            end = min(max(1, end), int(lengths[row]))
            if end <= start:
                start = min(int(lengths[row]) - 1, max(1, start))
                end = min(int(lengths[row]), start + 1)
            family_index = int(family_probabilities[row].argmax())
            family_probability = float(family_probabilities[row, family_index])
            abstained = family_probability < abstain_threshold
            predictions.append(
                Prediction(
                    protein_id=protein_id,
                    sequence_length=int(lengths[row]),
                    window_start=windows[row][0] + 1,
                    window_end=windows[row][1],
                    rdrp_probability=float(probabilities[row]),
                    rdrp_called=bool(probabilities[row] >= threshold),
                    family="unknown" if abstained else self.family_names[family_index],
                    family_probability=family_probability,
                    family_abstained=bool(abstained),
                    qstart=start,
                    qend=end,
                    start_interval_low=max(
                        1,
                        round(start + float(self.calibration.get("start_low", -10.0))),
                    ),
                    start_interval_high=min(
                        int(lengths[row]),
                        round(start + float(self.calibration.get("start_high", 10.0))),
                    ),
                    end_interval_low=max(
                        1,
                        round(end + float(self.calibration.get("end_low", -10.0))),
                    ),
                    end_interval_high=min(
                        int(lengths[row]),
                        round(end + float(self.calibration.get("end_high", 10.0))),
                    ),
                    checkpoint_sha256=self.checkpoint_sha256,
                )
            )
        return predictions

    def predict_fasta(self, fasta: str | Path, features: str | Path) -> list[Prediction]:
        return self.predict(read_fasta(fasta), features)

    def _decode_endpoint(
        self,
        argmax: int,
        expectation: float,
        prefix: str,
        window_start: int,
    ) -> int:
        method = self.calibration.get(f"{prefix}_decode_method", "argmax")
        value = float(argmax) if method == "argmax" else float(round(expectation))
        value -= float(self.calibration.get(f"{prefix}_bias", 0.0))
        return round(window_start + value + 1)


def predictions_to_json(
    predictions: list[Prediction],
    checkpoint: str | Path,
    features: str | Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": "NeoRdRp-ML V1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "features": str(features),
        "features_sha256": sha256_file(features),
        "coordinate_system": "1-based closed",
        "predictions": [prediction.to_dict() for prediction in predictions],
    }


def write_json(payload: dict[str, Any], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
