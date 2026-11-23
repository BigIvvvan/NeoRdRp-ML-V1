#!/usr/bin/env python
"""Train the NeoRdRp-ML V1 model from a YAML recipe."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from neordrp_ml_v1.config import load_config
from neordrp_ml_v1.data import build_cohort, collate, move_batch, plan_batches
from neordrp_ml_v1.losses import multi_task_loss
from neordrp_ml_v1.model import NeoRdRpV1
from neordrp_ml_v1.trainer import Trainer


def _paths(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def dry_run(config, splits, family_names: list[str], longest: bool = False) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only training requested, but CUDA is unavailable")
    device = torch.device("cuda")
    model = NeoRdRpV1(config.model, len(family_names)).to(device)
    split = splits["train"]
    ordered = sorted(
        range(len(split)),
        key=lambda index: int(split.lengths[index]),
    )
    if longest:
        selected = ordered[-max(config.train.batch_size * 4, 64) :]
    else:
        width = max(config.train.batch_size * 4, 64)
        middle = len(ordered) // 2
        selected = ordered[max(0, middle - width // 2) : middle + width // 2]
    batch_indices = plan_batches(
        split,
        selected,
        config.train.batch_size,
        config.train.max_tokens,
        config.data.max_length,
        training=False,
    )[0]
    batch = move_batch(
        collate([split.sample(index, config.data.max_length, False) for index in batch_indices]),
        device,
    )
    dtype = torch.bfloat16 if config.train.amp == "bf16" else torch.float16
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda" and config.train.amp != "none",
    ):
        output = model(batch)
        loss, parts = multi_task_loss(output, batch, config)
    if not torch.isfinite(loss):
        raise FloatingPointError("Dry-run loss is not finite")
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    step_seconds = time.perf_counter() - started
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        json.dumps(
            {
                "dry_run": True,
                "device": str(device),
                "parameters": parameters,
                "batch_size": len(batch_indices),
                "max_batch_length": int(batch["input_ids"].shape[1]),
                "losses": parts,
                "fwd_bwd_seconds": round(step_seconds, 3),
                "sequences_per_second": round(len(batch_indices) / step_seconds, 2),
                "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)
                if device.type == "cuda"
                else None,
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--init-from")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=["cuda"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--limit-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-longest", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    root = Path(config.data.project_root).expanduser().resolve()
    loop_updates = {}
    if args.init_from:
        loop_updates["init_from"] = args.init_from
    if args.output_dir:
        loop_updates["output_dir"] = args.output_dir
    if args.device:
        loop_updates["device"] = args.device
    if args.batch_size:
        loop_updates["batch_size"] = args.batch_size
    if args.grad_accum:
        loop_updates["grad_accum"] = args.grad_accum
    if args.max_tokens:
        loop_updates["max_tokens"] = args.max_tokens
    if loop_updates:
        config = dataclasses.replace(config, train=dataclasses.replace(config.train, **loop_updates))
    if any(
        value is not None and value < 1
        for value in (args.batch_size, args.grad_accum, args.max_tokens)
    ):
        parser.error("batch, accumulation, and token limits must be positive")
    if args.limit_steps is not None and args.limit_steps < 1:
        parser.error("--limit-steps must be positive")

    splits, family_names = build_cohort(config.data)
    summary = {
        split: splits[split].modality_coverage() for split in ("train", "dev", "calibration")
    }
    print(json.dumps({"data": summary, "families": len(family_names)}, indent=2), flush=True)
    if args.dry_run:
        dry_run(config, splits, family_names, longest=args.dry_run_longest)
        return 0

    resume_path = _paths(root, args.resume)
    trainer = Trainer(config, splits, family_names, resume_path=resume_path)
    trainer.train(
        resume=str(resume_path) if resume_path is not None else None,
        limit_steps=args.limit_steps,
    )
    print(json.dumps({"run_dir": str(trainer.run_dir), "complete": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
