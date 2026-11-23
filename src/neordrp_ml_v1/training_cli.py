"""Command-line entry point for reproducible GPU training."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from .config import load_config
from .data import build_cohort, collate, move_batch, plan_batches
from .losses import multi_task_loss
from .model import NeoRdRpV1
from .trainer import Trainer


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _dry_run(config, splits, family_names: list[str]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only training dry run requested without CUDA")
    model = NeoRdRpV1(config.model, len(family_names)).to("cuda")
    split = splits["train"]
    indices = list(range(len(split)))[: min(len(split), config.train.batch_size)]
    batch_indices = plan_batches(
        split,
        indices,
        config.train.batch_size,
        config.train.max_tokens,
        config.data.max_length,
        training=False,
    )[0]
    samples = [split.sample(index, config.data.max_length, False) for index in batch_indices]
    batch = move_batch(collate(samples), torch.device("cuda"))
    dtype = torch.bfloat16 if config.train.amp == "bf16" else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=config.train.amp != "none"):
        output = model(batch)
        loss, parts = multi_task_loss(output, batch, config)
    if not torch.isfinite(loss):
        raise FloatingPointError("Dry-run loss is not finite")
    loss.backward()
    print(
        json.dumps(
            {
                "dry_run": True,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "batch_size": len(batch_indices),
                "losses": parts,
                "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
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
    args = parser.parse_args(argv)

    config = load_config(args.config)
    root = config.resolve_root()
    updates = {}
    for key in ("init_from", "output_dir", "device", "batch_size", "grad_accum", "max_tokens"):
        value = getattr(args, key)
        if value is not None:
            updates[key] = value
    if updates:
        config = dataclasses.replace(
            config, train=dataclasses.replace(config.train, **updates)
        )
    for value in (args.batch_size, args.grad_accum, args.max_tokens):
        if value is not None and value < 1:
            parser.error("Batch, accumulation, and token limits must be positive")
    if args.limit_steps is not None and args.limit_steps < 1:
        parser.error("--limit-steps must be positive")

    splits, family_names = build_cohort(config.data)
    coverage = {
        split: splits[split].modality_coverage() for split in ("train", "dev", "calibration")
    }
    print(json.dumps({"coverage": coverage, "families": len(family_names)}, indent=2))
    if args.dry_run:
        _dry_run(config, splits, family_names)
        return 0

    resume_path = _resolve(root, args.resume)
    trainer = Trainer(config, splits, family_names, resume_path=resume_path)
    trainer.train(
        resume=str(resume_path) if resume_path is not None else None,
        limit_steps=args.limit_steps,
    )
    print(json.dumps({"run_dir": str(trainer.run_dir), "complete": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
