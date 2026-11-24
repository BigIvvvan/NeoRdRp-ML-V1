#!/usr/bin/env python
"""Strip optimizer state from a trained V1 checkpoint for public inference."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = torch.load(args.input, map_location="cpu", weights_only=False)
    payload = {
        "format_version": 1,
        "model_name": "NeoRdRp-ML-V1",
        "source_checkpoint": args.input.name,
        "source_checkpoint_sha256": sha256_file(args.input),
        "config": {"model": source["config"]["model"]},
        "model_state": source["model_state"],
        "family_names": source["family_names"],
        "calibration": source.get("calibration", {}),
        "dev_metrics": source.get("dev_metrics", source.get("best_metrics", {})),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"model_sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
