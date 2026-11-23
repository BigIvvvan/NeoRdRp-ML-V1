#!/usr/bin/env python
"""Download the released data and reproduce the two-stage V1 training run."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neordrp_ml_v1.data_bundle import DEFAULT_REPO_ID, download_training_bundle
from neordrp_ml_v1.training_cli import main as train_main


def _install_training_extras() -> None:
    required = ("pandas", "pyarrow", "sklearn", "yaml", "torch")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        command = [sys.executable, "-m", "pip", "install", "-e", f"{ROOT}[train]"]
        print("Installing training dependencies:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--limit-steps", type=int)
    args = parser.parse_args()

    _install_training_extras()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to reproduce V1 training")
    download_training_bundle(
        ROOT, repo_id=args.repo_id, force=args.force_download
    )

    if not args.skip_stage1:
        command = ["--config", str(ROOT / "configs/train_stage1.yaml")]
        if args.limit_steps:
            command.extend(["--limit-steps", str(args.limit_steps)])
        train_main(command)
    if not args.skip_stage2:
        command = ["--config", str(ROOT / "configs/train_stage2.yaml")]
        if args.limit_steps:
            command.extend(["--limit-steps", str(args.limit_steps)])
        train_main(command)
    print("V1 training reproduction completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
