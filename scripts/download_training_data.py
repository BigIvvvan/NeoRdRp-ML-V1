#!/usr/bin/env python
"""Download the processed NeoRdRp-ML V1 training data from Hugging Face."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neordrp_ml_v1.data_bundle import DEFAULT_BUNDLE_FILE, DEFAULT_REPO_ID, download_training_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--bundle-file", default=DEFAULT_BUNDLE_FILE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_training_bundle(
        args.project_root,
        repo_id=args.repo_id,
        bundle_file=args.bundle_file,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
