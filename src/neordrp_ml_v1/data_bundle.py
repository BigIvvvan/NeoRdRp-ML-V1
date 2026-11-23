"""Download, verify, and extract the public V1 training-data bundle."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path

DEFAULT_REPO_ID = "BigIvvvan/NeoRdRp-ML-V1"
DEFAULT_BUNDLE_FILE = "data/neordrp_ml_v1_training_bundle.tar.gz"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_hash(project_root: Path) -> str | None:
    manifest_path = (
        project_root
        / "src"
        / "neordrp_ml_v1"
        / "assets"
        / "training_data_manifest.json"
    )
    if not manifest_path.is_file():
        return None
    value = str(json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"])
    return None if value.startswith("pending:") else value


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not permitted: {member.name}")
        handle.extractall(destination, filter="data")


def download_training_bundle(
    project_root: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    bundle_file: str = DEFAULT_BUNDLE_FILE,
    force: bool = False,
) -> Path:
    """Download and extract the processed training data used by V1."""
    root = Path(project_root).expanduser().resolve()
    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    expected = _expected_hash(root)
    marker = root / "data" / ".v1_training_bundle.sha256"
    if marker.is_file() and not force:
        current = marker.read_text(encoding="utf-8").strip()
        if current and (expected is None or current == expected):
            print(f"Training bundle already verified at {marker}")
            return marker

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface-hub to download training data") from exc
    cached = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=bundle_file,
            token=os.environ.get("HF_TOKEN"),
            local_dir=downloads,
        )
    )
    digest = sha256_file(cached)
    if expected is not None and digest != expected:
        raise ValueError(
            f"Training bundle SHA-256 mismatch: expected {expected}, received {digest}"
        )
    _safe_extract(cached, root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(digest + "\n", encoding="utf-8")
    print(f"Verified training bundle {digest}")
    return marker
