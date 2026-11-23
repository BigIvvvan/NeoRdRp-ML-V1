"""Data assembly and batching from the frozen NeoRdRp-ML split files."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from .config import DataConfig
from .model import PAD_ID, tokenize

SPLITS = ("train", "dev", "calibration")


@dataclass
class Record:
    protein_id: str
    sequence: str
    length: int
    detection_label: int
    family: str | None
    cluster_id: str
    split: str
    boundary_start: int | None = None
    boundary_end: int | None = None
    boundary_weight: float = 0.0
    teacher_start: int | None = None
    teacher_end: int | None = None
    teacher_weight: float = 0.0


def _safe_existing(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    lowered = str(resolved).lower()
    if "test_lockbox" in lowered or "/lockbox/" in lowered.replace("\\", "/"):
        raise PermissionError(f"Training data loader refuses lockbox path: {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


class TeacherStore:
    """Pooled teacher features and teacher logits keyed by protein_id."""

    def __init__(
        self,
        dimension: int,
        name: str,
        expected_window: int = 2048,
        allow_legacy_window: bool = False,
    ):
        self.dimension = int(dimension)
        self.name = name
        self.expected_window = int(expected_window)
        self.allow_legacy_window = bool(allow_legacy_window)
        self.features: dict[str, np.ndarray] = {}
        self.logits: dict[str, float] = {}
        self.status: dict[str, str] = {}
        self.source_rows: dict[str, int] = {}
        self.source_windows: dict[str, int | None] = {}

    def read(self, sources: Iterable[str], root: Path) -> None:
        for source_name in sources:
            path = _safe_existing(root / source_name)
            parquet = pq.ParquetFile(path)
            metadata = parquet.metadata.metadata or {}
            raw_metadata = metadata.get(b"neordrp_teacher", metadata.get("neordrp_teacher"))
            source_window = None
            if raw_metadata:
                if isinstance(raw_metadata, bytes):
                    raw_metadata = raw_metadata.decode("utf-8")
                source_window = json.loads(raw_metadata).get("max_len")
            self.source_windows[str(path)] = source_window
            if (
                self.expected_window
                and source_window is not None
                and int(source_window) != self.expected_window
                and not self.allow_legacy_window
            ):
                raise ValueError(
                    f"{path} uses embedding window {source_window}, expected "
                    f"{self.expected_window}; regenerate the cache or explicitly "
                    "set data.allow_legacy_embedding_window"
                )
            names = set(parquet.schema_arrow.names)
            if not {"protein_id", "pooled"} <= names:
                raise ValueError(f"{path} lacks protein_id/pooled columns")
            has_status = "status" in names
            has_logit = "logit" in names
            columns = ["protein_id", "pooled"]
            if has_status:
                columns.append("status")
            if has_logit:
                columns.append("logit")
            added = 0
            for batch in parquet.iter_batches(batch_size=4096, columns=columns):
                ids = batch.column("protein_id").to_pylist()
                pooled = batch.column("pooled")
                if pooled.null_count or pooled.values.null_count:
                    raise ValueError(f"Null pooled feature in {path}")
                matrix = pooled.values.to_numpy(zero_copy_only=False)
                if matrix.size != len(ids) * self.dimension:
                    raise ValueError(
                        f"Expected {self.dimension} features in {path}, got shape {matrix.shape}"
                    )
                matrix = matrix.reshape(len(ids), self.dimension).astype(np.float16, copy=False)
                statuses = batch.column("status").to_pylist() if has_status else ["ok"] * len(ids)
                logits = batch.column("logit").to_pylist() if has_logit else [float("nan")] * len(ids)
                for protein_id, feature, status, logit in zip(ids, matrix, statuses, logits):
                    if not protein_id or status != "ok" or not np.isfinite(feature).all():
                        continue
                    previous_logit = self.logits.get(protein_id, float("nan"))
                    new_logit = float(logit) if logit is not None else float("nan")
                    if protein_id not in self.features:
                        self.features[protein_id] = feature.copy()
                        self.logits[protein_id] = new_logit
                        self.status[protein_id] = status
                        self.source_rows[protein_id] = int(parquet.metadata.num_rows)
                        added += 1
                    elif not math.isfinite(previous_logit) and math.isfinite(new_logit):
                        self.logits[protein_id] = new_logit
            print(
                json.dumps(
                    {
                        "teacher": self.name,
                        "source": str(path),
                        "added": added,
                        "cached_total": len(self.features),
                    }
                ),
                flush=True,
            )

    def get(self, protein_id: str) -> np.ndarray | None:
        return self.features.get(protein_id)

    def available_logit(self, protein_id: str) -> float:
        return self.logits.get(protein_id, float("nan"))

    @property
    def with_logit(self) -> int:
        return sum(math.isfinite(value) for value in self.logits.values())


class FamilyTeacherStore:
    """Fixed-order family teacher logits keyed by protein_id."""

    def __init__(self, dimension: int, name: str = "family"):
        self.dimension = int(dimension)
        self.name = name
        self.logits: dict[str, np.ndarray] = {}
        self.source_rows: dict[str, int] = {}

    def read(self, sources: Iterable[str], root: Path) -> None:
        for source_name in sources:
            path = _safe_existing(root / source_name)
            parquet = pq.ParquetFile(path)
            names = set(parquet.schema_arrow.names)
            if "protein_id" not in names:
                raise ValueError(f"{path} lacks protein_id")
            logit_column = next(
                (
                    name
                    for name in (
                        "teacher_family_logits",
                        "family_logits",
                        "teacher_logits",
                    )
                    if name in names
                ),
                None,
            )
            if logit_column is None:
                raise ValueError(
                    f"{path} lacks teacher_family_logits/family_logits/teacher_logits"
                )
            columns = ["protein_id", logit_column]
            has_status = "status" in names
            if has_status:
                columns.append("status")
            added = 0
            for batch in parquet.iter_batches(batch_size=2048, columns=columns):
                ids = batch.column("protein_id").to_pylist()
                values = batch.column(logit_column).to_pylist()
                statuses = batch.column("status").to_pylist() if has_status else ["ok"] * len(ids)
                for protein_id, value, status in zip(ids, values, statuses):
                    if not protein_id or status != "ok" or value is None:
                        continue
                    array = np.asarray(value, dtype=np.float32)
                    if array.shape != (self.dimension,) or not np.isfinite(array).all():
                        continue
                    if protein_id not in self.logits:
                        self.logits[str(protein_id)] = array.copy()
                        self.source_rows[str(protein_id)] = int(parquet.metadata.num_rows)
                        added += 1
            print(
                json.dumps(
                    {
                        "family_teacher": self.name,
                        "source": str(path),
                        "added": added,
                        "cached_total": len(self.logits),
                    }
                ),
                flush=True,
            )

    def get(self, protein_id: str) -> np.ndarray | None:
        return self.logits.get(protein_id)


class ScalarTeacherStore:
    """Finite scalar teacher scores keyed by protein_id."""

    def __init__(self, name: str):
        self.name = name
        self.values: dict[str, float] = {}
        self.source_rows: dict[str, int] = {}

    def read(self, sources: Iterable[str], root: Path) -> None:
        for source_name in sources:
            path = _safe_existing(root / source_name)
            parquet = pq.ParquetFile(path)
            names = set(parquet.schema_arrow.names)
            value_column = next(
                (
                    name
                    for name in (
                        "teacher_logit",
                        "detection_teacher_logit",
                        "raw_detection_logit",
                        "logit",
                    )
                    if name in names
                ),
                None,
            )
            if "protein_id" not in names or value_column is None:
                raise ValueError(
                    f"{path} requires protein_id and a scalar teacher logit column"
                )
            columns = ["protein_id", value_column]
            has_status = "status" in names
            if has_status:
                columns.append("status")
            added = 0
            for batch in parquet.iter_batches(batch_size=8192, columns=columns):
                ids = batch.column("protein_id").to_pylist()
                values = batch.column(value_column).to_pylist()
                statuses = (
                    batch.column("status").to_pylist()
                    if has_status
                    else ["ok"] * len(ids)
                )
                for protein_id, value, status in zip(ids, values, statuses):
                    if not protein_id or status != "ok" or value is None:
                        continue
                    value = float(value)
                    if not math.isfinite(value):
                        continue
                    key = str(protein_id)
                    if key not in self.values:
                        self.values[key] = value
                        self.source_rows[key] = int(parquet.metadata.num_rows)
                        added += 1
            print(
                json.dumps(
                    {
                        "teacher": self.name,
                        "source": str(path),
                        "added": added,
                        "cached_total": len(self.values),
                    }
                ),
                flush=True,
            )

    def get(self, protein_id: str) -> float:
        return self.values.get(protein_id, float("nan"))


class TokenBank:
    def __init__(self, records: list[Record]):
        lengths = np.asarray([len(record.sequence) for record in records], dtype=np.int64)
        total = int(lengths.sum())
        self.flat = np.empty(total, dtype=np.uint8)
        self.offsets = np.empty(len(records) + 1, dtype=np.int64)
        self.offsets[0] = 0
        cursor = 0
        for index, record in enumerate(records):
            values = np.fromiter(tokenize(record.sequence), dtype=np.uint8, count=record.length)
            self.flat[cursor : cursor + record.length] = values
            cursor += record.length
            self.offsets[index + 1] = cursor

    def slice(self, index: int, start: int, length: int) -> np.ndarray:
        source = int(self.offsets[index]) + start
        return self.flat[source : source + length]


class PreparedSplit:
    def __init__(
        self,
        records: list[Record],
        tokens: TokenBank,
        family_names: list[str],
        esm: TeacherStore,
        protrek: TeacherStore,
        family_teacher: FamilyTeacherStore | None = None,
        detection_teacher: ScalarTeacherStore | None = None,
    ):
        self.records = records
        self.tokens = tokens
        self.family_names = family_names
        self.esm = esm
        self.protrek = protrek
        self.family_teacher = family_teacher or FamilyTeacherStore(len(family_names))
        self.detection_teacher = detection_teacher or ScalarTeacherStore("detection")
        self.family_index = {name: index for index, name in enumerate(family_names)}
        self.lengths = np.asarray([record.length for record in records], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.records)

    def modality_coverage(self) -> dict[str, int]:
        ids = {record.protein_id for record in self.records}
        return {
            "rows": len(ids),
            "esm": len(ids & set(self.esm.features)),
            "protrek": len(ids & set(self.protrek.features)),
            "both": len(ids & set(self.esm.features) & set(self.protrek.features)),
            "family_teacher": len(ids & set(self.family_teacher.logits)),
            "detection_teacher": len(ids & set(self.detection_teacher.values)),
            "family_labels": sum(record.family is not None for record in self.records),
            "boundary_labels": sum(record.boundary_start is not None for record in self.records),
            "teacher_boundaries": sum(record.teacher_start is not None for record in self.records),
            "detection_pos": sum(record.detection_label == 1 for record in self.records),
            "detection_neg": sum(record.detection_label == 0 for record in self.records),
        }

    def _window(
        self,
        index: int,
        max_length: int,
        training: bool,
        rng: np.random.Generator | None,
    ) -> tuple[int, int]:
        record = self.records[index]
        if record.length <= max_length:
            return 0, record.length
        margin = min(128, max_length // 8)
        if record.boundary_start is not None and record.boundary_end is not None:
            left = max(0, record.boundary_start - 1 - margin)
            right = min(record.length, record.boundary_end + margin)
            if right - left <= max_length:
                extra = max_length - (right - left)
                left = max(0, left - extra // 2)
                right = min(record.length, left + max_length)
                left = right - max_length
                return left, right
            center = (record.boundary_start - 1 + record.boundary_end) // 2
        else:
            center = record.length // 2
        if training and rng is not None:
            center += int(rng.integers(-max_length // 4, max_length // 4 + 1))
        start = min(max(0, center - max_length // 2), record.length - max_length)
        return start, start + max_length

    def sample(
        self,
        index: int,
        max_length: int,
        training: bool,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        record = self.records[index]
        start, end = self._window(index, max_length, training, rng)
        token_ids = self.tokens.slice(index, start, end - start)
        start_label = end_label = -100
        teacher_start_label = teacher_end_label = -100
        if record.boundary_start is not None and record.boundary_end is not None:
            visible_start = record.boundary_start - 1
            visible_end = record.boundary_end - 1
            if start <= visible_start < end:
                start_label = visible_start - start
            if start <= visible_end < end:
                end_label = visible_end - start
        if record.teacher_start is not None and record.teacher_end is not None:
            visible_teacher_start = record.teacher_start - 1
            visible_teacher_end = record.teacher_end - 1
            if start <= visible_teacher_start < end:
                teacher_start_label = visible_teacher_start - start
            if start <= visible_teacher_end < end:
                teacher_end_label = visible_teacher_end - start
        family_label = (
            self.family_index[record.family] if record.family is not None else -100
        )
        teacher_logits = [
            value
            for value in (self.detection_teacher.get(record.protein_id),)
            if math.isfinite(value)
        ]
        if not teacher_logits:
            teacher_logits = [
                value
                for value in (
                    self.esm.available_logit(record.protein_id),
                    self.protrek.available_logit(record.protein_id),
                )
                if math.isfinite(value)
            ]
        teacher_family = self.family_teacher.get(record.protein_id)
        return {
            "protein_id": record.protein_id,
            "input_ids": torch.from_numpy(token_ids.astype(np.int64, copy=False)),
            "detection_label": float(record.detection_label),
            "family_label": int(family_label),
            "start_label": int(start_label),
            "end_label": int(end_label),
            "boundary_weight": float(record.boundary_weight)
            if record.boundary_start is not None and record.boundary_end is not None
            else 0.0,
            "status_label": -100,
            "teacher_start_label": int(teacher_start_label),
            "teacher_end_label": int(teacher_end_label),
            "teacher_weight": float(record.teacher_weight),
            "cluster_id": record.cluster_id,
            "window_start": start,
            "window_end": end,
            "full_length": record.length,
            "esm": torch.from_numpy(
                self.esm.get(record.protein_id)
                if record.protein_id in self.esm.features
                else np.zeros(self.esm.dimension, dtype=np.float16)
            ),
            "esm_mask": torch.tensor(record.protein_id in self.esm.features),
            "protrek": torch.from_numpy(
                self.protrek.get(record.protein_id)
                if record.protein_id in self.protrek.features
                else np.zeros(self.protrek.dimension, dtype=np.float16)
            ),
            "protrek_mask": torch.tensor(record.protein_id in self.protrek.features),
            "teacher_logit": float(np.mean(teacher_logits)) if teacher_logits else float("nan"),
            "teacher_family_logits": torch.from_numpy(
                teacher_family
                if teacher_family is not None
                else np.zeros(self.family_teacher.dimension, dtype=np.float32)
            ),
            "teacher_family_mask": torch.tensor(teacher_family is not None),
        }


def _read_family(path: Path, split: str) -> list[dict[str, Any]]:
    table = pq.read_table(
        path, columns=["protein_id", "family", "cluster_id", "length", "seq"]
    )
    data = table.to_pydict()
    rows = []
    for protein_id, family, cluster_id, length, sequence in zip(
        data["protein_id"],
        data["family"],
        data["cluster_id"],
        data["length"],
        data["seq"],
    ):
        if not protein_id or not family or not sequence:
            raise ValueError(f"Missing core family row in {path}")
        if len(sequence) != int(length):
            raise ValueError(f"Length mismatch for {protein_id} in {path}")
        rows.append(
            {
                "protein_id": str(protein_id),
                "sequence": str(sequence).upper(),
                "length": int(length),
                "family": str(family),
                "cluster_id": str(cluster_id),
                "detection_label": 1,
                "split": split,
            }
        )
    return rows


def _read_slice(path: Path, split: str, label: int) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["protein_id", "seq", "seq_len"])
    data = table.to_pydict()
    rows = []
    for protein_id, sequence, length in zip(
        data["protein_id"], data["seq"], data["seq_len"]
    ):
        if not protein_id or not sequence or len(sequence) != int(length):
            raise ValueError(f"Invalid slice row in {path}")
        rows.append(
            {
                "protein_id": str(protein_id),
                "sequence": str(sequence).upper(),
                "length": int(length),
                "family": None,
                "cluster_id": str(protein_id),
                "detection_label": label,
                "split": split,
            }
        )
    return rows


def _read_supplementary(path: Path, split: str) -> list[dict[str, Any]]:
    """Read optional split-aware proteins with the common column aliases."""
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    sequence_column = "sequence" if "sequence" in names else "seq"
    length_column = "length" if "length" in names else "seq_len"
    required = {"protein_id", sequence_column, length_column}
    missing = required - names
    if missing:
        raise ValueError(f"{path} lacks supplementary columns: {sorted(missing)}")
    optional = [
        name
        for name in ("family", "cluster_id", "detection_label", "label", "split")
        if name in names
    ]
    frame = pq.read_table(
        path, columns=sorted(required | set(optional))
    ).to_pandas()
    if "split" in frame:
        frame = frame[frame["split"].astype(str) == split]
    elif split != "train":
        return []
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        protein_id = str(row.protein_id)
        sequence = str(getattr(row, sequence_column)).upper()
        length = int(getattr(row, length_column))
        if not protein_id or not sequence or len(sequence) != length:
            raise ValueError(f"Invalid supplementary row for {protein_id} in {path}")
        family_value = getattr(row, "family", None)
        family = None if family_value is None or pd.isna(family_value) else str(family_value)
        cluster_value = getattr(row, "cluster_id", protein_id)
        if "detection_label" in names:
            detection_label = int(row.detection_label)
        elif "label" in names:
            detection_label = int(row.label)
        else:
            detection_label = 1 if family else 0
        rows.append(
            {
                "protein_id": protein_id,
                "sequence": sequence,
                "length": length,
                "family": family,
                "cluster_id": str(cluster_value),
                "detection_label": detection_label,
                "split": split,
            }
        )
    return rows


def _boundary_map(path: Path, split: str, minimum_agreement: float) -> dict[str, tuple[int, int, float]]:
    table = pq.read_table(
        path, columns=["protein_id", "start", "end", "agreement", "split", "length"]
    )
    data = table.to_pydict()
    result = {}
    for protein_id, start, end, agreement, assigned_split, length in zip(
        data["protein_id"],
        data["start"],
        data["end"],
        data["agreement"],
        data["split"],
        data["length"],
    ):
        if assigned_split != split or float(agreement) < minimum_agreement:
            continue
        if not (1 <= int(start) <= int(end) <= int(length)):
            continue
        if str(protein_id) in result:
            raise ValueError(f"Duplicate boundary label for {protein_id}")
        result[str(protein_id)] = (int(start), int(end), float(agreement))
    return result


def _boundary_position_weight(start: int, end: int, length: int, cfg: DataConfig) -> float:
    margin = cfg.boundary_internal_margin_aa
    internal_sides = int(start > margin) + int(end < length - margin)
    if internal_sides == 2:
        return cfg.boundary_internal_weight
    if internal_sides == 1:
        return cfg.boundary_single_internal_weight
    return cfg.boundary_terminal_weight


def _build_split_rows(cfg: DataConfig, root: Path, split: str) -> list[dict[str, Any]]:
    slice_split = "cal" if split == "calibration" else split
    if cfg.family_cohort:
        family_path = _safe_existing(root / cfg.family_dir / f"{split}.parquet")
        rows = _read_family(family_path, split)
    else:
        pos_path = _safe_existing(
            root / cfg.research_slice_dir / f"{slice_split}_pos.parquet"
        )
        rows = _read_slice(pos_path, split, 1)
    if split == "train":
        seen_positive = {row["protein_id"] for row in rows}
    else:
        seen_positive = {row["protein_id"] for row in rows}
    for source in cfg.supplementary_sources:
        rows.extend(_read_supplementary(_safe_existing(root / source), split))
    negative_ids: set[str] = set()
    if cfg.include_negatives:
        neg_path = _safe_existing(
            root / cfg.research_slice_dir / f"{slice_split}_neg.parquet"
        )
        negatives = _read_slice(neg_path, split, 0)
        negative_ids = {row["protein_id"] for row in negatives}
        overlap = seen_positive & negative_ids
        if overlap:
            raise ValueError(f"Positive/negative ID overlap in {split}: {sorted(overlap)[:5]}")
        rows.extend(negatives)
    ids = [row["protein_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate protein IDs in {split}")
    boundaries = _boundary_map(
        _safe_existing(root / cfg.boundary_labels), split, cfg.min_boundary_agreement
    )
    teacher_boundaries = {}
    if cfg.structure_teacher_labels:
        teacher_boundaries = _boundary_map(
            _safe_existing(root / cfg.structure_teacher_labels),
            split,
            cfg.min_teacher_agreement,
        )
    for row in rows:
        boundary = boundaries.get(row["protein_id"])
        if boundary:
            start, end, agreement = boundary
            row["boundary_start"] = start
            row["boundary_end"] = end
            row["boundary_weight"] = agreement * _boundary_position_weight(
                start, end, row["length"], cfg
            )
        teacher = teacher_boundaries.get(row["protein_id"])
        if teacher:
            start, end, agreement = teacher
            row["teacher_start"] = start
            row["teacher_end"] = end
            row["teacher_weight"] = agreement * _boundary_position_weight(
                start, end, row["length"], cfg
            )
    return rows


def build_cohort(cfg: DataConfig) -> tuple[dict[str, PreparedSplit], list[str]]:
    root = Path(cfg.project_root).expanduser().resolve()
    esm = TeacherStore(
        1152,
        "ESM-C",
        expected_window=cfg.embedding_window,
        allow_legacy_window=cfg.allow_legacy_embedding_window,
    )
    protrek = TeacherStore(
        1280,
        "ProTrek",
        expected_window=cfg.embedding_window,
        allow_legacy_window=cfg.allow_legacy_embedding_window,
    )
    esm.read(cfg.esm_sources, root)
    protrek.read(cfg.protrek_sources, root)

    rows_by_split = {
        split: _build_split_rows(cfg, root, split)
        for split in SPLITS
    }
    split_ids = {split: {row["protein_id"] for row in rows} for split, rows in rows_by_split.items()}
    for left, right in (("train", "dev"), ("train", "calibration"), ("dev", "calibration")):
        overlap = split_ids[left] & split_ids[right]
        if overlap:
            raise ValueError(f"Protein IDs cross {left}/{right}: {sorted(overlap)[:5]}")

    if cfg.family_cohort:
        family_names = sorted(
            {
                row["family"]
                for row in rows_by_split["train"]
                if row["family"] is not None
            }
        )
    else:
        family_names = sorted(
            {
                str(value)
                for value in pq.read_table(
                    _safe_existing(root / cfg.family_dir / "train.parquet"),
                    columns=["family"],
                )["family"].to_pylist()
                if value
            }
        )
    if len(family_names) < 2:
        raise ValueError("Training requires at least two families")
    for split in SPLITS[1:]:
        unknown = {
            row["family"]
            for row in rows_by_split[split]
            if row["family"] is not None and row["family"] not in family_names
        }
        if unknown:
            raise ValueError(f"{split} has families absent from train: {sorted(unknown)}")

    family_teacher = FamilyTeacherStore(len(family_names), "family_teacher")
    family_teacher.read(cfg.family_teacher_sources, root)
    detection_teacher = ScalarTeacherStore("detection_teacher")
    detection_teacher.read(cfg.detection_teacher_sources, root)

    prepared: dict[str, PreparedSplit] = {}
    for split in SPLITS:
        rows = rows_by_split[split]
        if cfg.require_all_modalities:
            missing = [
                row["protein_id"]
                for row in rows
                if row["protein_id"] not in esm.features
                or row["protein_id"] not in protrek.features
            ]
            if missing and not cfg.drop_missing_modalities:
                raise ValueError(
                    f"{split} is missing one or both requested modalities for "
                    f"{len(missing)} rows; first IDs: {missing[:5]}"
                )
            rows = [
                row
                for row in rows
                if row["protein_id"] in esm.features and row["protein_id"] in protrek.features
            ]
        if not rows:
            raise ValueError(f"No rows remain for {split}")
        records = [
            Record(
                protein_id=row["protein_id"],
                sequence=row["sequence"],
                length=row["length"],
                detection_label=row["detection_label"],
                family=row["family"],
                cluster_id=row["cluster_id"],
                split=split,
                boundary_start=row.get("boundary_start"),
                boundary_end=row.get("boundary_end"),
                boundary_weight=float(row.get("boundary_weight", 0.0)),
                teacher_start=row.get("teacher_start"),
                teacher_end=row.get("teacher_end"),
                teacher_weight=float(row.get("teacher_weight", 0.0)),
            )
            for row in rows
        ]
        prepared[split] = PreparedSplit(
            records,
            TokenBank(records),
            family_names,
            esm,
            protrek,
            family_teacher,
            detection_teacher,
        )
    return prepared, family_names


def plan_batches(
    split: PreparedSplit,
    indices: Iterable[int],
    batch_size: int,
    max_tokens: int,
    max_length: int,
    training: bool,
    rng: np.random.Generator | None = None,
) -> list[list[int]]:
    indices = list(indices)
    if training:
        if rng is None:
            raise ValueError("Training batches require an RNG")
        noise = rng.integers(0, 128, size=len(indices))
        indices = [
            index
            for _, index in sorted(zip(split.lengths[indices] + noise, indices))
        ]
    else:
        indices = sorted(indices, key=lambda index: int(split.lengths[index]))
    batches: list[list[int]] = []
    current: list[int] = []
    tokens = 0
    for index in indices:
        length = min(int(split.lengths[index]), max_length)
        if current and (len(current) >= batch_size or tokens + length > max_tokens):
            batches.append(current)
            current, tokens = [], 0
        current.append(index)
        tokens += length
    if current:
        batches.append(current)
    if training:
        assert rng is not None
        rng.shuffle(batches)
    return batches


def iter_batches(
    split: PreparedSplit,
    indices: Iterable[int],
    batch_size: int,
    max_tokens: int,
    max_length: int,
    training: bool,
    rng: np.random.Generator | None = None,
) -> Iterator[list[dict[str, Any]]]:
    for batch_indices in plan_batches(
        split, indices, batch_size, max_tokens, max_length, training, rng
    ):
        yield [split.sample(index, max_length, training, rng) for index in batch_indices]


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    max_length = max(len(sample["input_ids"]) for sample in samples)
    input_ids = torch.full((len(samples), max_length), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(samples), max_length), dtype=torch.bool)
    for row, sample in enumerate(samples):
        length = len(sample["input_ids"])
        input_ids[row, :length] = sample["input_ids"]
        attention_mask[row, :length] = True
    return {
        "protein_id": [sample["protein_id"] for sample in samples],
        "cluster_id": [sample["cluster_id"] for sample in samples],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "detection_labels": torch.tensor(
            [sample["detection_label"] for sample in samples], dtype=torch.float32
        ),
        "family_labels": torch.tensor(
            [sample["family_label"] for sample in samples], dtype=torch.long
        ),
        "start_labels": torch.tensor(
            [sample["start_label"] for sample in samples], dtype=torch.long
        ),
        "end_labels": torch.tensor(
            [sample["end_label"] for sample in samples], dtype=torch.long
        ),
        "status_labels": torch.tensor(
            [sample["status_label"] for sample in samples], dtype=torch.long
        ),
        "teacher_start_labels": torch.tensor(
            [sample["teacher_start_label"] for sample in samples], dtype=torch.long
        ),
        "teacher_end_labels": torch.tensor(
            [sample["teacher_end_label"] for sample in samples], dtype=torch.long
        ),
        "teacher_weights": torch.tensor(
            [sample["teacher_weight"] for sample in samples], dtype=torch.float32
        ),
        "boundary_weights": torch.tensor(
            [sample["boundary_weight"] for sample in samples], dtype=torch.float32
        ),
        "teacher_logits": torch.tensor(
            [sample["teacher_logit"] for sample in samples], dtype=torch.float32
        ),
        "esm": torch.stack([sample["esm"] for sample in samples]).float(),
        "esm_mask": torch.stack([sample["esm_mask"] for sample in samples]),
        "protrek": torch.stack([sample["protrek"] for sample in samples]).float(),
        "protrek_mask": torch.stack([sample["protrek_mask"] for sample in samples]),
        "teacher_family_logits": torch.stack(
            [sample["teacher_family_logits"] for sample in samples]
        ).float(),
        "teacher_family_mask": torch.stack(
            [sample["teacher_family_mask"] for sample in samples]
        ),
    }


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
