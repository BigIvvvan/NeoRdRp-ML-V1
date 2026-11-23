"""Strict configuration objects used by training and inference."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _check_keys(data: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {section}: {unknown}")


@dataclass(frozen=True)
class DataConfig:
    project_root: str = "."
    family_dir: str = "data/family_v1"
    research_slice_dir: str = "data/research_slice"
    boundary_labels: str = "data/boundary_weak_labels.parquet"
    structure_teacher_labels: str | None = None
    supplementary_sources: tuple[str, ...] = ()
    esm_sources: tuple[str, ...] = ()
    protrek_sources: tuple[str, ...] = ()
    detection_teacher_sources: tuple[str, ...] = ()
    family_teacher_sources: tuple[str, ...] = ()
    require_all_modalities: bool = False
    drop_missing_modalities: bool = False
    include_negatives: bool = True
    family_cohort: bool = True
    max_length: int = 2048
    eval_max_length: int = 4096
    embedding_window: int = 2048
    allow_legacy_embedding_window: bool = False
    boundary_sigma_aa: float = 5.0
    min_boundary_agreement: float = 0.50
    min_teacher_agreement: float = 0.0
    boundary_internal_margin_aa: int = 10
    boundary_internal_weight: float = 1.0
    boundary_single_internal_weight: float = 1.0
    boundary_terminal_weight: float = 1.0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DataConfig:
        _check_keys(raw, set(cls.__dataclass_fields__), "data")
        values = dict(raw)
        for key in (
            "supplementary_sources",
            "esm_sources",
            "protrek_sources",
            "detection_teacher_sources",
            "family_teacher_sources",
        ):
            values[key] = tuple(str(item) for item in values.get(key, ()))
        cfg = cls(**values)
        if cfg.embedding_window != 2048:
            raise ValueError("ESM-C and ProTrek feature windows must be 2048 residues")
        if min(cfg.max_length, cfg.eval_max_length) < 1:
            raise ValueError("Sequence limits must be positive")
        if cfg.boundary_sigma_aa <= 0:
            raise ValueError("Boundary sigma must be positive")
        if not 0 <= cfg.min_boundary_agreement <= 1:
            raise ValueError("Boundary agreement must be within [0, 1]")
        return cfg


@dataclass(frozen=True)
class ModelConfig:
    encoder_type: str = "mamba"
    d_model: int = 384
    n_layers: int = 6
    d_state: int = 128
    d_head: int = 64
    chunk_size: int = 64
    dropout: float = 0.1
    share_direction_weights: bool = False
    gradient_checkpointing: bool = True
    dt_limit: float = 0.1
    esm_dim: int = 1152
    protrek_dim: int = 1280
    fusion_hidden: int = 512
    feature_usage: str = "input"
    boundary_head_hidden: int = 384

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ModelConfig:
        _check_keys(raw, set(cls.__dataclass_fields__), "model")
        cfg = cls(**raw)
        if cfg.encoder_type not in {"mamba", "transformer"}:
            raise ValueError("model.encoder_type must be mamba or transformer")
        if cfg.feature_usage not in {"input", "kd_only", "both", "none"}:
            raise ValueError("model.feature_usage is invalid")
        if cfg.d_model % cfg.d_head or cfg.d_model % 2:
            raise ValueError("Model dimensions are incompatible with the encoder")
        if min(cfg.d_model, cfg.n_layers, cfg.d_state, cfg.d_head, cfg.chunk_size) < 1:
            raise ValueError("Model dimensions must be positive")
        if not 0 <= cfg.dropout < 1 or cfg.dt_limit <= 0:
            raise ValueError("Dropout or dt_limit is invalid")
        return cfg


@dataclass(frozen=True)
class LossConfig:
    detection_weight: float = 1.0
    family_weight: float = 1.0
    boundary_weight: float = 1.0
    start_weight: float = 1.0
    end_weight: float = 1.0
    boundary_expected_l1_weight: float = 0.0
    detection_pos_weight: float = 1.0
    family_label_smoothing: float = 0.0
    kd_detection_weight: float = 0.0
    kd_feature_weight: float = 0.0
    kd_contrastive_weight: float = 0.0
    kd_family_weight: float = 0.0
    kd_temperature: float = 2.0
    kd_contrastive_temperature: float = 0.07
    detection_focal_gamma: float = 0.0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LossConfig:
        _check_keys(raw, set(cls.__dataclass_fields__), "loss")
        cfg = cls(**raw)
        numeric = [getattr(cfg, name) for name in cls.__dataclass_fields__]
        if any(float(value) < 0 for value in numeric):
            raise ValueError("Loss weights must be non-negative")
        if cfg.detection_pos_weight <= 0:
            raise ValueError("loss.detection_pos_weight must be positive")
        if cfg.kd_temperature <= 0 or cfg.kd_contrastive_temperature <= 0:
            raise ValueError("Knowledge-distillation temperatures must be positive")
        return cfg


@dataclass(frozen=True)
class OptimConfig:
    lr: float = 5e-4
    min_lr: float = 5e-5
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_fraction: float = 0.05
    grad_clip: float = 1.0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> OptimConfig:
        _check_keys(raw, set(cls.__dataclass_fields__), "optim")
        values = dict(raw)
        if "betas" in values:
            values["betas"] = tuple(float(value) for value in values["betas"])
        cfg = cls(**values)
        if cfg.lr <= 0 or not 0 <= cfg.min_lr <= cfg.lr:
            raise ValueError("Learning rates are invalid")
        if not 0 <= cfg.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be within [0, 1)")
        if len(cfg.betas) != 2 or any(not 0 <= value < 1 for value in cfg.betas):
            raise ValueError("optim.betas must contain two values in [0, 1)")
        return cfg


@dataclass(frozen=True)
class TrainLoopConfig:
    epochs: int = 3
    batch_size: int = 64
    grad_accum: int = 2
    max_tokens: int = 131072
    eval_batch_size: int = 8
    eval_max_tokens: int = 8192
    amp: str = "bf16"
    log_every: int = 20
    save_every: int = 250
    patience: int = 0
    min_delta: float = 0.0002
    num_threads: int = 8
    device: str = "cuda"
    output_dir: str = "runs/neordrp_ml_v1"
    init_from: str | None = None
    init_allow_missing_prefixes: tuple[str, ...] = ()
    selection: Mapping[str, float] = field(
        default_factory=lambda: {
            "detection_auprc_p10": 1.0,
            "detection_recall_fpr_1e_3": 1.0,
            "family_macro_f1": 1.0,
            "boundary_both_e10": 0.5,
        }
    )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TrainLoopConfig:
        _check_keys(raw, set(cls.__dataclass_fields__), "train")
        values = dict(raw)
        if "init_allow_missing_prefixes" in values:
            values["init_allow_missing_prefixes"] = tuple(
                str(item) for item in values["init_allow_missing_prefixes"]
            )
        selection = values.get("selection", {})
        if not isinstance(selection, Mapping):
            raise TypeError("train.selection must be a mapping")
        values["selection"] = {str(key): float(value) for key, value in selection.items()}
        cfg = cls(**values)
        if cfg.device != "cuda":
            raise ValueError("NeoRdRp-ML training is GPU-only")
        if cfg.amp not in {"bf16", "fp16", "none"}:
            raise ValueError("train.amp must be bf16, fp16, or none")
        positive = (
            cfg.epochs,
            cfg.batch_size,
            cfg.grad_accum,
            cfg.max_tokens,
            cfg.eval_batch_size,
            cfg.eval_max_tokens,
            cfg.log_every,
            cfg.save_every,
            cfg.num_threads,
        )
        if min(positive) < 1:
            raise ValueError("Training schedule values must be positive")
        return cfg


@dataclass(frozen=True)
class TrainConfig:
    name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    optim: OptimConfig
    train: TrainLoopConfig

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TrainConfig:
        _check_keys(raw, {"name", "seed", "data", "model", "loss", "optim", "train"}, "root")
        return cls(
            name=str(raw["name"]),
            seed=int(raw["seed"]),
            data=DataConfig.from_dict(raw["data"]),
            model=ModelConfig.from_dict(raw["model"]),
            loss=LossConfig.from_dict(raw["loss"]),
            optim=OptimConfig.from_dict(raw["optim"]),
            train=TrainLoopConfig.from_dict(raw["train"]),
        )

    def canonical_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resolve_root(self) -> Path:
        return Path(self.data.project_root).expanduser().resolve()


def load_config(path: str | Path) -> TrainConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError("Configuration root must be a mapping")
    return TrainConfig.from_dict(raw)
