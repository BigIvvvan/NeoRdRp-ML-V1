"""Protein-language-model feature extraction for ESM-C and ProTrek."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")
PROTREK_PREFIX = "1.model.esm."


def clean_sequence(sequence: str) -> str:
    """Normalize whitespace and map unsupported residues to unknown."""
    compact = "".join(str(sequence).split()).upper()
    compact = compact.replace("*", "").replace("-", "").replace(".", "")
    return "".join(residue if residue in AA20 else "X" for residue in compact)


def plan_windows(
    length: int,
    strategy: str,
    maximum: int = 2048,
    stride: int = 512,
) -> list[tuple[int, int]]:
    """Return zero-based half-open windows for one sequence."""
    if length <= 0:
        return []
    if length <= maximum:
        return [(0, length)]
    if strategy == "head":
        return [(0, maximum)]
    if strategy == "center":
        start = (length - maximum) // 2
        return [(start, start + maximum)]
    if strategy != "window":
        raise ValueError(f"Unknown window strategy: {strategy}")
    starts = list(range(0, length - maximum + 1, stride))
    if starts[-1] + maximum < length:
        starts.append(length - maximum)
    return [(start, start + maximum) for start in starts]


def _protrek_directory() -> Path:
    configured = os.environ.get("NEORDRP_PROTREK_DIR")
    if configured:
        return Path(configured).expanduser()
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    root = cache / "hub/models--westlake-repl--ProTrek_650M/snapshots"
    snapshots = sorted(
        path for path in root.glob("*") if (path / "ProTrek_650M.pt").is_file()
    )
    if not snapshots:
        raise FileNotFoundError(
            "ProTrek weights are absent. Set NEORDRP_PROTREK_DIR or download "
            "westlake-repl/ProTrek_650M from Hugging Face."
        )
    return snapshots[-1]


def _load_protrek_protein_branch():
    from transformers import EsmConfig, EsmModel, EsmTokenizer

    directory = _protrek_directory()
    esm_directory = directory / "esm2_t33_650M_UR50D"
    config = EsmConfig.from_pretrained(esm_directory)
    tokenizer = EsmTokenizer.from_pretrained(esm_directory)
    checkpoint = torch.load(
        directory / "ProTrek_650M.pt", map_location="cpu", weights_only=False
    )["model"]
    state = {
        key[len(PROTREK_PREFIX) :]: value
        for key, value in checkpoint.items()
        if key.startswith(PROTREK_PREFIX)
    }
    model = EsmModel(config, add_pooling_layer=False)
    loaded = model.load_state_dict(state, strict=False)
    missing = [
        key
        for key in loaded.missing_keys
        if "position_embeddings" not in key and "contact_head" not in key
    ]
    unexpected = [key for key in loaded.unexpected_keys if key != "embeddings.position_ids"]
    if missing or unexpected:
        raise ValueError(
            f"ProTrek protein branch is incompatible: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    return model, tokenizer


def _patch_esm_attention() -> None:
    import transformers.models.esm.modeling_esm as module

    if getattr(module.EsmSelfAttention, "_neordrp_sdpa", False):
        return

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        query = self.transpose_for_scores(self.query(hidden_states))
        query = query * self.attention_head_size**-0.5
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        if self.position_embedding_type == "rotary":
            query, key = self.rotary_embeddings(query, key)
        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask.to(query.dtype) if attention_mask is not None else None,
            scale=1.0,
        )
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))
        return (context,)

    module.EsmSelfAttention.forward = forward
    module.EsmSelfAttention._neordrp_sdpa = True


class ProteinBackbone:
    """Frozen pooled-residue extractor for one supported protein model."""

    def __init__(self, kind: str, device: str = "auto", dtype: torch.dtype = torch.bfloat16):
        if kind not in {"esmc_600m", "protrek_650m"}:
            raise ValueError("kind must be esmc_600m or protrek_650m")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kind = kind
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        if kind == "esmc_600m":
            from esm.models.esmc import ESMC

            model = ESMC.from_pretrained("esmc_600m", device=device)
            self.tokenizer = model.tokenizer
            self.model = model.to(dtype).eval()
            self.dimension = 1152
            self.layer_count = 36
            self.layers = self.model.transformer.blocks
            del self.model.sequence_head
        else:
            _patch_esm_attention()
            model, self.tokenizer = _load_protrek_protein_branch()
            self.model = model.to(self.device, dtype).eval()
            self.dimension = 1280
            self.layer_count = 33
            self.layers = self.model.encoder.layer
        if len(self.layers) != self.layer_count:
            raise ValueError("Unexpected protein-model layer count")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _tokenize(self, sequences: Sequence[str]):
        encoded = self.tokenizer(
            list(sequences), padding=True, return_tensors="pt", add_special_tokens=True
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        lengths = torch.tensor([len(sequence) for sequence in sequences])
        positions = torch.arange(input_ids.shape[1])[None, :]
        residue_mask = (positions >= 1) & (positions <= lengths[:, None])
        return input_ids, attention_mask, residue_mask.to(self.device)

    def _hidden(self, input_ids, attention_mask):
        if self.kind == "protrek_650m":
            hidden = self.model.embeddings(input_ids=input_ids, attention_mask=attention_mask)
            context = self.model.get_extended_attention_mask(
                attention_mask, input_ids.shape
            ).to(hidden.dtype)
            for layer in self.layers:
                hidden = layer(hidden, attention_mask=context)[0]
            final_norm = getattr(self.model.encoder, "emb_layer_norm_after", None)
            return final_norm(hidden) if final_norm is not None else hidden
        hidden = self.model.embed(input_ids)
        context = attention_mask.bool()
        chain = torch.ones(hidden.shape[:2], dtype=torch.int64, device=hidden.device)
        for layer in self.layers:
            hidden = layer(hidden, context, None, None, chain)
        return self.model.transformer.norm(hidden)

    @torch.no_grad()
    def pool(self, sequences: Sequence[str]) -> torch.Tensor:
        input_ids, attention_mask, residue_mask = self._tokenize(sequences)
        hidden = self._hidden(input_ids, attention_mask).float()
        weights = residue_mask.unsqueeze(-1).float()
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


@dataclass
class WindowJob:
    sequence_index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _pack_batches(
    jobs: list[WindowJob], max_tokens: int, max_sequences: int
) -> list[list[WindowJob]]:
    ordered = sorted(jobs, key=lambda job: job.length)
    batches: list[list[WindowJob]] = []
    current: list[WindowJob] = []
    current_max = 0
    for job in ordered:
        new_max = max(current_max, job.length + 2)
        if current and (
            new_max * (len(current) + 1) > max_tokens
            or len(current) >= max_sequences
        ):
            batches.append(current)
            current = []
            new_max = job.length + 2
        current.append(job)
        current_max = new_max
    if current:
        batches.append(current)
    return batches


def embed_sequences(
    backbone: ProteinBackbone,
    sequences: Sequence[str],
    max_tokens: int = 32768,
    max_windows_per_batch: int = 16,
) -> np.ndarray:
    """Pool every 2048-residue window and aggregate by sequence length."""
    cleaned = [clean_sequence(sequence) for sequence in sequences]
    jobs = [
        WindowJob(row, start, end)
        for row, sequence in enumerate(cleaned)
        for start, end in plan_windows(len(sequence), "window")
    ]
    accumulator = torch.zeros(len(sequences), backbone.dimension, dtype=torch.float64)
    weight_sum = torch.zeros(len(sequences), dtype=torch.float64)
    for batch in _pack_batches(jobs, max_tokens, max_windows_per_batch):
        windows = [cleaned[job.sequence_index][job.start : job.end] for job in batch]
        pooled = backbone.pool(windows).double().cpu()
        for row, job in enumerate(batch):
            accumulator[job.sequence_index] += pooled[row] * job.length
            weight_sum[job.sequence_index] += job.length
    pooled = accumulator / weight_sum.clamp_min(1).unsqueeze(-1)
    return pooled.float().numpy()
