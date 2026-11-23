"""Dual-representation, bidirectional Mamba student with three task outputs."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
EXTENDED_AA = "BZXUOJ"
SPECIALS = ("<pad>", "<cls>", "<eos>", "<mask>", "<unk>")
VOCAB = SPECIALS + tuple(STANDARD_AA) + tuple(EXTENDED_AA) + ("<res0>", "<res1>")
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}
PAD_ID = TOKEN_TO_ID["<pad>"]
UNK_ID = TOKEN_TO_ID["<unk>"]


def tokenize(sequence: str) -> list[int]:
    return [TOKEN_TO_ID.get(residue.upper(), UNK_ID) for residue in sequence]


def sinusoidal_position_encoding(
    length: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32)
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = positions[:, None] * frequencies[None, :]
    encoding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    return encoding.unsqueeze(0).to(dtype=dtype)


class RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        if gate is not None:
            x = x * F.silu(gate.float())
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class Mamba2Block(nn.Module):
    """Pure PyTorch chunked SSD block.

    The recurrence is evaluated chunk by chunk so memory does not scale with a
    dense L x L matrix. Calculations of the state recurrence use float32.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_head: int = 64,
        chunk_size: int = 64,
        dt_limit: float = 0.1,
        expand: int = 2,
        n_groups: int = 1,
        conv_kernel: int = 4,
        eps: float = 1e-5,
    ):
        super().__init__()
        if d_model % d_head:
            raise ValueError("d_model must be divisible by d_head")
        self.d_model = d_model
        self.d_state = d_state
        self.d_head = d_head
        self.d_inner = expand * d_model
        if self.d_inner % d_head:
            raise ValueError("expand*d_model must be divisible by d_head")
        self.n_heads = self.d_inner // d_head
        if self.n_heads % n_groups:
            raise ValueError("n_heads must be divisible by n_groups")
        self.n_groups = n_groups
        self.chunk_size = chunk_size
        self.dt_limit = dt_limit
        self.eps = eps
        self.conv_dim = self.d_inner + 2 * self.n_groups * d_state
        self.in_proj = nn.Linear(
            d_model, 2 * self.d_inner + 2 * self.n_groups * d_state + self.n_heads, bias=False
        )
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=self.conv_dim,
            bias=True,
        )
        self.dt_bias = nn.Parameter(torch.empty(self.n_heads))
        self.A_log = nn.Parameter(torch.empty(self.n_heads))
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.norm = RMSNormGated(self.d_inner, eps=eps)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dt_min, dt_max = 1e-3, 0.1
        dt = torch.exp(
            torch.rand(self.n_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            self.A_log.copy_(torch.log(torch.empty(self.n_heads).uniform_(1.0, 16.0)))

    def _ssd(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        Bm: torch.Tensor,
        Cm: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, heads, head_dim = x.shape
        n_states = Bm.shape[-1]
        chunk_size = self.chunk_size
        n_chunks = math.ceil(length / chunk_size)
        pad = n_chunks * chunk_size - length
        if pad:
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
            dt = F.pad(dt, (0, 0, 0, pad))
            Bm = F.pad(Bm, (0, 0, 0, 0, 0, pad))
            Cm = F.pad(Cm, (0, 0, 0, 0, 0, pad))
            mask = F.pad(mask, (0, pad))

        xc = x.reshape(batch, n_chunks, chunk_size, heads, head_dim).float()
        dtc = dt.reshape(batch, n_chunks, chunk_size, heads).float()
        Bc = Bm.reshape(batch, n_chunks, chunk_size, heads, n_states).float()
        Cc = Cm.reshape(batch, n_chunks, chunk_size, heads, n_states).float()
        mc = mask.reshape(batch, n_chunks, chunk_size).bool()
        dtc = dtc * mc[..., None].float()
        Bc = Bc * mc[..., None, None].float()
        xc = xc * mc[..., None, None].float()

        A = -torch.exp(self.A_log.float())
        dA = (dtc * A).permute(0, 1, 3, 2)
        cumulative = torch.cumsum(dA, dim=-1)
        within = torch.arange(chunk_size, device=x.device)
        causal = within[:, None] >= within[None, :]
        segment = cumulative[..., :, None] - cumulative[..., None, :]
        decay_within = torch.exp(
            segment.masked_fill(~causal, float("-inf"))
        )
        cb = torch.einsum("bclhn,bcshn->bchls", Cc, Bc)
        dt_x = xc * dtc.unsqueeze(-1)
        diagonal_output = torch.einsum(
            "bchls,bcshp->bclhp", cb * decay_within, dt_x
        )

        end_decay = torch.exp(cumulative[..., -1:] - cumulative)
        chunk_states = torch.einsum(
            "bchl,bclhn,bclhp->bchpn", end_decay, Bc, dt_x
        )
        chunk_log = cumulative[..., -1]
        inclusive = torch.cumsum(chunk_log, dim=1)
        exclusive = inclusive - chunk_log
        difference = exclusive.permute(0, 2, 1)[..., :, None] - inclusive.permute(
            0, 2, 1
        )[..., None, :]
        strict_lower = torch.tril(
            torch.ones(n_chunks, n_chunks, dtype=torch.bool, device=x.device),
            diagonal=-1,
        )
        previous_decay = torch.exp(
            difference.masked_fill(~strict_lower, float("-inf"))
        )
        previous_state = torch.einsum(
            "bhcd,bdhpn->bchpn", previous_decay, chunk_states
        )
        cross_chunk = torch.einsum(
            "bclhn,bchpn,bchl->bclhp",
            Cc,
            previous_state,
            torch.exp(cumulative),
        )
        y = (diagonal_output + cross_chunk).reshape(
            batch, length + pad, heads, head_dim
        )
        y = y[:, :length] * mask[:, :length, None, None].float()
        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            mask = mask.bool()
        projected = self.in_proj(x)
        z, xbc, dt_raw = torch.split(
            projected, [self.d_inner, self.conv_dim, self.n_heads], dim=-1
        )
        xbc = xbc * mask.unsqueeze(-1).to(xbc.dtype)
        xbc = self.conv1d(xbc.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        xbc = F.silu(xbc) * mask.unsqueeze(-1).to(xbc.dtype)
        with torch.autocast(device_type=x.device.type, enabled=False):
            xbc = xbc.float()
            x_ssm, bm, cm = torch.split(
                xbc,
                [
                    self.d_inner,
                    self.n_groups * self.d_state,
                    self.n_groups * self.d_state,
                ],
                dim=-1,
            )
            x_heads = x_ssm.reshape(
                x.shape[0], x.shape[1], self.n_heads, self.d_head
            )
            repeat = self.n_heads // self.n_groups
            bm = bm.reshape(x.shape[0], x.shape[1], self.n_groups, self.d_state)
            cm = cm.reshape(x.shape[0], x.shape[1], self.n_groups, self.d_state)
            bm = bm.repeat_interleave(repeat, dim=2)
            cm = cm.repeat_interleave(repeat, dim=2)
            dt = F.softplus(dt_raw.float() + self.dt_bias.float()).clamp(
                max=self.dt_limit
            )
            y = self._ssd(x_heads, dt, bm, cm, mask)
            y = y + self.D.float()[None, None, :, None] * x_heads
            y = y.reshape(x.shape[0], x.shape[1], self.d_inner)
        y = self.norm(y.to(x.dtype), z)
        y = self.out_proj(y)
        return y * mask.unsqueeze(-1).to(y.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderStream(nn.Module):
    def __init__(self, kind: str, d_model: int, n_layers: int, dropout: float, **kwargs: Any):
        super().__init__()
        self.gradient_checkpointing = bool(kwargs.get("gradient_checkpointing", False))
        if kind == "mamba":
            self.blocks = nn.ModuleList(
                Mamba2Block(
                    d_model,
                    d_state=kwargs["d_state"],
                    d_head=kwargs["d_head"],
                    chunk_size=kwargs["chunk_size"],
                    dt_limit=kwargs["dt_limit"],
                )
                for _ in range(n_layers)
            )
        else:
            if kwargs["d_head"] < 8 or d_model % (kwargs["d_head"] // 8):
                raise ValueError("transformer encoder requires a valid attention head width")
            self.blocks = nn.ModuleList(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=kwargs["d_head"] // 8,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            )
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.drop = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.kind = kind

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for norm, block in zip(self.norms, self.blocks):
            if self.kind == "mamba":
                if self.gradient_checkpointing and self.training:
                    update = torch.utils.checkpoint.checkpoint(
                        block, norm(x), mask, use_reentrant=False
                    )
                else:
                    update = block(norm(x), mask)
            else:
                if self.gradient_checkpointing and self.training:
                    update = torch.utils.checkpoint.checkpoint(
                        lambda values, valid, block=block: block(
                            values, src_key_padding_mask=~valid
                        ),
                        norm(x),
                        mask,
                        use_reentrant=False,
                    )
                else:
                    update = block(norm(x), src_key_padding_mask=~mask)
            x = x + self.drop(update)
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return self.final_norm(x)


class BiEncoder(nn.Module):
    def __init__(
        self,
        kind: str,
        d_model: int,
        n_layers: int,
        dropout: float,
        share_direction_weights: bool,
        position_embedding: str = "none",
        **kwargs: Any,
    ):
        super().__init__()
        self.embed = nn.Embedding(len(VOCAB), d_model, padding_idx=PAD_ID)
        self.position_embedding = position_embedding
        self.d_model = d_model
        checkpointing = bool(kwargs.get("gradient_checkpointing", False))
        stream_kwargs = {
            key: value for key, value in kwargs.items() if key != "gradient_checkpointing"
        }
        self.fwd = EncoderStream(
            kind,
            d_model,
            n_layers,
            dropout,
            gradient_checkpointing=checkpointing,
            **stream_kwargs,
        )
        self.rev = self.fwd if share_direction_weights else EncoderStream(
            kind,
            d_model,
            n_layers,
            dropout,
            gradient_checkpointing=checkpointing,
            **stream_kwargs,
        )
        self.merge = nn.Linear(2 * d_model, d_model)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.bool()
        x = self.embed(input_ids)
        if self.position_embedding == "sinusoidal":
            x = x + sinusoidal_position_encoding(
                input_ids.shape[1], self.d_model, input_ids.device, x.dtype
            )
        x = x + context.unsqueeze(1)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        forward = self.fwd(x, mask)
        reverse = torch.flip(
            self.rev(torch.flip(x, dims=[1]), torch.flip(mask, dims=[1])), dims=[1]
        )
        hidden = self.merge(torch.cat([forward, reverse], dim=-1))
        return hidden * mask.unsqueeze(-1).to(hidden.dtype)


class PooledFeatureFusion(nn.Module):
    def __init__(self, esm_dim: int, protrek_dim: int, hidden: int, d_model: int):
        super().__init__()
        self.esm = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, d_model),
        )
        self.protrek = nn.Sequential(
            nn.LayerNorm(protrek_dim),
            nn.Linear(protrek_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        esm: torch.Tensor,
        esm_mask: torch.Tensor,
        protrek: torch.Tensor,
        protrek_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        esm_ok = esm_mask.bool()
        pro_ok = protrek_mask.bool()
        esm_hidden = self.esm(esm.float())
        pro_hidden = self.protrek(protrek.float())
        presence = torch.stack([esm_ok, pro_ok], dim=-1).to(esm_hidden.dtype)
        gate_logits = self.gate(torch.cat([esm_hidden, pro_hidden, presence], dim=-1))
        gate_logits = gate_logits.masked_fill(~torch.stack([esm_ok, pro_ok], dim=-1), -1e4)
        weights = torch.softmax(gate_logits, dim=-1)
        any_feature = esm_ok | pro_ok
        fused = weights[:, 0:1] * esm_hidden + weights[:, 1:2] * pro_hidden
        fused = fused * any_feature.unsqueeze(-1).to(fused.dtype)
        weights = weights * any_feature.unsqueeze(-1).to(weights.dtype)
        return fused, weights, esm_hidden, pro_hidden


class MaskedMean(nn.Module):
    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class NeoRdRpV1(nn.Module):
    """Dual-representation bidirectional Mamba model used for the V1 release."""

    def __init__(self, cfg: Any, n_families: int):
        super().__init__()
        self.cfg = cfg
        self.n_families = int(n_families)
        self.encoder = BiEncoder(
            cfg.encoder_type,
            cfg.d_model,
            cfg.n_layers,
            cfg.dropout,
            cfg.share_direction_weights,
            position_embedding="none",
            gradient_checkpointing=cfg.gradient_checkpointing,
            d_state=cfg.d_state,
            d_head=cfg.d_head,
            chunk_size=cfg.chunk_size,
            dt_limit=cfg.dt_limit,
        )
        self.fusion = PooledFeatureFusion(
            cfg.esm_dim, cfg.protrek_dim, cfg.fusion_hidden, cfg.d_model
        )
        self.pool = MaskedMean()
        joint_dim = 2 * cfg.d_model
        self.joint_norm = nn.LayerNorm(joint_dim)
        self.detection = nn.Sequential(
            nn.Linear(joint_dim, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )
        self.family = nn.Sequential(
            nn.Linear(joint_dim, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, self.n_families),
        )
        self.boundary_trunk = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.boundary_head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.start = nn.Linear(cfg.boundary_head_hidden, 1)
        self.end = nn.Linear(cfg.boundary_head_hidden, 1)
        self.status = nn.Linear(cfg.boundary_head_hidden, 8)
        self.kd_esm = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, cfg.fusion_hidden),
            nn.GELU(),
            nn.Linear(cfg.fusion_hidden, cfg.esm_dim),
        )
        self.kd_protrek = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, cfg.fusion_hidden),
            nn.GELU(),
            nn.Linear(cfg.fusion_hidden, cfg.protrek_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].bool()
        fused, modality_weights, esm_hidden, protrek_hidden = self.fusion(
            batch["esm"],
            batch["esm_mask"],
            batch["protrek"],
            batch["protrek_mask"],
        )
        use_context = self.cfg.feature_usage in {"input", "both"}
        context = fused if use_context else torch.zeros_like(fused)
        hidden = self.encoder(input_ids, attention_mask, context)
        pooled = self.pool(hidden, attention_mask)
        joint = self.joint_norm(torch.cat([pooled, fused], dim=-1))
        detection_logits = self.detection(joint).squeeze(-1)
        family_logits = self.family(joint)
        boundary = self.boundary_trunk(hidden)
        start_logits = self.start(boundary).squeeze(-1)
        end_logits = self.end(boundary).squeeze(-1)
        status_logits = self.status(self.pool(boundary, attention_mask)).reshape(
            input_ids.shape[0], 2, 4
        )
        return {
            "detection_logits": detection_logits,
            "family_logits": family_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "status_logits": status_logits,
            "hidden": hidden,
            "pooled": pooled,
            "fused_features": fused,
            "modality_weights": modality_weights,
            "esm_hidden": esm_hidden,
            "protrek_hidden": protrek_hidden,
            "kd_esm": self.kd_esm(joint),
            "kd_protrek": self.kd_protrek(joint),
        }
