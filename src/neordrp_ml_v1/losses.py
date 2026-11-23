"""Masked multi-task objective with feature and logit distillation."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def _zero_like(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.sum() * 0.0


def masked_focal_binary_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float = 1.0,
    gamma: float = 0.0,
) -> torch.Tensor:
    """Binary cross-entropy with optional focal reweighting and masking."""
    mask = mask.bool()
    if not mask.any():
        return _zero_like(logits)
    target = labels.float().masked_fill(~mask, 0.0)
    loss = F.binary_cross_entropy_with_logits(
        logits.float(),
        target,
        pos_weight=torch.tensor(pos_weight, device=logits.device),
        reduction="none",
    )
    if gamma > 0:
        probability = torch.sigmoid(logits.float())
        probability = probability * target + (1 - probability) * (1 - target)
        loss = loss * (1 - probability).clamp_min(0).pow(gamma)
    return loss[mask].mean()


def masked_soft_endpoint_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    sigma_aa: float,
    expected_l1_weight: float = 0.0,
) -> torch.Tensor:
    """Gaussian-smoothed endpoint likelihood for weak boundary labels."""
    valid = labels.ge(0) & labels.lt(logits.shape[1]) & attention_mask.bool().any(dim=1)
    if not valid.any():
        return _zero_like(logits)
    log_probability = F.log_softmax(
        logits.float().masked_fill(~attention_mask.bool(), -1e4), dim=-1
    )
    positions = torch.arange(logits.shape[1], device=logits.device).float()
    safe_labels = labels.clamp_min(0).float()
    target = torch.exp(
        -0.5 * ((positions[None, :] - safe_labels[:, None]) / sigma_aa).pow(2)
    )
    target = target * attention_mask.float()
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    negative_log_likelihood = -(target * log_probability).sum(dim=-1)
    expected = (log_probability.exp() * positions[None, :]).sum(dim=-1)
    expected_error = (expected - safe_labels).abs()
    weight = sample_weight.float().clamp_min(0)
    denominator = (valid.float() * weight).sum().clamp_min(1)
    return (
        (negative_log_likelihood * valid.float() * weight).sum()
        + expected_l1_weight * (expected_error * valid.float() * weight).sum()
    ) / denominator


def bernoulli_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Temperature-scaled binary knowledge distillation."""
    teacher_probability = torch.sigmoid(teacher_logits.float() / temperature)
    student_scaled = student_logits.float() / temperature
    cross_entropy = F.binary_cross_entropy_with_logits(
        student_scaled, teacher_probability, reduction="mean"
    )
    entropy = (
        teacher_probability * torch.log(teacher_probability.clamp_min(1e-8))
        + (1 - teacher_probability)
        * torch.log((1 - teacher_probability).clamp_min(1e-8))
    ).mean()
    return (cross_entropy - entropy) * temperature**2


def feature_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    """Variance-normalized squared error between projected and teacher features."""
    if not available.any():
        return _zero_like(student)
    difference = student[available].float() - teacher[available].float()
    scale = teacher[available].float().var().detach().clamp_min(1e-6)
    return difference.square().mean() / scale


def contrastive_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    available: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric in-batch contrastive alignment between student and teacher."""
    if int(available.sum()) < 2:
        return _zero_like(student)
    student = F.normalize(student[available].float(), dim=-1)
    teacher = F.normalize(teacher[available].float(), dim=-1)
    logits = student @ teacher.transpose(0, 1) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )


def multi_task_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | Any],
    cfg: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine detection, family, boundary, and modality distillation terms."""
    loss_cfg = cfg.loss
    detection_labels = batch["detection_labels"].float()
    detection = masked_focal_binary_cross_entropy(
        output["detection_logits"].float(),
        detection_labels,
        detection_labels.ge(0),
        pos_weight=loss_cfg.detection_pos_weight,
        gamma=loss_cfg.detection_focal_gamma,
    )

    family_labels = batch["family_labels"].long()
    family_valid = family_labels.ge(0)
    if family_valid.any():
        family = F.cross_entropy(
            output["family_logits"].float()[family_valid],
            family_labels[family_valid],
            label_smoothing=loss_cfg.family_label_smoothing,
        )
    else:
        family = _zero_like(output["family_logits"])

    weights = batch["boundary_weights"].float()
    start = masked_soft_endpoint_loss(
        output["start_logits"],
        batch["start_labels"].long(),
        batch["attention_mask"],
        weights,
        cfg.data.boundary_sigma_aa,
        loss_cfg.boundary_expected_l1_weight,
    )
    end = masked_soft_endpoint_loss(
        output["end_logits"],
        batch["end_labels"].long(),
        batch["attention_mask"],
        weights,
        cfg.data.boundary_sigma_aa,
        loss_cfg.boundary_expected_l1_weight,
    )
    boundary = loss_cfg.start_weight * start + loss_cfg.end_weight * end

    teacher_logits = batch["teacher_logits"].float()
    teacher_mask = torch.isfinite(teacher_logits)
    detection_kd = (
        bernoulli_kl(
            output["detection_logits"][teacher_mask],
            teacher_logits[teacher_mask],
            loss_cfg.kd_temperature,
        )
        if teacher_mask.any()
        else _zero_like(output["detection_logits"])
    )
    feature_kd = 0.5 * (
        feature_loss(output["kd_esm"], batch["esm"], batch["esm_mask"])
        + feature_loss(output["kd_protrek"], batch["protrek"], batch["protrek_mask"])
    )
    contrastive_kd = 0.5 * (
        contrastive_loss(
            output["kd_esm"],
            batch["esm"],
            batch["esm_mask"],
            loss_cfg.kd_contrastive_temperature,
        )
        + contrastive_loss(
            output["kd_protrek"],
            batch["protrek"],
            batch["protrek_mask"],
            loss_cfg.kd_contrastive_temperature,
        )
    )

    total = (
        loss_cfg.detection_weight * detection
        + loss_cfg.family_weight * family
        + loss_cfg.boundary_weight * boundary
        + loss_cfg.kd_detection_weight * detection_kd
        + loss_cfg.kd_feature_weight * feature_kd
        + loss_cfg.kd_contrastive_weight * contrastive_kd
    )
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite multi-task loss")
    parts = {
        "loss": float(total.detach()),
        "detect": float(detection.detach()),
        "family": float(family.detach()),
        "boundary": float(boundary.detach()),
        "start": float(start.detach()),
        "end": float(end.detach()),
        "kd_detect": float(detection_kd.detach()),
        "kd_feature": float(feature_kd.detach()),
        "kd_contrastive": float(contrastive_kd.detach()),
        "kd_detect_coverage": float(teacher_mask.float().mean()),
        "kd_esm_coverage": float(batch["esm_mask"].float().mean()),
        "kd_protrek_coverage": float(batch["protrek_mask"].float().mean()),
    }
    return total, parts
