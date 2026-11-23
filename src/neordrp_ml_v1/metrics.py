"""Small, dependency-light metrics used for development and calibration."""

from __future__ import annotations

import numpy as np


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    if positive == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels)
    precision = true_positive / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / positive)


def threshold_at_fpr(negative_scores: np.ndarray, fpr: float) -> float:
    if not 0 < fpr < 1:
        raise ValueError("fpr must be in (0, 1)")
    negative_scores = np.asarray(negative_scores)
    if not len(negative_scores):
        raise ValueError("No negative scores for FPR threshold")
    return float(np.quantile(negative_scores, 1.0 - fpr, method="higher"))


def recall_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    labels = np.asarray(labels, dtype=bool)
    if not labels.any():
        return float("nan")
    return float((np.asarray(scores)[labels] >= threshold).mean())


def detection_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    threshold_0p1: float | None = None,
    threshold_1: float | None = None,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    prevalence = float(labels.mean()) if len(labels) else float("nan")
    if threshold_0p1 is None:
        threshold_0p1 = threshold_at_fpr(probabilities[labels == 0], 1e-3)
    if threshold_1 is None:
        threshold_1 = threshold_at_fpr(probabilities[labels == 0], 1e-2)
    p10 = 0.1
    neg_weight = (p10 / (1 - p10)) * ((1 - prevalence) / max(prevalence, 1e-8))
    weighted_labels = np.concatenate([labels, labels[labels == 0]])
    weighted_scores = np.concatenate([probabilities, probabilities[labels == 0]])
    weights = np.concatenate([np.ones(len(labels)), np.full(int((labels == 0).sum()), neg_weight)])
    order = np.argsort(-weighted_scores, kind="mergesort")
    sorted_labels = weighted_labels[order].astype(bool)
    sorted_weights = weights[order]
    tp = np.cumsum(sorted_weights * sorted_labels)
    fp = np.cumsum(sorted_weights * ~sorted_labels)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(tp[-1], 1e-12)
    auprc_p10 = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return {
        "n": float(len(labels)),
        "n_pos": float((labels == 1).sum()),
        "n_neg": float((labels == 0).sum()),
        "prevalence": prevalence,
        "auroc": auroc(labels, probabilities),
        "auprc": average_precision(labels, probabilities),
        "auprc_p10": auprc_p10,
        "recall_fpr_1e_3": recall_at_threshold(labels, probabilities, threshold_0p1),
        "recall_fpr_1e_2": recall_at_threshold(labels, probabilities, threshold_1),
        "threshold_fpr_1e_3": float(threshold_0p1),
        "threshold_fpr_1e_2": float(threshold_1),
    }


def macro_f1(predictions: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if not len(labels):
        return float("nan")
    scores = []
    for class_id in range(n_classes):
        predicted = predictions == class_id
        actual = labels == class_id
        tp = int((predicted & actual).sum())
        denominator = int(predicted.sum() + actual.sum())
        if denominator:
            scores.append(2.0 * tp / denominator)
    return float(np.mean(scores)) if scores else float("nan")


def family_metrics(
    logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0
) -> dict[str, float]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if not len(labels):
        return {
            "n": 0.0,
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "nll": float("nan"),
            "ece_15": float("nan"),
            "brier": float("nan"),
        }
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    true_probability = probabilities[np.arange(len(labels)), labels]
    one_hot = np.eye(logits.shape[1])[labels]
    confidence = probabilities.max(axis=1)
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = 0.0
    for bin_id in range(15):
        mask = bins == bin_id
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - (predictions[mask] == labels[mask]).mean())
    return {
        "n": float(len(labels)),
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": macro_f1(predictions, labels, logits.shape[1]),
        "nll": float(-np.log(np.clip(true_probability, 1e-12, 1.0)).mean()),
        "ece_15": float(ece),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
    }


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, minimum: float = 0.05, maximum: float = 20.0
) -> float:
    grid = np.geomspace(minimum, maximum, 241)
    scores = [family_metrics(logits, labels, value)["nll"] for value in grid]
    return float(grid[int(np.nanargmin(scores))])


def boundary_metrics(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    start_labels: np.ndarray,
    end_labels: np.ndarray,
    attention_mask: np.ndarray,
) -> dict[str, float]:
    start_logits = np.asarray(start_logits)
    end_logits = np.asarray(end_logits)
    start_labels = np.asarray(start_labels, dtype=np.int64)
    end_labels = np.asarray(end_labels, dtype=np.int64)
    mask = np.asarray(attention_mask, dtype=bool)
    start_logits = np.where(mask, start_logits, -1e9)
    end_logits = np.where(mask, end_logits, -1e9)
    predicted_start = start_logits.argmax(axis=1)
    predicted_end = end_logits.argmax(axis=1)
    row_ids = np.arange(len(mask))
    start_valid = (
        (start_labels >= 0)
        & (start_labels < mask.shape[1])
        & mask[row_ids, start_labels.clip(0, mask.shape[1] - 1)]
    )
    end_valid = (
        (end_labels >= 0)
        & (end_labels < mask.shape[1])
        & mask[row_ids, end_labels.clip(0, mask.shape[1] - 1)]
    )
    both = start_valid & end_valid
    result = {
        "n": float(len(start_labels)),
        "n_start": float(start_valid.sum()),
        "n_end": float(end_valid.sum()),
        "n_both": float(both.sum()),
    }
    for tolerance in (5, 10, 20, 30):
        result[f"start_e{tolerance}"] = (
            float((np.abs(predicted_start[start_valid] - start_labels[start_valid]) <= tolerance).mean())
            if start_valid.any()
            else float("nan")
        )
        result[f"end_e{tolerance}"] = (
            float((np.abs(predicted_end[end_valid] - end_labels[end_valid]) <= tolerance).mean())
            if end_valid.any()
            else float("nan")
        )
        result[f"both_e{tolerance}"] = (
            float(
                (
                    (np.abs(predicted_start[both] - start_labels[both]) <= tolerance)
                    & (np.abs(predicted_end[both] - end_labels[both]) <= tolerance)
                ).mean()
            )
            if both.any()
            else float("nan")
        )
    result["start_mae"] = (
        float(np.abs(predicted_start[start_valid] - start_labels[start_valid]).mean())
        if start_valid.any()
        else float("nan")
    )
    result["end_mae"] = (
        float(np.abs(predicted_end[end_valid] - end_labels[end_valid]).mean())
        if end_valid.any()
        else float("nan")
    )
    return result


def boundary_prediction_metrics(
    predicted_start: np.ndarray,
    predicted_end: np.ndarray,
    start_labels: np.ndarray,
    end_labels: np.ndarray,
) -> dict[str, float]:
    predicted_start = np.asarray(predicted_start, dtype=np.int64)
    predicted_end = np.asarray(predicted_end, dtype=np.int64)
    start_labels = np.asarray(start_labels, dtype=np.int64)
    end_labels = np.asarray(end_labels, dtype=np.int64)
    start_valid = start_labels >= 0
    end_valid = end_labels >= 0
    both = start_valid & end_valid
    result = {
        "n": float(len(start_labels)),
        "n_start": float(start_valid.sum()),
        "n_end": float(end_valid.sum()),
        "n_both": float(both.sum()),
    }
    for tolerance in (5, 10, 20, 30):
        result[f"start_e{tolerance}"] = (
            float(
                (
                    np.abs(predicted_start[start_valid] - start_labels[start_valid])
                    <= tolerance
                ).mean()
            )
            if start_valid.any()
            else float("nan")
        )
        result[f"end_e{tolerance}"] = (
            float(
                (np.abs(predicted_end[end_valid] - end_labels[end_valid]) <= tolerance).mean()
            )
            if end_valid.any()
            else float("nan")
        )
        result[f"both_e{tolerance}"] = (
            float(
                (
                    (np.abs(predicted_start[both] - start_labels[both]) <= tolerance)
                    & (np.abs(predicted_end[both] - end_labels[both]) <= tolerance)
                ).mean()
            )
            if both.any()
            else float("nan")
        )
    result["start_mae"] = (
        float(np.abs(predicted_start[start_valid] - start_labels[start_valid]).mean())
        if start_valid.any()
        else float("nan")
    )
    result["end_mae"] = (
        float(np.abs(predicted_end[end_valid] - end_labels[end_valid]).mean())
        if end_valid.any()
        else float("nan")
    )
    return result


def interval_prediction_metrics(
    predicted_start: np.ndarray,
    predicted_end: np.ndarray,
    start_labels: np.ndarray,
    end_labels: np.ndarray,
    lengths: np.ndarray,
    internal_margin: int = 10,
) -> dict[str, float]:
    predicted_start = np.asarray(predicted_start, dtype=np.int64)
    predicted_end = np.asarray(predicted_end, dtype=np.int64)
    start_labels = np.asarray(start_labels, dtype=np.int64)
    end_labels = np.asarray(end_labels, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    valid = (start_labels >= 0) & (end_labels >= 0) & (end_labels >= start_labels)
    if not valid.any():
        return {
            "interval_n": 0.0,
            "interval_iou": float("nan"),
            "interval_truth_coverage": float("nan"),
            "interval_prediction_precision": float("nan"),
            "interval_length_ratio": float("nan"),
            "internal_start_e10": float("nan"),
            "internal_end_e10": float("nan"),
        }
    truth_start = start_labels[valid]
    truth_end = end_labels[valid]
    pred_start = predicted_start[valid]
    pred_end = predicted_end[valid]
    intersection = np.maximum(
        0,
        np.minimum(truth_end, pred_end) - np.maximum(truth_start, pred_start) + 1,
    )
    truth_length = np.maximum(1, truth_end - truth_start + 1)
    pred_length = np.maximum(1, pred_end - pred_start + 1)
    union = truth_length + pred_length - intersection
    internal_start = truth_start > internal_margin
    internal_end = truth_end < lengths[valid] - internal_margin
    return {
        "interval_n": float(valid.sum()),
        "interval_iou": float((intersection / np.maximum(union, 1)).mean()),
        "interval_truth_coverage": float((intersection / truth_length).mean()),
        "interval_prediction_precision": float((intersection / pred_length).mean()),
        "interval_length_ratio": float((pred_length / truth_length).mean()),
        "internal_start_e10": (
            float((np.abs(pred_start[internal_start] - truth_start[internal_start]) <= 10).mean())
            if internal_start.any()
            else float("nan")
        ),
        "internal_end_e10": (
            float((np.abs(pred_end[internal_end] - truth_end[internal_end]) <= 10).mean())
            if internal_end.any()
            else float("nan")
        ),
    }


def fit_endpoint_interval_offsets(
    start_residuals: np.ndarray,
    end_residuals: np.ndarray,
    coverage: float = 0.9,
) -> dict[str, float]:
    if not 0 < coverage < 1:
        raise ValueError("coverage must be in (0, 1)")
    tail = (1.0 - coverage) / 2.0
    starts = np.asarray(start_residuals, dtype=np.float64)
    ends = np.asarray(end_residuals, dtype=np.float64)
    if len(starts) == 0 or len(ends) == 0:
        return {
            "start_low": -10.0,
            "start_high": 10.0,
            "end_low": -10.0,
            "end_high": 10.0,
        }
    return {
        "start_low": float(np.quantile(starts, tail)),
        "start_high": float(np.quantile(starts, 1.0 - tail)),
        "end_low": float(np.quantile(ends, tail)),
        "end_high": float(np.quantile(ends, 1.0 - tail)),
    }


def selection_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    total = 0.0
    for key, weight in weights.items():
        if weight:
            value = metrics.get(key, float("nan"))
            if not np.isfinite(value):
                value = 0.0
            total += float(weight) * float(value)
    return total
