import numpy as np


METRIC_NAMES = (
    "accuracy",
    "sensitivity",
    "specificity",
    "dice",
    "iou",
    "auc_roc",
    "auc_pr",
)


def _roc_auc(y_true, scores):
    positives = int(y_true.sum())
    negatives = int((~y_true).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    s = scores[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    true_positive = np.cumsum(y)[distinct]
    false_positive = (1 + distinct) - true_positive
    tpr = np.r_[0.0, true_positive / positives]
    fpr = np.r_[0.0, false_positive / negatives]
    return float(np.trapezoid(tpr, fpr))


def _average_precision(y_true, scores):
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_true = y_true[order]
    true_positive = np.cumsum(sorted_true)
    precision = true_positive / np.arange(1, len(sorted_true) + 1)
    return float(precision[sorted_true].sum() / positives)


def calculate_metrics(y_true, probability, threshold=0.5, fov_mask=None):
    y_true = np.asarray(y_true, dtype=bool)
    probability = np.asarray(probability, dtype=np.float32)
    if fov_mask is None:
        fov_mask = np.ones_like(y_true, dtype=bool)
    else:
        fov_mask = np.asarray(fov_mask, dtype=bool)

    y_true = y_true[fov_mask].reshape(-1)
    scores = probability[fov_mask].reshape(-1)
    prediction = scores > threshold

    tp = int(np.logical_and(y_true, prediction).sum())
    tn = int(np.logical_and(~y_true, ~prediction).sum())
    fp = int(np.logical_and(~y_true, prediction).sum())
    fn = int(np.logical_and(y_true, ~prediction).sum())
    eps = 1e-8

    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "sensitivity": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "auc_roc": _roc_auc(y_true, scores),
        "auc_pr": _average_precision(y_true, scores),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def summarize_metrics(rows):
    summary = {}
    for name in METRIC_NAMES:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        summary[name] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary
