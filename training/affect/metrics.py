"""Evaluation metrics: macro-F1, UAR, CCC, ECE against soft labels, speaker-bootstrap CIs, teacher fidelity."""

from __future__ import annotations

import numpy as np

ECE_BINS = 15


def concordance_cc(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.size < 2:
        return float("nan")
    pm, tm = pred.mean(), target.mean()
    pv, tv = pred.var(), target.var()
    cov = ((pred - pm) * (target - tm)).mean()
    return float(2 * cov / (pv + tv + (pm - tm) ** 2 + 1e-12))


def macro_f1(pred: np.ndarray, target: np.ndarray, num_classes: int, exclude: set[int] = frozenset()) -> float:
    scores = []
    for c in range(num_classes):
        if c in exclude:
            continue
        tp = np.sum((pred == c) & (target == c))
        fp = np.sum((pred == c) & (target != c))
        fn = np.sum((pred != c) & (target == c))
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def unweighted_average_recall(pred: np.ndarray, target: np.ndarray, num_classes: int, exclude: set[int] = frozenset()) -> float:
    recalls = []
    for c in range(num_classes):
        if c in exclude:
            continue
        mask = target == c
        if mask.sum() == 0:
            continue
        recalls.append(float(np.mean(pred[mask] == c)))
    return float(np.mean(recalls)) if recalls else float("nan")


def expected_calibration_error(probs: np.ndarray, target_probs: np.ndarray, bins: int = ECE_BINS) -> float:
    """ECE where the 'correct' signal is the soft-label mass on the predicted class."""
    confidences = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    accuracies = target_probs[np.arange(len(predicted)), predicted]
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidences > low) & (confidences <= high)
        if mask.any():
            ece += mask.mean() * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


def speaker_bootstrap_ci(values_by_speaker: dict[str, list[float]], iterations: int = 500, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean over speakers, resampling speakers with replacement."""
    rng = np.random.default_rng(seed)
    speakers = list(values_by_speaker)
    if not speakers:
        return float("nan"), float("nan")
    means = np.array([np.mean(values_by_speaker[s]) for s in speakers])
    samples = [np.mean(means[rng.integers(0, len(speakers), len(speakers))]) for _ in range(iterations)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def teacher_fidelity(student_avd: np.ndarray, teacher_avd: np.ndarray) -> dict[str, float]:
    return {
        f"fidelity_ccc_{name}": concordance_cc(student_avd[:, i], teacher_avd[:, i])
        for i, name in enumerate(("arousal", "dominance", "valence"))
    }


def binary_metrics(scores: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    target = target.astype(int)
    accuracy = float(np.mean(pred == target)) if target.size else float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = target.sum()
    n_neg = len(target) - n_pos
    auroc = float((ranks[target == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)) if n_pos and n_neg else float("nan")
    return {"accuracy": accuracy, "auroc": auroc}
