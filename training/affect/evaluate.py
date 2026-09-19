"""Evaluation: per-benchmark metrics on a loader, plus test-exclusive personalization using the runtime baseline code."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from affect.baseline import SpeakerBaseline  # noqa: E402
from affect.config import BaselineConfig  # noqa: E402
from training.affect.metrics import (  # noqa: E402
    concordance_cc,
    expected_calibration_error,
    macro_f1,
    speaker_bootstrap_ci,
    unweighted_average_recall,
)

AVD_NAMES = ("arousal", "dominance", "valence")


@torch.no_grad()
def collect_predictions(model, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray | list]:
    model.eval()
    rows: dict[str, list] = defaultdict(list)
    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch.wave, batch.lengths)
        rows["ids"].extend(batch.ids)
        rows["speakers"].extend(batch.speakers)
        if "avd" in outputs:
            rows["avd_pred"].append(outputs["avd"].float().cpu().numpy())
        if "cat_logits" in outputs:
            rows["cat_probs"].append(torch.softmax(outputs["cat_logits"].float(), dim=-1).cpu().numpy())
        if "exertion_logit" in outputs:
            rows["exertion_pred"].append(torch.sigmoid(outputs["exertion_logit"].float()).cpu().numpy())
        rows["avd_true"].append(batch.avd.cpu().numpy())
        rows["has_avd"].append(batch.has_avd.cpu().numpy())
        rows["class_probs"].append(batch.class_probs.cpu().numpy())
        rows["has_cat"].append(batch.has_categorical.cpu().numpy())
        rows["exertion_true"].append(batch.exertion.cpu().numpy())
        rows["has_exertion"].append(batch.has_exertion.cpu().numpy())
    out: dict = {"ids": rows["ids"], "speakers": rows["speakers"]}
    for key, value in rows.items():
        if key not in ("ids", "speakers") and value:
            out[key] = np.concatenate(value)
    return out


def metrics_from_predictions(pred: dict, labels: list[str], prefix: str = "dev") -> dict[str, float]:
    metrics: dict[str, float] = {}
    if "avd_pred" in pred:
        mask = pred["has_avd"].astype(bool)
        if mask.sum() >= 2:
            cccs = []
            for i, name in enumerate(AVD_NAMES):
                value = concordance_cc(pred["avd_pred"][mask, i], pred["avd_true"][mask, i])
                metrics[f"{prefix}/ccc_{name}"] = round(value, 4)
                cccs.append(value)
            metrics[f"{prefix}/avg_ccc"] = round(float(np.mean(cccs)), 4)
    if "cat_probs" in pred:
        mask = pred["has_cat"].astype(bool)
        if mask.any():
            probs = pred["cat_probs"][mask]
            target_probs = pred["class_probs"][mask]
            predicted = probs.argmax(axis=1)
            target = target_probs.argmax(axis=1)
            exclude = {labels.index("other")} if "other" in labels else set()
            metrics[f"{prefix}/macro_f1"] = round(macro_f1(predicted, target, len(labels), exclude), 4)
            metrics[f"{prefix}/uar"] = round(unweighted_average_recall(predicted, target, len(labels), exclude), 4)
            metrics[f"{prefix}/ece"] = round(expected_calibration_error(probs, target_probs), 4)
    if "exertion_pred" in pred:
        mask = pred["has_exertion"].astype(bool)
        if mask.any():
            from training.affect.metrics import binary_metrics

            binary = binary_metrics(pred["exertion_pred"][mask], pred["exertion_true"][mask])
            metrics[f"{prefix}/exertion_acc"] = round(binary["accuracy"], 4)
            metrics[f"{prefix}/exertion_auroc"] = round(binary["auroc"], 4)
    if "avd_pred" in pred and pred["has_avd"].any():
        by_speaker: dict[str, list[float]] = defaultdict(list)
        mask = pred["has_avd"].astype(bool)
        for speaker, p, t in zip(np.array(pred["speakers"])[mask], pred["avd_pred"][mask, 0], pred["avd_true"][mask, 0]):
            by_speaker[speaker].append(abs(float(p) - float(t)))
        low, high = speaker_bootstrap_ci({s: [-np.mean(v)] for s, v in by_speaker.items()})
        metrics[f"{prefix}/arousal_neg_mae_ci_low"] = round(low, 4)
        metrics[f"{prefix}/arousal_neg_mae_ci_high"] = round(high, 4)
    return metrics


def evaluate_model(model, loader: DataLoader, labels: list[str], device: torch.device, prefix: str = "dev") -> dict[str, float]:
    return metrics_from_predictions(collect_predictions(model, loader, device), labels, prefix)


def personalization_gain(
    pred: dict,
    population_mean: np.ndarray,
    population_std: np.ndarray,
    config: BaselineConfig | None = None,
    enrollment_fraction: float = 0.3,
    oracle_neutral: bool = True,
) -> dict[str, float]:
    """CCC before/after per-speaker output normalization with test-EXCLUSIVE baselines.

    For each speaker, the first `enrollment_fraction` of their utterances enroll the baseline
    (neutral gating on the true label when oracle_neutral, else on the population z-score). The
    remaining utterances are scored with the runtime SpeakerBaseline z-score → re-projected onto
    the label scale, and CCC is compared with the raw predictions on the same utterances.
    """
    config = config or BaselineConfig(enrollment_target_seconds=1e9, enrollment_floor_seconds=0.0)
    mask = pred["has_avd"].astype(bool)
    ids = np.array(pred["ids"])[mask]
    speakers = np.array(pred["speakers"])[mask]
    avd_pred = pred["avd_pred"][mask]
    avd_true = pred["avd_true"][mask]
    raw_rows, norm_rows, true_rows = [], [], []
    for speaker in np.unique(speakers):
        idx = np.flatnonzero(speakers == speaker)
        if idx.size < 6:
            continue
        n_enroll = max(2, int(len(idx) * enrollment_fraction))
        baseline = SpeakerBaseline(config, population_mean, population_std, user_id=str(speaker))
        for i in idx[:n_enroll]:
            neutral = bool(np.all(np.abs((avd_true[i] - 0.5)) < 0.15)) if oracle_neutral else True
            baseline.observe(avd_pred[i], 1.0, neutral_context=neutral, utterance_id=str(ids[i]))
        for i in idx[n_enroll:]:
            z = baseline.z_scores(avd_pred[i], exclude_ids={str(ids[i])})
            norm_rows.append(population_mean + z * population_std)
            raw_rows.append(avd_pred[i])
            true_rows.append(avd_true[i])
    if not raw_rows:
        return {}
    raw, norm, true = np.stack(raw_rows), np.stack(norm_rows), np.stack(true_rows)
    result: dict[str, float] = {}
    for i, name in enumerate(AVD_NAMES):
        before = concordance_cc(raw[:, i], true[:, i])
        after = concordance_cc(norm[:, i], true[:, i])
        result[f"personalization/ccc_{name}_raw"] = round(before, 4)
        result[f"personalization/ccc_{name}_norm"] = round(after, 4)
        result[f"personalization/delta_{name}"] = round(after - before, 4)
    result["personalization/scored_utterances"] = float(len(raw_rows))
    return result
