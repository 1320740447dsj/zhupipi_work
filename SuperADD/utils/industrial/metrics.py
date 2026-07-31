"""Metrics for MVTec-style anomaly detection and localization."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from sklearn.metrics import auc, roc_auc_score


def _require_two_classes(labels: np.ndarray, metric_name: str) -> None:
    if np.unique(labels).size != 2:
        raise ValueError(
            f"{metric_name} requires both normal and anomalous labels. "
            "Set evaluate_good_images=true and check the ground-truth masks."
        )


def compute_image_auroc(
    image_labels: Sequence[int],
    anomaly_maps: Sequence[np.ndarray],
) -> float:
    """Compute image-level AUROC using each anomaly map's maximum score."""
    labels = np.asarray(image_labels, dtype=np.uint8)
    _require_two_classes(labels, "I-AUROC")
    scores = np.asarray(
        [float(np.max(anomaly_map)) for anomaly_map in anomaly_maps],
        dtype=np.float64,
    )
    return float(roc_auc_score(labels, scores))


def compute_pixel_auroc(
    ground_truths: Sequence[np.ndarray],
    anomaly_maps: Sequence[np.ndarray],
) -> float:
    """Compute AUROC over all pixels from all normal and anomalous images."""
    labels = np.concatenate(
        [np.asarray(mask, dtype=bool).reshape(-1) for mask in ground_truths]
    )
    _require_two_classes(labels, "P-AUROC")
    scores = np.concatenate(
        [np.asarray(anomaly_map, dtype=np.float32).reshape(-1) for anomaly_map in anomaly_maps]
    )
    return float(roc_auc_score(labels, scores))


def compute_aupro(
    ground_truths: Sequence[np.ndarray],
    anomaly_maps: Sequence[np.ndarray],
    max_fpr: float = 0.30,
    num_thresholds: int = 200,
) -> float:
    """Compute normalized AUPRO over the FPR interval [0, max_fpr].

    PRO is the mean pixel recall over every connected anomalous region. Normal
    pixels from both good and anomalous images contribute to the false-positive
    rate. The returned area is normalized to the range [0, 1].
    """
    if len(ground_truths) != len(anomaly_maps) or not ground_truths:
        raise ValueError("AUPRO requires equally sized, non-empty mask/map lists")
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("max_fpr must be in (0, 1]")
    if num_thresholds < 2:
        raise ValueError("num_thresholds must be at least 2")

    masks = [np.asarray(mask, dtype=bool) for mask in ground_truths]
    maps = [np.asarray(anomaly_map, dtype=np.float32) for anomaly_map in anomaly_maps]
    for index, (mask, anomaly_map) in enumerate(zip(masks, maps)):
        if mask.shape != anomaly_map.shape:
            raise ValueError(
                f"AUPRO shape mismatch at image {index}: "
                f"mask={mask.shape}, anomaly_map={anomaly_map.shape}"
            )
        if not np.isfinite(anomaly_map).all():
            raise ValueError(f"Non-finite anomaly scores at image {index}")

    region_scores: list[np.ndarray] = []
    normal_score_chunks: list[np.ndarray] = []
    for mask, anomaly_map in zip(masks, maps):
        component_count, components = cv2.connectedComponents(
            mask.astype(np.uint8), connectivity=8
        )
        for component_id in range(1, component_count):
            region_scores.append(
                np.sort(anomaly_map[components == component_id].astype(np.float64))
            )
        normal_score_chunks.append(anomaly_map[~mask].astype(np.float64))

    normal_scores = np.sort(np.concatenate(normal_score_chunks))
    normal_pixel_count = normal_scores.size

    if not region_scores:
        raise ValueError("AUPRO requires at least one connected anomalous region")
    if normal_pixel_count == 0:
        raise ValueError("AUPRO requires at least one normal pixel")

    minimum = min(float(anomaly_map.min()) for anomaly_map in maps)
    maximum = max(float(anomaly_map.max()) for anomaly_map in maps)
    thresholds = np.linspace(maximum, minimum, num_thresholds, dtype=np.float64)

    false_positive_rates = [0.0]
    per_region_overlaps = [0.0]
    for threshold in thresholds:
        false_positives = normal_pixel_count - np.searchsorted(
            normal_scores, threshold, side="left"
        )
        overlaps = [
            (scores.size - np.searchsorted(scores, threshold, side="left"))
            / scores.size
            for scores in region_scores
        ]
        false_positive_rates.append(false_positives / normal_pixel_count)
        per_region_overlaps.append(float(np.mean(overlaps)))

    fpr = np.asarray(false_positive_rates, dtype=np.float64)
    pro = np.asarray(per_region_overlaps, dtype=np.float64)
    order = np.argsort(fpr)
    fpr = fpr[order]
    pro = pro[order]

    unique_fpr = np.unique(fpr)
    unique_pro = np.asarray(
        [np.max(pro[fpr == value]) for value in unique_fpr],
        dtype=np.float64,
    )
    below_limit = unique_fpr < max_fpr
    curve_fpr = np.append(unique_fpr[below_limit], max_fpr)
    curve_pro = np.append(
        unique_pro[below_limit],
        np.interp(max_fpr, unique_fpr, unique_pro),
    )
    return float(auc(curve_fpr / max_fpr, curve_pro))
