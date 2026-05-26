"""
PPV@90Recall metric — mirrors the official RARE26 evaluation exactly.

Official procedure:
1. All non-dysplastic images included
2. Neoplasia images sampled WITH replacement to simulate ~1% prevalence
3. Repeated 1000 times
4. Final score = MEDIAN PPV@90Recall across all iterations

This module provides:
- Official evaluation simulation (for calibration and local validation)
- Fast batch metric computation (for training monitoring)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def ppv_at_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_recall: float = 0.90,
    threshold: float | None = None,
) -> tuple[float, float]:
    """
    Compute PPV at a fixed recall level.

    Args:
        y_true: Binary ground truth labels
        y_score: Predicted probabilities
        target_recall: Recall level to fix (default 0.90)
        threshold: If provided, use this threshold directly.
                   If None, find threshold that achieves target_recall.
    Returns:
        (ppv, actual_threshold)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if threshold is None:
        thresholds = np.sort(np.unique(y_score))[::-1]
        best_threshold = thresholds[0]
        for t in thresholds:
            preds = (y_score >= t).astype(int)
            tp = np.sum((preds == 1) & (y_true == 1))
            fn = np.sum((preds == 0) & (y_true == 1))
            recall = tp / (tp + fn + 1e-10)
            if recall >= target_recall:
                best_threshold = t
                break
        threshold = best_threshold

    preds = (y_score >= threshold).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    ppv = tp / (tp + fp + 1e-10)
    return float(ppv), float(threshold)


def bootstrap_ppv_at_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    n_iterations: int = 1000,
    prevalence: float = 0.01,
    target_recall: float = 0.90,
    seed: int = 42,
) -> dict:
    """
    Official RARE26 bootstrap evaluation procedure.

    Args:
        y_true: Ground truth labels
        y_score: Predicted probabilities (already calibrated)
        threshold: Decision threshold (from calibration)
        n_iterations: Number of bootstrap iterations (official: 1000)
        prevalence: Target prevalence (official: 0.01)
        target_recall: Fixed recall level (official: 0.90)
        seed: Random seed for reproducibility
    Returns:
        dict with median_ppv, mean_ppv, std_ppv, p10_ppv, p90_ppv,
        median_recall, ppv_values, recall_values
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_neg = len(neg_idx)

    if len(pos_idx) == 0:
        logger.warning("No positive samples — PPV bootstrap undefined.")
        return {
            "median_ppv": 0.0, "mean_ppv": 0.0, "std_ppv": 0.0,
            "p10_ppv": 0.0, "p90_ppv": 0.0, "median_recall": 0.0,
            "ppv_values": np.array([]), "recall_values": np.array([]),
        }

    # Target: 1 pos per 100 neg (1% prevalence)
    n_pos_target = max(1, int(round(n_neg * prevalence)))

    ppv_values = []
    recall_values = []

    for _ in range(n_iterations):
        # Sample positives with replacement (as per official procedure)
        sampled_pos = rng.choice(pos_idx, size=n_pos_target, replace=True)
        idx = np.concatenate([neg_idx, sampled_pos])

        yt = y_true[idx]
        ys = y_score[idx]

        preds = (ys >= threshold).astype(int)
        tp = np.sum((preds == 1) & (yt == 1))
        fp = np.sum((preds == 1) & (yt == 0))
        fn = np.sum((preds == 0) & (yt == 1))

        ppv = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        ppv_values.append(ppv)
        recall_values.append(recall)

    ppv_arr = np.array(ppv_values)
    recall_arr = np.array(recall_values)

    return {
        "median_ppv": float(np.median(ppv_arr)),
        "mean_ppv": float(np.mean(ppv_arr)),
        "std_ppv": float(np.std(ppv_arr)),
        "p10_ppv": float(np.percentile(ppv_arr, 10)),
        "p90_ppv": float(np.percentile(ppv_arr, 90)),
        "median_recall": float(np.median(recall_arr)),
        "ppv_values": ppv_arr,
        "recall_values": recall_arr,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_iterations: int = 1000,
    prevalence: float = 0.01,
    target_recall: float = 0.90,
    threshold_start: float = 0.001,
    threshold_stop: float = 0.999,
    threshold_step: float = 0.001,
    seed: int = 42,
) -> dict:
    """
    Grid search for optimal threshold on val_calibration set.
    Optimizes MEDIAN PPV@90Recall under bootstrap simulation.

    Critical: Only call this on val_calibration — never val_selection.
    """
    thresholds = np.arange(threshold_start, threshold_stop, threshold_step)
    best_threshold = 0.5
    best_median_ppv = 0.0
    results = []

    logger.info("Searching optimal threshold over %d candidates...", len(thresholds))

    for t in thresholds:
        result = bootstrap_ppv_at_recall(
            y_true=y_true,
            y_score=y_score,
            threshold=t,
            n_iterations=n_iterations,
            prevalence=prevalence,
            target_recall=target_recall,
            seed=seed,
        )
        median_recall = result["median_recall"]
        # Only consider thresholds that achieve target recall
        if median_recall >= target_recall:
            results.append({"threshold": t, **result})
            if result["median_ppv"] > best_median_ppv:
                best_median_ppv = result["median_ppv"]
                best_threshold = t

    logger.info(
        "Optimal threshold: %.4f → median PPV@90R = %.4f",
        best_threshold, best_median_ppv,
    )
    return {
        "optimal_threshold": best_threshold,
        "median_ppv": best_median_ppv,
        "all_results": results,
    }
