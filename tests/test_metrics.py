"""
Unit tests for RARE26 metrics.
Focus: PPV@90Recall computation and bootstrap simulation.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.metrics import (
    ppv_at_recall,
    bootstrap_ppv_at_recall,
    find_optimal_threshold,
)


class TestPPVAtRecall:
    def test_perfect_classifier(self):
        """Perfect classifier should achieve PPV=1.0."""
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        ppv, threshold = ppv_at_recall(y_true, y_score, target_recall=0.90)
        assert ppv == pytest.approx(1.0, abs=0.01)

    def test_ppv_formula(self):
        """Verify PPV = TP / (TP + FP) at given threshold."""
        # 10 positives, 90 negatives — 9 TP, 10 FP
        y_true = np.array([1] * 10 + [0] * 90)
        scores = np.array([0.9] * 9 + [0.3] + [0.4] * 10 + [0.1] * 80)
        ppv, _ = ppv_at_recall(y_true, scores, target_recall=0.90, threshold=0.35)
        # TP=9, FP=10 → PPV = 9/19
        expected = 9 / 19
        assert ppv == pytest.approx(expected, abs=0.01)

    def test_threshold_achieves_recall(self):
        """Returned threshold must achieve at least target_recall."""
        rng = np.random.default_rng(42)
        y_true = np.array([1] * 20 + [0] * 180)
        y_score = rng.random(200)
        y_score[:20] += 0.3  # positives slightly higher scores
        y_score = y_score.clip(0, 1)

        _, threshold = ppv_at_recall(y_true, y_score, target_recall=0.90)
        preds = (y_score >= threshold).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        recall = tp / (tp + fn + 1e-10)
        assert recall >= 0.89  # small tolerance


class TestBootstrapPPV:
    def test_prevalence_simulation(self):
        """Bootstrap should simulate ~1% prevalence."""
        y_true = np.array([1] * 50 + [0] * 950)
        y_score = np.random.default_rng(42).random(1000)
        y_score[:50] += 0.3
        y_score = y_score.clip(0, 1)

        result = bootstrap_ppv_at_recall(
            y_true=y_true,
            y_score=y_score,
            threshold=0.6,
            n_iterations=100,
            prevalence=0.01,
            seed=42,
        )
        assert "median_ppv" in result
        assert "median_recall" in result
        assert 0.0 <= result["median_ppv"] <= 1.0

    def test_reproducibility(self):
        """Same seed must produce identical results."""
        y_true = np.array([1] * 30 + [0] * 270)
        y_score = np.random.default_rng(0).random(300)

        r1 = bootstrap_ppv_at_recall(y_true, y_score, threshold=0.5, n_iterations=50, seed=7)
        r2 = bootstrap_ppv_at_recall(y_true, y_score, threshold=0.5, n_iterations=50, seed=7)
        assert r1["median_ppv"] == r2["median_ppv"]

    def test_higher_threshold_lower_recall(self):
        """Higher threshold should reduce recall."""
        rng = np.random.default_rng(123)
        y_true = np.array([1] * 20 + [0] * 180)
        y_score = rng.beta(5, 2, 200)  # positives distributed toward 1

        r_low = bootstrap_ppv_at_recall(
            y_true, y_score, threshold=0.2, n_iterations=100, seed=42
        )
        r_high = bootstrap_ppv_at_recall(
            y_true, y_score, threshold=0.9, n_iterations=100, seed=42
        )
        assert r_high["median_recall"] <= r_low["median_recall"]


class TestFindOptimalThreshold:
    def test_returns_valid_threshold(self):
        """Optimal threshold must be in [0, 1]."""
        rng = np.random.default_rng(42)
        y_true = np.array([1] * 15 + [0] * 135)
        y_score = rng.random(150)
        y_score[:15] += 0.4
        y_score = y_score.clip(0, 1)

        result = find_optimal_threshold(
            y_true=y_true,
            y_score=y_score,
            n_iterations=50,
            threshold_step=0.05,
        )
        assert 0.0 <= result["optimal_threshold"] <= 1.0
        assert result["median_ppv"] >= 0.0


class TestAsymmetricLoss:
    def test_loss_reduces_with_good_predictions(self):
        """Loss should be lower when predictions are correct."""
        import torch
        from src.losses.asymmetric_loss import AsymmetricLoss

        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1)

        logits_good = torch.tensor([3.0, -3.0, 3.0, -3.0])
        logits_bad = torch.tensor([-3.0, 3.0, -3.0, 3.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

        loss_good = criterion(logits_good, targets)
        loss_bad = criterion(logits_bad, targets)
        assert loss_good < loss_bad

    def test_loss_nonnegative(self):
        import torch
        from src.losses.asymmetric_loss import AsymmetricLoss

        criterion = AsymmetricLoss()
        logits = torch.randn(32)
        targets = torch.randint(0, 2, (32,)).float()
        loss = criterion(logits, targets)
        assert loss.item() >= 0.0
