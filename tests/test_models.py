"""
Unit tests for model components: GatedAttentionMIL, FrameQualityFilter,
EnsemblePredictor aggregation strategies, PatchCore HybridScorer.
All tests run without GPU and without trained weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


class TestGatedAttentionMIL:
    def test_forward_output_shapes(self):
        from src.models.mil import GatedAttentionMIL
        model = GatedAttentionMIL(feature_dim=64, hidden_dim=32)
        H = torch.randn(7, 64)
        logit, attn = model(H)
        assert logit.shape == torch.Size([]), "logit must be a scalar"
        assert attn.shape == (7, 1), "attention must be (N_frames, 1)"

    def test_attention_sums_to_one(self):
        from src.models.mil import GatedAttentionMIL
        model = GatedAttentionMIL(feature_dim=32, hidden_dim=16)
        for n_frames in [1, 5, 20]:
            H = torch.randn(n_frames, 32)
            _, attn = model(H)
            assert abs(attn.sum().item() - 1.0) < 1e-5, f"attention sum != 1 for n={n_frames}"

    def test_variable_bag_sizes(self):
        from src.models.mil import GatedAttentionMIL
        model = GatedAttentionMIL(feature_dim=16, hidden_dim=8)
        for n in [1, 3, 16, 50]:
            logit, attn = model(torch.randn(n, 16))
            assert attn.shape[0] == n

    def test_save_load(self, tmp_path):
        from src.models.mil import GatedAttentionMIL
        model = GatedAttentionMIL(feature_dim=32, hidden_dim=16)
        path = str(tmp_path / "mil.pt")
        model.save(path)
        loaded = GatedAttentionMIL(feature_dim=32, hidden_dim=16)
        loaded.load(path)
        H = torch.randn(5, 32)
        with torch.no_grad():
            l1, _ = model(H)
            l2, _ = loaded(H)
        assert abs(l1.item() - l2.item()) < 1e-6


class TestFrameQualityFilter:
    def _make_frame(self, h=64, w=64, blur=False, overexposed=False) -> np.ndarray:
        rng = np.random.default_rng(0)
        frame = (rng.random((h, w, 3)) * 255).astype(np.uint8)
        if blur:
            import cv2
            frame = cv2.GaussianBlur(frame, (21, 21), 0)
        if overexposed:
            frame[:] = 255
        return frame

    def test_score_range(self):
        from src.data.frame_filter import FrameQualityFilter
        filt = FrameQualityFilter()
        for kwargs in [{}, {"blur": True}, {"overexposed": True}]:
            score = filt.score(self._make_frame(**kwargs))
            assert 0.0 <= score <= 1.0, f"score out of [0,1] for {kwargs}"

    def test_sharp_scores_higher_than_blurry(self):
        pytest.importorskip("cv2")
        from src.data.frame_filter import FrameQualityFilter
        filt = FrameQualityFilter()
        sharp  = filt.score(self._make_frame(blur=False))
        blurry = filt.score(self._make_frame(blur=True))
        assert sharp > blurry

    def test_overexposed_scores_low(self):
        pytest.importorskip("cv2")
        from src.data.frame_filter import FrameQualityFilter
        filt = FrameQualityFilter()
        score = filt.score(self._make_frame(overexposed=True))
        assert score < 0.05

    def test_filter_returns_top_k(self):
        pytest.importorskip("cv2")
        from src.data.frame_filter import FrameQualityConfig, FrameQualityFilter
        frames = [self._make_frame() for _ in range(10)]
        filt = FrameQualityFilter(FrameQualityConfig(top_k=4))
        result = filt.filter(frames)
        assert len(result) == 4

    def test_filter_fallback_when_all_fail(self):
        pytest.importorskip("cv2")
        from src.data.frame_filter import FrameQualityConfig, FrameQualityFilter
        frames = [self._make_frame(overexposed=True) for _ in range(5)]
        filt = FrameQualityFilter(FrameQualityConfig(min_sharpness_score=0.99))
        result = filt.filter(frames)
        assert len(result) == 5, "must return all frames as fallback"


class TestEnsemblePredictorAggregation:
    def _make_predictor(self, aggregation: str):
        from omegaconf import OmegaConf
        from src.inference.predictor import EnsemblePredictor
        cfg = OmegaConf.create({
            "tta": {"enabled": False},
            "ensemble": {"aggregation": aggregation, "uncertainty_penalty": 0.0},
            "threshold": 0.5,
        })
        return EnsemblePredictor(cfg, torch.device("cpu"))

    def test_mean_logits_range(self):
        pred = self._make_predictor("mean_logits")
        logits = np.array([[2.0, -2.0, 0.0], [1.0, -1.0, 0.5]])
        probs, std = pred._aggregate(logits)
        assert probs.shape == (3,)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_noisy_or_higher_than_mean_for_high_confidence(self):
        pred_nor  = self._make_predictor("noisy_or")
        pred_mean = self._make_predictor("mean_logits")
        logits = np.array([[3.0, 3.0], [-3.0, -3.0]])
        p_nor,  _ = pred_nor._aggregate(logits)
        p_mean, _ = pred_mean._aggregate(logits)
        assert p_nor[0] >= p_mean[0], "Noisy-OR should be >= mean for high-confidence positives"
        assert p_nor[1] <= p_mean[1] + 1e-4, "Noisy-OR should be <= mean for confident negatives"

    def test_uncertainty_is_zero_for_identical_models(self):
        pred = self._make_predictor("noisy_or")
        same_logit = np.array([[1.5], [1.5], [1.5]])
        _, std = pred._aggregate(same_logit)
        assert std[0] < 1e-6

    def test_invalid_aggregation_raises(self):
        from omegaconf import OmegaConf
        from src.inference.predictor import EnsemblePredictor
        cfg = OmegaConf.create({
            "tta": {"enabled": False},
            "ensemble": {"aggregation": "unknown_method", "uncertainty_penalty": 0.0},
            "threshold": 0.5,
        })
        with pytest.raises(ValueError, match="unknown_method"):
            EnsemblePredictor(cfg, torch.device("cpu"))


class TestHybridScorer:
    def test_combine_output_in_range(self):
        from src.models.patchcore import HybridScorer
        scorer = HybridScorer(alpha=0.9, beta=0.3)
        scorer.fit_normalizer(np.array([0.0, 0.5, 1.0, 2.0]))
        for p in [0.0, 0.5, 1.0]:
            for s in [0.0, 1.0, 2.0]:
                result = scorer.combine(p, s)
                assert 0.0 <= result <= 1.0, f"combine({p}, {s}) = {result} out of [0,1]"

    def test_high_anomaly_raises_low_classifier(self):
        from src.models.patchcore import HybridScorer
        scorer = HybridScorer(alpha=0.9, beta=0.3)
        scorer.fit_normalizer(np.array([0.0, 1.0]))
        low  = scorer.combine(classifier_prob=0.1, patchcore_score=0.0)
        high = scorer.combine(classifier_prob=0.1, patchcore_score=1.0)
        assert high > low

    def test_save_load_roundtrip(self, tmp_path):
        from src.models.patchcore import HybridScorer
        scorer = HybridScorer(alpha=0.8, beta=0.25)
        scorer.fit_normalizer(np.array([0.1, 0.5, 0.9]))
        path = str(tmp_path / "hybrid.json")
        scorer.save(path)
        loaded = HybridScorer()
        loaded.load(path)
        assert loaded.alpha == pytest.approx(0.8)
        assert loaded.beta  == pytest.approx(0.25)
        assert loaded._score_normalizer == pytest.approx(scorer._score_normalizer)
