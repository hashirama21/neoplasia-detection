"""
Unit tests for calibration classes.
AffineCalibrator tests are skipped if psrcal is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestIsotonicCalibrator:
    def _make_data(self):
        rng = np.random.default_rng(42)
        probs  = np.clip(rng.random(200), 1e-4, 1 - 1e-4)
        labels = (probs + rng.normal(0, 0.2, 200) > 0.5).astype(int)
        return probs, labels

    def test_fit_transform_range(self):
        from src.calibration.calibrator import IsotonicCalibrator
        cal = IsotonicCalibrator()
        probs, labels = self._make_data()
        cal.fit(probs, labels)
        out = cal.transform(probs)
        assert np.all((out >= 0) & (out <= 1))

    def test_calibrate_probs_alias(self):
        from src.calibration.calibrator import IsotonicCalibrator
        cal = IsotonicCalibrator()
        probs, labels = self._make_data()
        cal.fit(probs, labels)
        a = cal.transform(probs)
        b = cal.calibrate_probs(probs)
        np.testing.assert_array_equal(a, b)

    def test_not_fitted_raises(self):
        from src.calibration.calibrator import IsotonicCalibrator
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError):
            cal.transform(np.array([0.5]))

    def test_save_load_roundtrip(self, tmp_path):
        from src.calibration.calibrator import IsotonicCalibrator
        cal = IsotonicCalibrator()
        probs, labels = self._make_data()
        cal.fit(probs, labels)
        path = str(tmp_path / "iso.pkl")
        cal.save(path)
        loaded = IsotonicCalibrator()
        loaded.load(path)
        np.testing.assert_array_almost_equal(
            cal.transform(probs), loaded.transform(probs)
        )


@pytest.mark.skipif(
    pytest.importorskip("psrcal", reason="psrcal not installed") is None,
    reason="psrcal not installed",
)
class TestAffineCalibrator:
    def _make_data(self, n=300):
        rng = np.random.default_rng(7)
        probs  = np.clip(rng.random(n), 1e-4, 1 - 1e-4)
        labels = (probs + rng.normal(0, 0.15, n) > 0.5).astype(int)
        return probs, labels

    def test_fit_transform_range(self):
        from src.calibration.calibrator import AffineCalibrator
        cal = AffineCalibrator()
        probs, labels = self._make_data()
        cal.fit(probs, labels)
        out = cal.transform(probs)
        assert np.all((out >= 0) & (out <= 1))

    def test_calibrate_probs_alias(self):
        from src.calibration.calibrator import AffineCalibrator
        cal = AffineCalibrator()
        probs, labels = self._make_data()
        cal.fit(probs, labels)
        a = cal.transform(probs)
        b = cal.calibrate_probs(probs)
        np.testing.assert_array_almost_equal(a, b)

    def test_not_fitted_raises(self):
        from src.calibration.calibrator import AffineCalibrator
        cal = AffineCalibrator()
        with pytest.raises(RuntimeError):
            cal.transform(np.array([0.5]))

    def test_save_load_roundtrip(self, tmp_path):
        from src.calibration.calibrator import AffineCalibrator
        cal = AffineCalibrator()
        probs, labels = self._make_data()
        cal.fit(probs, labels)
        path = str(tmp_path / "affine.pt")
        cal.save(path)
        loaded = AffineCalibrator()
        loaded.load(path)
        np.testing.assert_array_almost_equal(
            cal.transform(probs), loaded.transform(probs), decimal=5
        )

    def test_prior_applied(self):
        """Calibrating with a very low prior should suppress predicted probabilities."""
        from src.calibration.calibrator import AffineCalibrator
        probs, labels = self._make_data()
        cal_low  = AffineCalibrator(priors=(0.999, 0.001))
        cal_high = AffineCalibrator(priors=(0.5, 0.5))
        cal_low.fit(probs, labels)
        cal_high.fit(probs, labels)
        assert cal_low.transform(probs).mean() < cal_high.transform(probs).mean()
