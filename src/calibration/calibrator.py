"""
PPVCalibrator — post-training probability calibration for RARE26.

Pipeline:
1. Isotonic Regression calibration (non-parametric, no distributional assumption)
2. Bootstrap threshold search on val_calibration set (ONLY — not val_selection)
3. Export: calibrated probs, optimal threshold, bootstrap metrics

Why Isotonic > Temperature Scaling for RARE26:
- Temperature Scaling assumes well-ordered, nearly-calibrated logits → not guaranteed
  with 158 positives and strong class imbalance.
- Isotonic Regression preserves rank ordering without any parametric assumption.
- More robust under the asymmetric bootstrap evaluation (1% prevalence).
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from omegaconf import DictConfig
from sklearn.isotonic import IsotonicRegression

from src.utils.metrics import bootstrap_ppv_at_recall, find_optimal_threshold

logger = logging.getLogger(__name__)


class IsotonicCalibrator:
    """
    Isotonic Regression calibrator.
    Fits on (raw_probs, labels) from val_calibration set.
    """

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self._fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        self.iso.fit(probs, labels)
        self._fitted = True
        logger.info("Isotonic calibration fitted on %d samples.", len(probs))
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        return self.iso.transform(probs)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.iso, f)
        logger.info("Isotonic calibrator saved to %s", path)

    def load(self, path: str) -> "IsotonicCalibrator":
        with open(path, "rb") as f:
            self.iso = pickle.load(f)
        self._fitted = True
        return self


class TemperatureScalingCalibrator:
    """Temperature Scaling — kept as ablation baseline."""

    def __init__(self, lr: float = 0.01, max_iter: int = 1000):
        self.temperature = 1.5
        self.lr = lr
        self.max_iter = max_iter
        self._fitted = False

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScalingCalibrator":
        import torch
        import torch.nn.functional as F

        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        temp = torch.nn.Parameter(torch.tensor([self.temperature]))
        optimizer = torch.optim.LBFGS([temp], lr=self.lr, max_iter=self.max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = logits_t / temp
            loss = F.binary_cross_entropy_with_logits(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = temp.item()
        self._fitted = True
        logger.info("Temperature Scaling fitted. T=%.4f", self.temperature)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        return 1 / (1 + np.exp(-scaled))


class PPVCalibrator:
    """
    Full calibration pipeline for RARE26.
    Combines probability calibration + optimal threshold search.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        if cfg.method == "isotonic":
            self.calibrator = IsotonicCalibrator()
        elif cfg.method == "temperature":
            self.calibrator = TemperatureScalingCalibrator(
                lr=cfg.temperature.lr, max_iter=cfg.temperature.max_iter
            )
        else:
            raise ValueError(f"Unknown calibration method: {cfg.method}")
        self.optimal_threshold: Optional[float] = None
        self.bootstrap_results: Optional[dict] = None

    def fit_and_optimize(
        self,
        raw_probs: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        """
        Full calibration pipeline on val_calibration set.

        Args:
            raw_probs: Raw sigmoid probabilities from model (pre-calibration)
            labels: Ground truth binary labels
        Returns:
            Calibration results including optimal threshold and bootstrap metrics
        """
        logger.info(
            "Calibrating on %d samples (pos=%d, neg=%d)",
            len(labels), labels.sum(), (labels == 0).sum(),
        )

        self.calibrator.fit(raw_probs, labels)
        calibrated_probs = self.calibrator.transform(raw_probs)

        threshold_results = find_optimal_threshold(
            y_true=labels,
            y_score=calibrated_probs,
            n_iterations=self.cfg.bootstrap.n_iterations,
            prevalence=self.cfg.bootstrap.prevalence,
            target_recall=self.cfg.bootstrap.target_recall,
            threshold_start=self.cfg.threshold_grid.start,
            threshold_stop=self.cfg.threshold_grid.stop,
            threshold_step=self.cfg.threshold_grid.step,
            seed=self.cfg.bootstrap.seed,
        )
        self.optimal_threshold = threshold_results["optimal_threshold"]

        self.bootstrap_results = bootstrap_ppv_at_recall(
            y_true=labels,
            y_score=calibrated_probs,
            threshold=self.optimal_threshold,
            n_iterations=self.cfg.bootstrap.n_iterations,
            prevalence=self.cfg.bootstrap.prevalence,
            target_recall=self.cfg.bootstrap.target_recall,
            seed=self.cfg.bootstrap.seed,
        )

        logger.info(
            "Calibration done | threshold=%.4f | median PPV@90R=%.4f | median recall=%.4f",
            self.optimal_threshold,
            self.bootstrap_results["median_ppv"],
            self.bootstrap_results["median_recall"],
        )
        return {
            "optimal_threshold": self.optimal_threshold,
            **{k: v for k, v in self.bootstrap_results.items()
               if k not in ("ppv_values", "recall_values")},
        }

    def calibrate_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        return self.calibrator.transform(raw_probs)

    def save(self, output_dir: str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(self.calibrator, IsotonicCalibrator):
            self.calibrator.save(str(output_dir / "isotonic_calibrator.pkl"))

        results = {
            "optimal_threshold": self.optimal_threshold,
            "method": self.cfg.method,
        }
        if self.bootstrap_results:
            results.update({
                k: float(v) for k, v in self.bootstrap_results.items()
                if k not in ("ppv_values", "recall_values")
            })
        with open(output_dir / "calibration_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Calibration artifacts saved to %s", output_dir)

    def load(self, output_dir: str) -> None:
        output_dir = Path(output_dir)
        if isinstance(self.calibrator, IsotonicCalibrator):
            self.calibrator.load(str(output_dir / "isotonic_calibrator.pkl"))
        with open(output_dir / "calibration_results.json") as f:
            data = json.load(f)
        self.optimal_threshold = data["optimal_threshold"]
        logger.info(
            "Calibration loaded. Threshold=%.4f", self.optimal_threshold
        )
