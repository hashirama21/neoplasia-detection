"""
EnsemblePredictor — TTA + multi-model ensemble for RARE26 inference.

Aggregation strategies (cfg.ensemble.aggregation):
  mean_logits — average logits across models, then sigmoid. Baseline.
  noisy_or    — 1 - Π(1 - p_i). Probabilistically correct for detection:
                "positive if at least one model is convinced."

Uncertainty penalty (cfg.ensemble.uncertainty_penalty, default 0.0):
  Raises the effective threshold by penalty × std(probs), reducing false
  positives on cases where the ensemble disagrees. Set 0.0 to disable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.models.rare26_model import Rare26Model

logger = logging.getLogger(__name__)

_VALID_AGGREGATIONS = ("mean_logits", "noisy_or")


class TTATransforms:
    """8 deterministic TTA views — no randomness at inference time."""

    def __init__(self, base_transform: transforms.Compose):
        self.base = base_transform

    def get_views(self, image: Image.Image) -> list[torch.Tensor]:
        return [
            self.base(image),
            self.base(TF.hflip(image)),
            self.base(TF.vflip(image)),
            self.base(TF.hflip(TF.vflip(image))),
            self.base(TF.rotate(image, 15)),
            self.base(TF.rotate(image, -15)),
            self.base(TF.adjust_saturation(image, 1.2)),
            self.base(TF.adjust_contrast(image, 1.15)),
        ]


class EnsemblePredictor:
    """
    Multi-model ensemble with TTA.

    Pipeline per image:
      1. Each model produces one logit per TTA view → mean over views.
      2. Per-model logits aggregated via strategy (mean_logits | noisy_or).
      3. Calibration applied to aggregated probability.
      4. Threshold applied, optionally penalised by ensemble disagreement.
    """

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.models: list[Rare26Model] = []
        self.calibrator = None
        self.threshold: float = cfg.threshold

        agg = cfg.ensemble.aggregation
        if agg not in _VALID_AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation '{agg}'. Choose from {_VALID_AGGREGATIONS}."
            )

    # ------------------------------------------------------------------ public

    def add_model(self, model: Rare26Model) -> None:
        model.eval()
        model.to(self.device)
        self.models.append(model)
        logger.info("Ensemble size: %d model(s)", len(self.models))

    def load_model(self, checkpoint_path: str, model_cfg: DictConfig) -> None:
        model = Rare26Model(model_cfg).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
        self.add_model(model)

    def set_calibrator(self, calibrator) -> None:
        self.calibrator = calibrator
        self.threshold = calibrator.optimal_threshold
        logger.info("Calibrator set. Threshold=%.4f", self.threshold)

    @torch.no_grad()
    def predict_single(
        self,
        image: Image.Image,
        val_transform: transforms.Compose,
    ) -> dict:
        """Predict a single PIL image. Used in inference.py for Docker."""
        if not self.models:
            raise RuntimeError("No models loaded.")

        tta = TTATransforms(val_transform) if self.cfg.tta.enabled else None
        per_model_logits = np.empty((len(self.models), 1))

        for i, model in enumerate(self.models):
            if tta:
                views = tta.get_views(image)
                batch = torch.stack(views).to(self.device)
                logits = model(batch).squeeze(-1)
                per_model_logits[i, 0] = logits.mean().item()
            else:
                tensor = val_transform(image).unsqueeze(0).to(self.device)
                per_model_logits[i, 0] = model(tensor).squeeze().item()

        probs, std_probs = self._aggregate(per_model_logits)
        cal_probs = self._calibrate(probs)
        prediction = int(self._threshold(cal_probs, std_probs)[0])

        return {
            "raw_prob": float(probs[0]),
            "calibrated_prob": float(cal_probs[0]),
            "uncertainty": float(std_probs[0]),
            "prediction": prediction,
            "threshold": self.threshold,
        }

    @torch.no_grad()
    def predict_loader(self, loader: DataLoader) -> dict:
        """Batch prediction for evaluation on a DataLoader."""
        if not self.models:
            raise RuntimeError("No models loaded.")

        per_model_logit_lists: list[list[np.ndarray]] = [[] for _ in self.models]

        for batch in tqdm(loader, desc="Inference"):
            images = batch["image"].to(self.device, non_blocking=True)
            for i, model in enumerate(self.models):
                if self.cfg.tta.enabled:
                    views = self._tta_tensor_views(images)
                    view_logits = torch.stack(
                        [model(v).squeeze(-1) for v in views]
                    )
                    logits = view_logits.mean(dim=0)
                else:
                    logits = model(images).squeeze(-1)
                per_model_logit_lists[i].append(logits.cpu().numpy())

        # (N_models, N_samples)
        per_model_logits = np.array([
            np.concatenate(ll) for ll in per_model_logit_lists
        ])

        probs, std_probs = self._aggregate(per_model_logits)
        cal_probs = self._calibrate(probs)
        predictions = self._threshold(cal_probs, std_probs)

        return {
            "raw_probs": probs,
            "calibrated_probs": cal_probs,
            "uncertainties": std_probs,
            "predictions": predictions,
            "threshold": self.threshold,
        }

    # ----------------------------------------------------------------- private

    def _aggregate(
        self, per_model_logits: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Aggregate (N_models, N_samples) logits into (N_samples,) probs.
        Also returns std_probs (N_samples,) for uncertainty-aware thresholding.
        """
        probs_per_model = 1.0 / (1.0 + np.exp(-per_model_logits))

        method = self.cfg.ensemble.aggregation
        if method == "mean_logits":
            mean_logit = per_model_logits.mean(axis=0)
            probs = 1.0 / (1.0 + np.exp(-mean_logit))
        else:  # noisy_or
            probs = 1.0 - np.prod(1.0 - probs_per_model, axis=0)

        std_probs = probs_per_model.std(axis=0)
        return probs, std_probs

    def _calibrate(self, probs: np.ndarray) -> np.ndarray:
        if self.calibrator is not None:
            return self.calibrator.calibrate_probs(probs)
        return probs

    def _threshold(
        self, probs: np.ndarray, std_probs: np.ndarray
    ) -> np.ndarray:
        penalty = float(getattr(self.cfg.ensemble, "uncertainty_penalty", 0.0))
        effective_thr = self.threshold + penalty * std_probs
        return (probs >= effective_thr).astype(int)

    def _tta_tensor_views(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [
            x,
            TF.hflip(x),
            TF.vflip(x),
            TF.hflip(TF.vflip(x)),
            TF.rotate(x, 15),
            TF.rotate(x, -15),
            TF.adjust_saturation(x, 1.2),
            TF.adjust_contrast(x, 1.15),
        ]
