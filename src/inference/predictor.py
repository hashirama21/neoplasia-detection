"""
EnsemblePredictor — TTA + multi-model ensemble for RARE26 inference.

Design:
- TTA: 8 deterministic views per image (no randomness at inference time)
- Ensemble: mean of logits across models BEFORE calibration
- Calibration applied ONCE on aggregated logits
- Threshold hardcoded from calibration step
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


class TTATransforms:
    """8 deterministic TTA views — no randomness."""

    def __init__(self, base_transform: transforms.Compose):
        self.base = base_transform

    def get_views(self, image: Image.Image) -> list[torch.Tensor]:
        views = []
        # 1. Original
        views.append(self.base(image))
        # 2. Horizontal flip
        views.append(self.base(TF.hflip(image)))
        # 3. Vertical flip
        views.append(self.base(TF.vflip(image)))
        # 4. Both flips
        views.append(self.base(TF.hflip(TF.vflip(image))))
        # 5. Rotate +15
        views.append(self.base(TF.rotate(image, 15)))
        # 6. Rotate -15
        views.append(self.base(TF.rotate(image, -15)))
        # 7. Saturation jitter (mild)
        views.append(self.base(TF.adjust_saturation(image, 1.2)))
        # 8. Contrast jitter (mild)
        views.append(self.base(TF.adjust_contrast(image, 1.15)))
        return views


class EnsemblePredictor:
    """
    Multi-model ensemble with TTA.
    Aggregation: mean logits across (models × TTA views) → single calibrated score.
    """

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.models: list[Rare26Model] = []
        self.calibrator = None
        self.threshold: float = cfg.threshold

    def add_model(self, model: Rare26Model) -> None:
        model.eval()
        model.to(self.device)
        self.models.append(model)
        logger.info("Ensemble size: %d model(s)", len(self.models))

    def load_model(self, checkpoint_path: str, model_cfg: DictConfig) -> None:
        model = Rare26Model(model_cfg).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=True)
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
        all_logits = []

        for model in self.models:
            if tta:
                views = tta.get_views(image)
                batch = torch.stack(views).to(self.device)
                logits = model(batch).squeeze(-1)
                all_logits.append(logits.mean().item())
            else:
                tensor = val_transform(image).unsqueeze(0).to(self.device)
                logit = model(tensor).squeeze().item()
                all_logits.append(logit)

        mean_logit = np.mean(all_logits)
        raw_prob = float(1 / (1 + np.exp(-mean_logit)))

        if self.calibrator is not None:
            calibrated_prob = float(self.calibrator.calibrate_probs(np.array([raw_prob]))[0])
        else:
            calibrated_prob = raw_prob

        prediction = int(calibrated_prob >= self.threshold)
        return {
            "raw_prob": raw_prob,
            "calibrated_prob": calibrated_prob,
            "prediction": prediction,
            "threshold": self.threshold,
        }

    @torch.no_grad()
    def predict_loader(
        self,
        loader: DataLoader,
        val_transform: Optional[transforms.Compose] = None,
    ) -> dict:
        """Batch prediction for evaluation."""
        if not self.models:
            raise RuntimeError("No models loaded.")

        all_logits_per_model = [[] for _ in self.models]

        for batch in tqdm(loader, desc="Inference"):
            images = batch["image"].to(self.device, non_blocking=True)
            for i, model in enumerate(self.models):
                logits = model(images).squeeze(-1)
                all_logits_per_model[i].append(logits.cpu().numpy())

        # Aggregate: mean logits across models
        ensemble_logits = np.mean(
            [np.concatenate(logits) for logits in all_logits_per_model], axis=0
        )
        raw_probs = 1 / (1 + np.exp(-ensemble_logits))

        if self.calibrator is not None:
            calibrated_probs = self.calibrator.calibrate_probs(raw_probs)
        else:
            calibrated_probs = raw_probs

        predictions = (calibrated_probs >= self.threshold).astype(int)
        return {
            "raw_probs": raw_probs,
            "calibrated_probs": calibrated_probs,
            "predictions": predictions,
            "threshold": self.threshold,
        }
