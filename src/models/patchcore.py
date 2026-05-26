"""
PatchCore anomaly detection on DINOv2 features.
Used as a complementary branch to the classification head.

Final score = alpha * classifier_prob + beta * (1 - patchcore_distance_normalized)

Note: Validate by ablation before including in final submission.
Biologically variable mucosal tissue may limit the quality of the normalcy bank.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PatchCoreDetector:
    """
    Simplified PatchCore using DINOv2 intermediate features as patch embeddings.
    Memory bank stores patch-level features from normal (non-dysplastic) training images.
    Anomaly score = nearest-neighbor distance in feature space.
    """

    def __init__(
        self,
        feature_layer: str = "blocks.9",
        memory_bank_size: int = 50_000,
        k_neighbors: int = 5,
    ):
        self.feature_layer = feature_layer
        self.memory_bank_size = memory_bank_size
        self.k_neighbors = k_neighbors
        self.memory_bank: Optional[torch.Tensor] = None
        self._fitted = False

    def _extract_patch_features(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> torch.Tensor:
        """Extract patch-level features from intermediate ViT layers."""
        all_features = []
        hooks = []

        def hook_fn(module, input, output):
            # ViT block output: (B, N_patches+1, embed_dim)
            # Skip CLS token (index 0), keep patch tokens
            patch_features = output[:, 1:, :]
            all_features.append(patch_features.detach().cpu())

        # Register hook on target layer
        for name, module in model.backbone.named_modules():
            if name == self.feature_layer:
                hooks.append(module.register_forward_hook(hook_fn))
                break

        if not hooks:
            logger.warning("Layer %s not found — using CLS token only.", self.feature_layer)

        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="PatchCore: extracting features"):
                images = batch["image"].to(device)
                _ = model.backbone(images)

        for h in hooks:
            h.remove()

        if not all_features:
            raise RuntimeError("No features extracted. Check layer name.")

        # Concatenate all patch features: (N_images * N_patches, embed_dim)
        features = torch.cat(all_features, dim=0)  # (B_total, N_patches, D)
        features = features.view(-1, features.shape[-1])  # flatten patches
        return F.normalize(features, dim=-1)

    def fit(
        self,
        model: torch.nn.Module,
        normal_loader: DataLoader,
        device: torch.device,
    ) -> "PatchCoreDetector":
        """Build normalcy memory bank from normal (label=0) training images."""
        logger.info("Building PatchCore memory bank from normal images...")
        features = self._extract_patch_features(model, normal_loader, device)

        # Subsample to memory_bank_size using coreset selection (greedy)
        if len(features) > self.memory_bank_size:
            indices = torch.randperm(len(features))[: self.memory_bank_size]
            features = features[indices]

        self.memory_bank = features
        self._fitted = True
        logger.info("Memory bank built: %d patch features.", len(self.memory_bank))
        return self

    @torch.no_grad()
    def score(
        self,
        model: torch.nn.Module,
        image_tensor: torch.Tensor,
        device: torch.device,
    ) -> float:
        """
        Compute anomaly score for a single image.
        Returns mean distance to k nearest neighbors in memory bank.
        Higher score = more anomalous.
        """
        if not self._fitted:
            raise RuntimeError("PatchCore not fitted. Call fit() first.")

        all_features = []

        def hook_fn(module, input, output):
            all_features.append(output[:, 1:, :].detach().cpu())

        hooks = []
        for name, module in model.backbone.named_modules():
            if name == self.feature_layer:
                hooks.append(module.register_forward_hook(hook_fn))
                break

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        model.eval()
        _ = model.backbone(image_tensor.to(device))
        for h in hooks:
            h.remove()

        if not all_features:
            return 0.0

        query = F.normalize(all_features[0].squeeze(0), dim=-1)  # (N_patches, D)

        # Compute distances to memory bank
        sim = query @ self.memory_bank.T  # (N_patches, M)
        top_k_sim, _ = sim.topk(self.k_neighbors, dim=-1)
        knn_distances = 1.0 - top_k_sim.mean(dim=-1)  # (N_patches,)
        return float(knn_distances.max().item())  # Image-level score: max patch score

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "memory_bank": self.memory_bank,
                "feature_layer": self.feature_layer,
                "k_neighbors": self.k_neighbors,
            }, f)
        logger.info("PatchCore saved to %s", path)

    def load(self, path: str) -> "PatchCoreDetector":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.memory_bank = data["memory_bank"].cpu()
        self.feature_layer = data["feature_layer"]
        self.k_neighbors = data["k_neighbors"]
        self._fitted = True
        return self


class HybridScorer:
    """
    Combine classifier probability and PatchCore anomaly score.
    score_final = alpha * p_classifier + beta * p_anomaly

    Run ablation to determine if PatchCore improves PPV@90Recall before production use.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        self.alpha = alpha
        self.beta = beta
        self._score_normalizer: Optional[tuple[float, float]] = None

    def fit_normalizer(self, patchcore_scores: np.ndarray) -> None:
        """Fit min-max normalizer for PatchCore scores on validation set."""
        self._score_normalizer = (float(patchcore_scores.min()), float(patchcore_scores.max()))

    def normalize_patchcore(self, score: float) -> float:
        if self._score_normalizer is None:
            return score
        s_min, s_max = self._score_normalizer
        return (score - s_min) / (s_max - s_min + 1e-10)

    def combine(self, classifier_prob: float, patchcore_score: float) -> float:
        normalized_pc = self.normalize_patchcore(patchcore_score)
        return self.alpha * classifier_prob + self.beta * normalized_pc
