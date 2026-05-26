"""
Asymmetric Loss (ASL) for extreme class imbalance.
Reference: Ben-Baruch et al., 2021 — "Asymmetric Loss For Multi-Label Classification"
Adapted for binary low-prevalence detection (PPV@90Recall context).

Key insight: gamma_neg >> gamma_pos deflates gradient from easy negatives (healthy images),
forcing the model to focus on the decision boundary near the 158 positive cases.
gamma_neg is a hyperparameter to sweep via cross-validation: {2, 4, 6}.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    Binary Asymmetric Loss.

    Args:
        gamma_neg: Focusing parameter for negatives. Higher = more focus on hard negatives.
                   Sweep: {2, 4, 6} via internal cross-validation.
        gamma_pos: Focusing parameter for positives (usually 0 or 1).
        clip: Probability margin for negative samples (shifts p_neg by +clip, then clamps).
              Prevents gradient from very easy negatives even before focusing.
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output, shape (B,) or (B, 1)
            targets: Binary labels {0, 1}, shape (B,)
        Returns:
            Scalar loss value
        """
        logits = logits.squeeze(-1)
        targets = targets.float()

        probs = torch.sigmoid(logits)

        # Positive term: standard BCE with positive focusing
        loss_pos = targets * torch.log(probs.clamp(min=1e-8)) * (
            (1.0 - probs) ** self.gamma_pos
        )

        # Negative term: clip + focusing to suppress easy negatives
        probs_neg = (probs + self.clip).clamp(max=1.0)
        loss_neg = (1.0 - targets) * torch.log(
            (1.0 - probs_neg).clamp(min=1e-8)
        ) * (probs_neg ** self.gamma_neg)

        loss = -(loss_pos + loss_neg)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """Standard Focal Loss — alternative baseline for ablation."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.squeeze(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()
