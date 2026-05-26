"""
Rare26Model: ViT-Base DINOv2 GastroNet-5M backbone with lightweight classification head.
Architecture designed for low-prevalence detection (PPV@90Recall metric).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import timm
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """Lightweight head — deliberately simple to avoid overfitting on 158 positives."""

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Rare26Model(nn.Module):
    """
    ViT-Base DINOv2 with GastroNet-5M pretrained weights.
    Uses LoRA for parameter-efficient fine-tuning of the backbone.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.backbone = timm.create_model(
            cfg.backbone.name,
            pretrained=False,
            num_classes=0,
            img_size=cfg.backbone.img_size,
            dynamic_img_size=True,
        )

        if cfg.checkpoint_path and Path(cfg.checkpoint_path).exists():
            self._load_gastronet_weights(cfg.checkpoint_path)
        else:
            logger.warning(
                "GastroNet-5M checkpoint not found at %s. "
                "Using random initialization.",
                cfg.checkpoint_path,
            )

        if cfg.lora.enabled:
            self._apply_lora(cfg.lora)

        self.head = ClassificationHead(
            embed_dim=cfg.head.embed_dim,
            hidden_dim=cfg.head.hidden_dim,
            dropout=cfg.head.dropout,
        )

    def _load_gastronet_weights(self, checkpoint_path: str) -> None:
        logger.info("Loading GastroNet-5M weights from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Handle various checkpoint formats
        for prefix in ("backbone.", "model.", "encoder."):
            if any(k.startswith(prefix) for k in state_dict.keys()):
                state_dict = {
                    k.replace(prefix, ""): v for k, v in state_dict.items()
                }
                break

        msg = self.backbone.load_state_dict(state_dict, strict=False)
        logger.info("Checkpoint loaded. Missing: %d, Unexpected: %d",
                    len(msg.missing_keys), len(msg.unexpected_keys))

    def _apply_lora(self, lora_cfg: DictConfig) -> None:
        """Apply LoRA to target modules for parameter-efficient fine-tuning."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_cfg.rank,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                target_modules=list(lora_cfg.target_modules),
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.print_trainable_parameters()
        except ImportError:
            logger.warning(
                "peft not installed — falling back to differential learning rates only. "
                "Install with: pip install peft"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def get_parameter_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        """Differential learning rates: low LR for backbone, high LR for head."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
