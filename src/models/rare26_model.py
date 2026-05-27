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
import torch.nn.functional as F
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

    def _interpolate_pos_embed(self, state_dict: dict) -> dict:
        """Bicubic interpolation of pos_embed when checkpoint and model resolutions differ."""
        if "pos_embed" not in state_dict:
            return state_dict

        src = state_dict["pos_embed"]          # (1, N_src+1, D)
        tgt = self.backbone.pos_embed          # (1, N_tgt+1, D)

        if src.shape == tgt.shape:
            return state_dict

        cls_tok   = src[:, :1]                 # (1, 1, D)
        src_patch = src[:, 1:].float()         # (1, N_src, D)
        tgt_n     = tgt.shape[1] - 1
        src_n     = src_patch.shape[1]

        h_src = int(src_n ** 0.5)
        w_src = src_n // h_src          # handles non-square grids
        if h_src * w_src != src_n:
            raise ValueError(
                f"Cannot infer patch grid from N={src_n} tokens "
                f"(tried {h_src}×{w_src}={h_src * w_src}). "
                "Checkpoint may use a non-standard resolution."
            )
        h_tgt = int(tgt_n ** 0.5)
        w_tgt = tgt_n // h_tgt
        if h_tgt * w_tgt != tgt_n:
            raise ValueError(f"Model target patch grid {tgt_n} is not factorizable.")

        src_patch = src_patch.reshape(1, h_src, w_src, -1).permute(0, 3, 1, 2)
        tgt_patch = F.interpolate(src_patch, size=(h_tgt, w_tgt), mode="bicubic", align_corners=False)
        tgt_patch = tgt_patch.permute(0, 2, 3, 1).reshape(1, h_tgt * w_tgt, -1)

        state_dict["pos_embed"] = torch.cat([cls_tok, tgt_patch], dim=1).to(src.dtype)
        logger.info(
            "pos_embed interpolated %s → %s (src %dx%d → tgt %dx%d patches)",
            list(src.shape), list(state_dict["pos_embed"].shape),
            h_src, w_src, h_tgt, w_tgt,
        )
        return state_dict

    def _load_gastronet_weights(self, checkpoint_path: str) -> None:
        logger.info("Loading GastroNet-5M weights from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        for prefix in ("backbone.", "model.", "encoder."):
            if any(k.startswith(prefix) for k in state_dict.keys()):
                state_dict = {
                    (k[len(prefix):] if k.startswith(prefix) else k): v
                    for k, v in state_dict.items()
                }
                break

        state_dict = self._interpolate_pos_embed(state_dict)

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
