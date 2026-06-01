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
    DINOv2 ViT-B or ResNet50 backbone with GastroNet-5M pretrained weights.
    LoRA fine-tuning via peft (ViT only); ResNet50 uses full backbone fine-tuning.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # img_size and dynamic_img_size are ViT-specific — ResNet ignores them
        is_vit = any(k in cfg.backbone.name.lower() for k in ("vit", "dino"))
        extra = (
            {"img_size": cfg.backbone.img_size, "dynamic_img_size": True}
            if is_vit and hasattr(cfg.backbone, "img_size")
            else {}
        )
        self.backbone = timm.create_model(
            cfg.backbone.name,
            pretrained=False,
            num_classes=0,
            **extra,
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
        w_src = src_n // h_src
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

        # Extract backbone sub-dict from DINO/DINOv2 training checkpoints
        # (teacher = EMA network, student, model, state_dict are all common wrappers)
        for dino_key in ("teacher", "student", "model", "state_dict"):
            val = state_dict.get(dino_key)
            if isinstance(val, dict) and len(val) > 10:
                logger.info("Extracting '%s' sub-dict from checkpoint", dino_key)
                state_dict = val
                break

        # Strip wrapper prefixes — order matters, most specific first:
        #   MOCOv2  : module.encoder_q.*
        #   SIMCLRv2: module.encoder.0.* (ResNet50 is encoder[0])
        #   DDP     : module.*
        #   GastroNet/DINO: backbone.* / model.* / encoder.*
        for prefix in (
            "module.encoder_q.",
            "module.encoder.0.",
            "module.encoder.",
            "module.",
            "backbone.",
            "model.",
            "encoder.",
        ):
            if any(k.startswith(prefix) for k in state_dict.keys()):
                state_dict = {
                    (k[len(prefix):] if k.startswith(prefix) else k): v
                    for k, v in state_dict.items()
                }
                logger.info("Stripped prefix '%s' from checkpoint keys", prefix)
                break

        state_dict = self._interpolate_pos_embed(state_dict)

        msg = self.backbone.load_state_dict(state_dict, strict=False)
        logger.info(
            "Checkpoint loaded. Missing: %d, Unexpected: %d",
            len(msg.missing_keys), len(msg.unexpected_keys),
        )

    def _apply_lora(self, lora_cfg: DictConfig) -> None:
        """Apply LoRA to TIMM ViT backbone via peft.

        Uses inject_adapter_in_model instead of get_peft_model to avoid wrapping
        the backbone in PeftModel, which injects NLP forward args (input_ids)
        incompatible with timm VisionTransformer.forward(x).
        """
        try:
            from peft import LoraConfig, inject_adapter_in_model
            from peft.tuners.lora import mark_only_lora_as_trainable

            lora_config = LoraConfig(
                r=lora_cfg.rank,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                target_modules=list(lora_cfg.target_modules),
                bias="none",
            )
            self.backbone = inject_adapter_in_model(lora_config, self.backbone)
            mark_only_lora_as_trainable(self.backbone)

            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in self.backbone.parameters())
            logger.info(
                "LoRA applied: trainable params: %d || all params: %d || trainable%%: %.4f",
                trainable, total, 100 * trainable / total,
            )
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
            {"params": head_params,     "lr": head_lr},
        ]
