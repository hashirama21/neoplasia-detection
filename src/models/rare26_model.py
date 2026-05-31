"""
Rare26Model: ViT-Base DINOv2 GastroNet-5M backbone with lightweight classification head.
Architecture designed for low-prevalence detection (PPV@90Recall metric).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

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


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper around nn.Linear.

    Native implementation — no peft dependency. Freezes the base weight and
    learns two small matrices (A, B) whose product approximates the full weight
    update: ΔW = B @ A * (alpha / rank).
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        d_out, d_in = linear.weight.shape
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.linear.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def merge_weights(self) -> None:
        """Fuse LoRA into the base weight for parameter-free inference."""
        with torch.no_grad():
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        for p in self.linear.parameters():
            p.requires_grad_(True)
        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(False)


class Rare26Model(nn.Module):
    """
    ViT-Base DINOv2 with GastroNet-5M pretrained weights.
    Uses native LoRA for parameter-efficient fine-tuning of the backbone.
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

        if cfg.checkpoint_path:
            if Path(cfg.checkpoint_path).exists():
                self._load_gastronet_weights(cfg.checkpoint_path)
            else:
                logger.warning(
                    "GastroNet-5M checkpoint not found at %s.",
                    cfg.checkpoint_path,
                )
        # empty checkpoint_path = intentional (weights restored via load_state_dict)

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

        # Handle DINO/DINOv2 training checkpoints that wrap the model under a
        # top-level key ('teacher' = EMA network, 'student', 'model', 'state_dict').
        # Without this extraction the entire backbone is missing (174 keys).
        for dino_key in ("teacher", "student", "model", "state_dict"):
            val = state_dict.get(dino_key)
            if isinstance(val, dict) and len(val) > 10:
                logger.info("Extracting '%s' sub-dict from DINO-style checkpoint", dino_key)
                state_dict = val
                break

        # Strip common wrapper prefixes left after extraction
        for prefix in ("backbone.", "model.", "encoder."):
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
        if msg.missing_keys:
            logger.info("Missing keys (first 10): %s", msg.missing_keys[:10])
        if msg.unexpected_keys:
            logger.info("Unexpected keys: %s", msg.unexpected_keys[:10])

    def _apply_lora(self, lora_cfg: DictConfig) -> None:
        """Inject LoRA adapters into target modules of the TIMM ViT backbone.

        Replaces each matching nn.Linear with a LoRALinear that keeps the frozen
        base weight and adds a trainable low-rank delta. All other backbone
        parameters are then frozen so only LoRA adapters and the head are trained.
        """
        target = set(lora_cfg.target_modules)
        replaced = 0

        for name, mod in list(self.backbone.named_modules()):
            leaf_name = name.split(".")[-1]
            if leaf_name in target and isinstance(mod, nn.Linear):
                parent_name, child_name = name.rsplit(".", 1)
                parent = self.backbone.get_submodule(parent_name)
                setattr(
                    parent,
                    child_name,
                    LoRALinear(mod, lora_cfg.rank, lora_cfg.alpha, lora_cfg.dropout),
                )
                replaced += 1

        if replaced == 0:
            logger.warning(
                "LoRA: no modules matched target_modules=%s. "
                "Check backbone module names with: "
                "[n for n, _ in model.backbone.named_modules()]",
                list(target),
            )
            return

        for name, param in self.backbone.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                param.requires_grad_(False)

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.backbone.parameters())
        logger.info(
            "LoRA applied to %d modules — trainable: %d / %d (%.4f%%)",
            replaced, trainable, total, 100.0 * trainable / total,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def get_parameter_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        """Differential learning rates: low LR for backbone (LoRA), high LR for head."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params,     "lr": head_lr},
        ]
