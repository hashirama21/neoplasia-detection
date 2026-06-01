"""
Gated Attention Multiple Instance Learning (Ilse et al., 2018).

Treats a stacked TIFF (bag of frames) as a MIL bag: the bag is positive if
at least one frame shows neoplasia. The attention mechanism learns to upweight
diagnostically relevant frames without per-frame supervision.

Training workflow:
  1. Freeze backbone (DINOv2 or ResNet50).
  2. Pre-extract frame embeddings → BagDataset.
  3. Train GatedAttentionMIL on (bag_embeddings, bag_label) pairs.
  4. At inference: embed all frames → MIL head → single bag score + attention map.

The attention map lets a clinician inspect which frames triggered the alert.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class GatedAttentionMIL(nn.Module):
    """
    Gated attention pooling over a variable-length bag of frame embeddings.

    Args:
        feature_dim: Embedding dimension of each frame (768 for ViT-B, 2048 for RN50).
        hidden_dim:  Attention bottleneck dimension.
        dropout:     Applied inside both attention gates to prevent memorisation.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Dropout(dropout))
        self.U = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Dropout(dropout))
        self.w = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(
        self, H: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            H: (N_frames, feature_dim) — single bag, variable N.
        Returns:
            logit:   scalar bag-level logit.
            weights: (N_frames, 1) attention weights (sum to 1, interpretable).
        """
        a_v = torch.tanh(self.V(H))          # (N, hidden_dim)
        a_u = torch.sigmoid(self.U(H))        # (N, hidden_dim)
        A = self.w(a_v * a_u)                 # (N, 1)
        A = torch.softmax(A, dim=0)           # normalised over frames
        z = (A * H).sum(dim=0, keepdim=True)  # (1, feature_dim)
        return self.classifier(z).squeeze(), A

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        logger.info("GatedAttentionMIL saved to %s", path)

    def load(self, path: str, device: torch.device | None = None) -> "GatedAttentionMIL":
        state = torch.load(path, map_location=device or "cpu", weights_only=True)
        self.load_state_dict(state)
        return self


class BagDataset(Dataset):
    """
    Dataset of pre-extracted bag embeddings for MIL training.

    Each sample is a bag (variable-length tensor of frame embeddings) and a label.
    Pre-computing embeddings with a frozen backbone avoids redundant forward passes
    and makes each MIL training epoch ~10× faster than end-to-end fine-tuning.

    Expected directory structure after feature extraction:
        features_dir/
            bag_0000.pt   # tensor of shape (N_frames, feature_dim)
            bag_0001.pt
            ...
        labels.csv        # columns: bag_id, label

    Args:
        features_dir: Directory containing per-bag .pt files.
        labels_csv:   CSV with columns [bag_id, label].
    """

    def __init__(self, features_dir: str, labels_csv: str):
        import pandas as pd

        self.features_dir = Path(features_dir)
        df = pd.read_csv(labels_csv)
        assert "bag_id" in df.columns and "label" in df.columns, (
            "labels.csv must have 'bag_id' and 'label' columns."
        )
        self.bag_ids = df["bag_id"].values
        self.labels = df["label"].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.bag_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.features_dir / f"{self.bag_ids[idx]}.pt"
        H = torch.load(path, weights_only=True)  # (N_frames, feature_dim)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return H, label


def extract_bag_features(
    backbone: nn.Module,
    frames: list[np.ndarray],
    transform,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract CLS-token embeddings for a list of frames using a frozen backbone.
    Returns (N_frames, feature_dim) on CPU.

    Typical usage at inference: call once per stacked TIFF, then pass to MIL head.
    """
    from PIL import Image as _PILImage
    backbone.eval()
    tensors = []
    for frame in frames:
        if isinstance(frame, torch.Tensor):
            tensors.append(frame)
        elif isinstance(frame, np.ndarray):
            tensors.append(transform(_PILImage.fromarray(frame)))
        else:
            tensors.append(transform(frame))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        H = backbone(batch)  # (N, feature_dim)
    return H.cpu()
