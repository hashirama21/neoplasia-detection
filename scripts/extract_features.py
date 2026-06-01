"""
Pre-compute backbone features for MIL training.

Groups individual training images by patient ID to form bags.
Saves one .pt file per bag (tensor of shape (N_frames, embed_dim))
and a labels CSV (bag_id, label) for BagDataset.

Usage:
    python scripts/extract_features.py \
        --checkpoint outputs/.../checkpoints/best.pt \
        --model-config configs/model/dinov2_gastronet.yaml \
        --train-csv data/train.csv \
        --val-csv data/val_selection.csv \
        --out-dir data/mil_features/

The patient_id is inferred from the image filename: pat{XX}_... → patXX.
Bag label = 1 if any frame in the bag is positive.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

log = logging.getLogger(__name__)


def build_val_transform(img_size: int = 392, resize_size: int = 448) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_patient_id(image_path: str) -> str:
    """
    Infer bag (patient) ID from filename.
    Rules tried in order:
      1. pat{XX}_...  → patXX          (RARE standard)
      2. {A}_{B}_...  → A_B            (e.g. batch_0_15 → batch_0)
      3. fallback     → full stem
    If all images get unique IDs (no grouping), MIL will train on bags of 1 frame
    and learn nothing useful. Run with --dry-run to preview groupings before fitting.
    """
    stem  = Path(image_path).stem
    parts = stem.split("_")

    if parts[0].lower().startswith("pat"):
        return parts[0]

    if len(parts) >= 2:
        bag_id = f"{parts[0]}_{parts[1]}"
        log.debug("Non-standard filename '%s' → bag_id='%s'", stem, bag_id)
        return bag_id

    log.warning("Cannot infer bag_id from '%s' — using full stem. Check filename convention.", stem)
    return stem


class ImageListDataset(Dataset):
    """Flat list of images with index — used for batched feature extraction."""

    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), idx


def _extract_all_features(
    paths: list[str],
    backbone: torch.nn.Module,
    transform,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
) -> np.ndarray:
    """Extract backbone features for a list of image paths. Returns (N, D)."""
    dataset = ImageListDataset(paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_features = np.empty((len(paths), 0), dtype=np.float32)
    feature_buffer = [None] * len(paths)

    backbone.eval()
    with torch.no_grad():
        for batch_tensors, indices in tqdm(loader, desc="Extracting features"):
            feats = backbone(batch_tensors.to(device)).cpu().numpy()
            for feat, idx in zip(feats, indices.numpy()):
                feature_buffer[idx] = feat

    return np.stack(feature_buffer)


def _build_bags(
    df: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """
    Group image rows by patient_id.
    Returns:
        path_groups : {patient_id: [img_path, ...]}
        labels      : {patient_id: bag_label}  — 1 if any frame is positive
    """
    df["patient_id"] = df["image_path"].apply(extract_patient_id)
    path_groups: dict[str, list[str]] = {}
    labels: dict[str, int] = {}

    for patient_id, group in df.groupby("patient_id"):
        path_groups[patient_id] = group["image_path"].tolist()
        labels[patient_id] = int(group["label"].max())

    return path_groups, labels


def process_csv(
    csv_path: str,
    backbone: torch.nn.Module,
    transform,
    device: torch.device,
    out_dir: Path,
    split_name: str,
    batch_size: int,
    num_workers: int,
) -> None:
    df = pd.read_csv(csv_path)
    path_groups, labels = _build_bags(df)

    all_paths = df["image_path"].tolist()
    log.info("Extracting features for %d images (%s)...", len(all_paths), split_name)
    all_features = _extract_all_features(
        all_paths, backbone, transform, device, batch_size, num_workers
    )
    path_to_feat = {p: all_features[i] for i, p in enumerate(all_paths)}

    bags_dir = out_dir / split_name
    bags_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for patient_id, paths in path_groups.items():
        bag_feats = np.stack([path_to_feat[p] for p in paths])
        torch.save(torch.from_numpy(bag_feats), bags_dir / f"{patient_id}.pt")
        rows.append({"bag_id": patient_id, "label": labels[patient_id]})

    pd.DataFrame(rows).to_csv(bags_dir / "labels.csv", index=False)
    n_pos = sum(r["label"] for r in rows)
    log.info(
        "%s: saved %d bags (%d pos, %d neg) to %s",
        split_name, len(rows), n_pos, len(rows) - n_pos, bags_dir,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned .pt checkpoint")
    parser.add_argument("--model-config", required=True, help="configs/model/....yaml")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from src.models.rare26_model import Rare26Model

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)

    model_cfg = OmegaConf.load(args.model_config)
    model_cfg.checkpoint_path = ""

    model = Rare26Model(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    backbone = model.backbone  # CLS-token extractor
    backbone.eval()

    transform = build_val_transform()

    for csv_path, split_name in [
        (args.train_csv, "train"),
        (args.val_csv, "val"),
    ]:
        process_csv(
            csv_path, backbone, transform, device,
            out_dir, split_name, args.batch_size, args.num_workers,
        )

    log.info("Feature extraction complete. Output: %s", out_dir)


if __name__ == "__main__":
    main()
