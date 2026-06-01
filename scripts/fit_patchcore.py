"""
Fit PatchCore anomaly detector on normal (label=0) training images.

PatchCore builds a memory bank of patch-level features extracted from
healthy Barrett mucosa. At inference, any frame whose patches diverge
from this bank (high nearest-neighbor distance) is flagged as anomalous.

Run AFTER training the classifier checkpoint.

Usage:
    python scripts/fit_patchcore.py \
        --checkpoint outputs/.../checkpoints/best.pt \
        --model-config configs/model/dinov2_gastronet.yaml \
        --train-csv data/train.csv \
        --val-csv data/val_calibration.csv \
        --out-dir weights/

Produces:
    weights/patchcore.pkl         ← PatchCore memory bank
    weights/hybrid_config.json    ← HybridScorer alpha/beta + normalizer bounds
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",   required=True, help="Fine-tuned .pt checkpoint")
    parser.add_argument("--model-config", required=True, help="configs/model/....yaml")
    parser.add_argument("--train-csv",    required=True, help="Training CSV (image_path, label)")
    parser.add_argument("--val-csv",      required=True, help="val_calibration CSV — for normalizer")
    parser.add_argument("--out-dir",      required=True, help="Output directory (e.g. weights/)")
    parser.add_argument("--feature-layer",  default="blocks.9", help="ViT block to hook")
    parser.add_argument("--memory-bank-size", type=int, default=50_000)
    parser.add_argument("--k-neighbors",  type=int, default=5)
    parser.add_argument("--batch-size",   type=int, default=16)
    parser.add_argument("--num-workers",  type=int, default=4)
    parser.add_argument("--alpha",  type=float, default=0.9,
                        help="Classifier weight in Noisy-OR fusion")
    parser.add_argument("--beta",   type=float, default=0.3,
                        help="PatchCore weight in Noisy-OR fusion")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from src.data.dataset import Rare26Dataset, build_val_transforms
    from src.models.patchcore import HybridScorer, PatchCoreDetector
    from src.models.rare26_model import Rare26Model

    device    = torch.device(args.device)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = OmegaConf.load(args.model_config)
    model_cfg.checkpoint_path = ""

    model = Rare26Model(model_cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    model.eval()
    log.info("Model loaded from %s", args.checkpoint)

    data_cfg = OmegaConf.create({
        "image_size": 392,
        "augmentation": {"val": {"resize": 448, "center_crop": 392}},
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    })
    transform = build_val_transforms(data_cfg)

    train_full = Rare26Dataset(args.train_csv, transform=transform)
    normal_indices = [i for i, lbl in enumerate(train_full.labels) if lbl == 0]
    from torch.utils.data import Subset
    normal_ds = Subset(train_full, normal_indices)
    normal_loader = DataLoader(
        normal_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    log.info("Normal training images: %d", len(normal_indices))

    detector = PatchCoreDetector(
        feature_layer=args.feature_layer,
        memory_bank_size=args.memory_bank_size,
        k_neighbors=args.k_neighbors,
    )
    detector.fit(model, normal_loader, device)
    detector.save(str(out_dir / "patchcore.pkl"))

    log.info("Computing PatchCore scores on val_calibration for normalizer fitting...")
    val_ds = Rare26Dataset(args.val_csv, transform=transform)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,          # score() requires batch=1
        shuffle=False,
        num_workers=0,
    )

    anom_scores: list[float] = []
    with torch.no_grad():
        for batch in val_loader:
            img_tensor = batch["image"].to(device)
            score = detector.score(model, img_tensor, device)
            anom_scores.append(score)

    hybrid = HybridScorer(alpha=args.alpha, beta=args.beta)
    hybrid.fit_normalizer(np.array(anom_scores))
    hybrid.save(str(out_dir / "hybrid_config.json"))

    log.info(
        "PatchCore fitted — memory bank: %d patches | "
        "normalizer: [%.4f, %.4f] | alpha=%.2f beta=%.2f",
        len(detector.memory_bank),
        hybrid._score_normalizer[0],
        hybrid._score_normalizer[1],
        args.alpha, args.beta,
    )
    log.info(
        "Next: copy %s and %s to WEIGHTS_DIR to activate hybrid scoring.",
        out_dir / "patchcore.pkl",
        out_dir / "hybrid_config.json",
    )


if __name__ == "__main__":
    main()