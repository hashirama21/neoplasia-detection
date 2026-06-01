"""
Train GatedAttentionMIL on pre-extracted bag features.

Run extract_features.py first to produce the bag .pt files and labels.csv.

Usage:
    python scripts/train_mil.py \
        --features-dir data/mil_features/ \
        --embed-dim 768 \
        --epochs 50 \
        --out-dir outputs/mil/

Produces:
    outputs/mil/mil_head.pt          ← best checkpoint (copy to WEIGHTS_DIR)
    outputs/mil/training_history.csv
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
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.mil import BagDataset, GatedAttentionMIL
from src.utils.metrics import bootstrap_ppv_at_recall

log = logging.getLogger(__name__)


def _collate_bags(batch: list) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Variable-length bags: return list of tensors + stacked labels."""
    bags, labels = zip(*batch)
    return list(bags), torch.stack(labels)


def train_one_epoch(
    model: GatedAttentionMIL,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for bags, labels in tqdm(loader, desc="Train", leave=False):
        labels = labels.to(device)
        batch_loss = torch.tensor(0.0, device=device)

        for bag, label in zip(bags, labels):
            logit, _ = model(bag.to(device))
            batch_loss = batch_loss + criterion(logit.unsqueeze(0), label.unsqueeze(0))

        batch_loss = batch_loss / len(bags)
        optimizer.zero_grad()
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += batch_loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: GatedAttentionMIL,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_probs, all_labels = [], []

    for bags, labels in loader:
        for bag, label in zip(bags, labels):
            logit, _ = model(bag.to(device))
            prob = float(torch.sigmoid(logit).item())
            all_probs.append(prob)
            all_labels.append(int(label.item()))

    y_true = np.array(all_labels)
    y_score = np.array(all_probs)

    # Use a simple fixed threshold for monitoring (calibration done post-training)
    threshold = 0.5
    results = bootstrap_ppv_at_recall(
        y_true=y_true,
        y_score=y_score,
        threshold=threshold,
        n_iterations=200,    # fast during training; use 1000 for final eval
        prevalence=0.01,
        target_recall=0.90,
        seed=42,
    )
    return {
        "val_ppv_at_90recall": results["median_ppv"],
        "val_recall": results["median_recall"],
        "probs": y_score,
        "labels": y_true,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True, help="Output of extract_features.py")
    parser.add_argument("--embed-dim", type=int, default=768,
                        help="Backbone embedding dimension (768 ViT-B / 2048 RN50)")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(args.features_dir)

    train_dataset = BagDataset(
        features_dir=str(features_dir / "train"),
        labels_csv=str(features_dir / "train" / "labels.csv"),
    )
    val_dataset = BagDataset(
        features_dir=str(features_dir / "val"),
        labels_csv=str(features_dir / "val" / "labels.csv"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,     # bags are already tensors in memory
        collate_fn=_collate_bags,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_bags,
    )

    n_pos = int(sum(train_dataset.labels))
    n_neg = len(train_dataset) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = GatedAttentionMIL(
        feature_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_ppv = -1.0
    patience_counter = 0
    history = []

    log.info(
        "Training MIL — %d train bags (%d pos) / %d val bags",
        len(train_dataset), n_pos, len(val_dataset),
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device)
        scheduler.step()

        ppv = metrics["val_ppv_at_90recall"]
        log.info(
            "E%03d | loss=%.4f | PPV@90R=%.4f | recall=%.4f",
            epoch, train_loss, ppv, metrics["val_recall"],
        )

        history.append({"epoch": epoch, "train_loss": train_loss, **{
            k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)
        }})

        if ppv > best_ppv + 1e-4:
            best_ppv = ppv
            patience_counter = 0
            model.save(str(out_dir / "mil_head.pt"))
            log.info("  ✓ New best — saved mil_head.pt (PPV=%.4f)", best_ppv)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info("Early stopping at epoch %d.", epoch)
                break

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    log.info(
        "MIL training complete. Best PPV@90R=%.4f. Checkpoint: %s",
        best_ppv, out_dir / "mil_head.pt",
    )
    log.info(
        "Next step: copy %s to WEIGHTS_DIR/mil_head.pt to activate MIL inference.",
        out_dir / "mil_head.pt",
    )


if __name__ == "__main__":
    main()
