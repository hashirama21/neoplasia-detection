"""
prepare_data.py — Create train/val split CSVs from raw RARE26 data directory.

Usage:
    python scripts/prepare_data.py \
        --data_dir /path/to/rare26/images \
        --output_dir data/ \
        --val_ratio 0.30 \
        --val_calibration_ratio 0.30

Expected directory structure:
    data_dir/
    ├── train/
    │   ├── normal/          (2937 images)
    │   └── neoplasia/       (158 images)
    └── validation/          (1530 images — optional)
        ├── normal/
        └── neoplasia/
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def collect_images(root: Path) -> pd.DataFrame:
    rows = []
    for label_name, label_val in [("normal", 0), ("neoplasia", 1)]:
        label_dir = root / label_name
        if not label_dir.exists():
            continue
        for img_path in label_dir.rglob("*"):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"image_path": str(img_path), "label": label_val})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="data/")
    parser.add_argument("--val_calibration_ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training set
    train_dir = data_dir / "train"
    if train_dir.exists():
        train_df = collect_images(train_dir)
        train_df.to_csv(output_dir / "train.csv", index=False)
        log.info(
            "Train: %d images (pos=%d, neg=%d)",
            len(train_df), train_df.label.sum(), (train_df.label == 0).sum()
        )

    # Validation set — split into selection + calibration
    val_dir = data_dir / "validation"
    if val_dir.exists():
        val_df = collect_images(val_dir)
        labels = val_df["label"].values
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=args.val_calibration_ratio, random_state=args.seed
        )
        sel_idx, cal_idx = next(splitter.split(np.zeros(len(labels)), labels))
        val_df.iloc[sel_idx].to_csv(output_dir / "val_selection.csv", index=False)
        val_df.iloc[cal_idx].to_csv(output_dir / "val_calibration.csv", index=False)
        val_df.to_csv(output_dir / "val.csv", index=False)
        log.info(
            "Val split: selection=%d, calibration=%d",
            len(sel_idx), len(cal_idx)
        )

    log.info("CSVs written to %s", output_dir)


if __name__ == "__main__":
    main()
