"""
Prepare EVC_Barretts_FullSet for integration with the RARE26 training set.

EVC structure assumed:
    EVC_Barretts_FullSet/
        ACHD/           <- positives (label=1)
            patXX/
                *.png
        NDBT/           <- negatives (label=0)
            patXX/
                *.png

Outputs:
    - evc_train.csv       : EVC images assigned to training (all NDBT + most ACHD patients)
    - evc_val_addition.csv: ~50 ACHD from held-out patients to enrich val_calibration
    - train_merged.csv    : original train.csv + evc_train.csv
    - val_calibration_enriched.csv: original val_calibration.csv + evc_val_addition.csv

Usage:
    python scripts/prepare_evc.py \
        --evc_dir /content/EVC_Barretts_FullSet \
        --train_csv /content/data/train.csv \
        --val_cal_csv /content/data/val_calibration.csv \
        --out_dir /content/data \
        --target_cal_pos 50 \
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def find_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]


def extract_patient_id(image_path: Path, label_dir: Path) -> str:
    """
    Try to extract patient_id from the directory name or filename.
    EVC structure: label_dir/patXX/*.png — patient_id = parent folder name.
    Fallback: first underscore-separated token of the stem.
    """
    # Check if immediate parent is a patient folder (e.g. pat01, patient_01, P01)
    parent = image_path.parent
    if parent != label_dir:
        return parent.name
    stem = image_path.stem
    part = stem.split("_")[0]
    return part if re.match(r"(?i)pat|p\d", part) else stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evc_dir", required=True, help="Root of EVC_Barretts_FullSet")
    parser.add_argument("--train_csv", required=True, help="Original train.csv")
    parser.add_argument("--val_cal_csv", required=True, help="val_calibration.csv to enrich")
    parser.add_argument("--out_dir", required=True, help="Output directory for new CSVs")
    parser.add_argument("--target_cal_pos", type=int, default=50,
                        help="Target positive count in val_calibration after enrichment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evc_dir = Path(args.evc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Collect EVC images ---
    rows = []
    for label, label_int in [("ACHD", 1), ("NDBT", 0)]:
        label_dir = evc_dir / label
        if not label_dir.exists():
            log.warning("Directory not found: %s — skipping", label_dir)
            continue
        images = find_images(label_dir)
        for img in images:
            patient_id = extract_patient_id(img, label_dir)
            rows.append({"image_path": str(img), "label": label_int, "patient_id": patient_id})

    evc_df = pd.DataFrame(rows)
    log.info("EVC total: %d images | %d ACHD | %d NDBT | %d patients",
             len(evc_df),
             (evc_df["label"] == 1).sum(),
             (evc_df["label"] == 0).sum(),
             evc_df["patient_id"].nunique())

    # --- Split ACHD patients: reserve some for val_calibration enrichment ---
    achd_df = evc_df[evc_df["label"] == 1].copy()
    achd_patients = achd_df["patient_id"].unique()

    existing_cal = pd.read_csv(args.val_cal_csv)
    current_cal_pos = existing_cal["label"].sum()
    needed = max(0, args.target_cal_pos - current_cal_pos)
    log.info("val_calibration currently has %d positives. Need %d more.", current_cal_pos, needed)

    # Estimate how many ACHD patients to reserve for calibration
    images_per_patient = len(achd_df) / len(achd_patients) if len(achd_patients) > 0 else 1
    patients_needed = max(1, int(np.ceil(needed / images_per_patient)))
    reserve_ratio = min(0.3, patients_needed / len(achd_patients))

    rng = np.random.default_rng(args.seed)
    shuffled = rng.permuted(achd_patients)
    n_reserve = max(1, int(np.ceil(reserve_ratio * len(achd_patients))))
    cal_patients = set(shuffled[:n_reserve])
    train_patients = set(shuffled[n_reserve:])

    achd_cal = achd_df[achd_df["patient_id"].isin(cal_patients)]
    achd_train = achd_df[achd_df["patient_id"].isin(train_patients)]
    ndbt_df = evc_df[evc_df["label"] == 0]

    log.info("ACHD split — train: %d images (%d patients) | calibration: %d images (%d patients)",
             len(achd_train), len(train_patients), len(achd_cal), len(cal_patients))

    assert not (set(achd_train["patient_id"]) & set(achd_cal["patient_id"])), \
        "Patient leakage in ACHD split"

    # --- Build outputs ---
    evc_train = pd.concat([achd_train, ndbt_df], ignore_index=True)[["image_path", "label"]]
    evc_val_addition = achd_cal[["image_path", "label"]]

    original_train = pd.read_csv(args.train_csv)
    train_merged = pd.concat([original_train, evc_train], ignore_index=True)
    val_cal_enriched = pd.concat([existing_cal, evc_val_addition], ignore_index=True)

    # Save
    evc_train.to_csv(out_dir / "evc_train.csv", index=False)
    evc_val_addition.to_csv(out_dir / "evc_val_addition.csv", index=False)
    train_merged.to_csv(out_dir / "train_merged.csv", index=False)
    val_cal_enriched.to_csv(out_dir / "val_calibration_enriched.csv", index=False)

    log.info("train_merged.csv        : %d images | %d pos",
             len(train_merged), train_merged["label"].sum())
    log.info("val_calibration_enriched: %d images | %d pos",
             len(val_cal_enriched), val_cal_enriched["label"].sum())
    log.info("Files written to %s", out_dir)


if __name__ == "__main__":
    main()