"""
Rebuild val_selection and val_calibration CSVs using GroupShuffleSplit by patient_id.

Run this once to fix existing splits generated without patient-level grouping.
The script overwrites val_selection_csv and val_calibration_csv in-place.

Usage:
    python scripts/rebuild_splits.py \
        --val_csv /content/data/val.csv \
        --sel_csv /content/data/val_selection.csv \
        --cal_csv /content/data/val_calibration.csv \
        --sel_ratio 0.70 \
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def extract_patient_id(image_path: str) -> str:
    stem = Path(image_path).stem
    part = stem.split("_")[0]
    return part if part.lower().startswith("pat") else stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--sel_csv", required=True)
    parser.add_argument("--cal_csv", required=True)
    parser.add_argument("--sel_ratio", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.val_csv)
    assert "image_path" in df.columns and "label" in df.columns, \
        "val_csv must have image_path and label columns"

    patient_ids = df["image_path"].apply(extract_patient_id).values
    labels = df["label"].values

    unique_patients = set(patient_ids)
    pos_patients = set(patient_ids[labels == 1])
    log.info("Val set: %d images | %d patients | %d positive patients",
             len(df), len(unique_patients), len(pos_patients))

    splitter = GroupShuffleSplit(n_splits=1, test_size=1 - args.sel_ratio, random_state=args.seed)
    sel_idx, cal_idx = next(splitter.split(np.zeros(len(labels)), labels, groups=patient_ids))

    sel_df = df.iloc[sel_idx]
    cal_df = df.iloc[cal_idx]

    overlap = set(patient_ids[sel_idx]) & set(patient_ids[cal_idx])
    assert not overlap, f"Patient leakage detected: {overlap}"

    sel_df.to_csv(args.sel_csv, index=False)
    cal_df.to_csv(args.cal_csv, index=False)

    log.info("val_selection  → %d images | %d pos | %d patients",
             len(sel_df), sel_df["label"].sum(), len(set(patient_ids[sel_idx])))
    log.info("val_calibration → %d images | %d pos | %d patients",
             len(cal_df), cal_df["label"].sum(), len(set(patient_ids[cal_idx])))
    log.info("No patient overlap confirmed.")


if __name__ == "__main__":
    main()