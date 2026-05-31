"""
DataModule for RARE26.
Implements strict separation of val_selection and val_calibration splits
to prevent information leakage during threshold optimization.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import Rare26Dataset, build_train_transforms, build_val_transforms

logger = logging.getLogger(__name__)


class Rare26DataModule:
    """
    Manages all data splits for RARE26.

    Splits:
    - train: 3095 labeled images (2937 neg / 158 pos)
    - val_selection: 70% of official val set — model selection + early stopping
    - val_calibration: 30% of official val set — threshold optimization ONLY
                       Never used for model selection. Strict information barrier.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._prepare_val_splits()

    @staticmethod
    def _extract_patient_id(image_path: str) -> str:
        """Extract patient_id from filename. Expects patXX_... pattern; falls back to full stem."""
        stem = Path(image_path).stem
        part = stem.split("_")[0]
        return part if part.lower().startswith("pat") else stem

    def _prepare_val_splits(self) -> None:
        """
        Split the official validation set by patient_id (GroupShuffleSplit).
        70% of patients → val_selection, 30% → val_calibration.
        Image-level stratification is not sufficient: DINOv2 can memorize
        patient endoscopic texture, so patients must never span both splits.
        """
        sel_path = Path(self.cfg.val_selection_csv)
        cal_path = Path(self.cfg.val_calibration_csv)

        if sel_path.exists() and cal_path.exists():
            logger.info("Using existing val_selection and val_calibration splits.")
            return

        val_csvs = [
            p for p in Path(self.cfg.train_csv).parent.glob("val*.csv")
            if "selection" not in str(p) and "calibration" not in str(p)
        ]
        if not val_csvs:
            logger.warning("No validation CSV found. Splits will be empty.")
            return

        val_df = pd.read_csv(val_csvs[0])
        patient_ids = val_df["image_path"].apply(self._extract_patient_id).values
        labels = val_df["label"].values

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=1 - self.cfg.val_selection_ratio,
            random_state=self.cfg.val_split_seed,
        )
        sel_idx, cal_idx = next(splitter.split(np.zeros(len(labels)), labels, groups=patient_ids))

        val_df.iloc[sel_idx].to_csv(sel_path, index=False)
        val_df.iloc[cal_idx].to_csv(cal_path, index=False)

        n_sel_pos = labels[sel_idx].sum()
        n_cal_pos = labels[cal_idx].sum()
        n_sel_pat = len(set(patient_ids[sel_idx]))
        n_cal_pat = len(set(patient_ids[cal_idx]))
        logger.info(
            "Val split by patient — selection: %d images (%d pos, %d patients) | "
            "calibration: %d images (%d pos, %d patients)",
            len(sel_idx), n_sel_pos, n_sel_pat,
            len(cal_idx), n_cal_pos, n_cal_pat,
        )

    def train_dataloader(self) -> DataLoader:
        dataset = Rare26Dataset(
            csv_path=self.cfg.train_csv,
            transform=build_train_transforms(self.cfg),
        )
        sampler = None
        if self.cfg.weighted_sampler:
            target_pos_ratio = float(getattr(self.cfg, "sampler_pos_ratio", 0.15))
            weights = dataset.get_class_weights(target_pos_ratio=target_pos_ratio)
            n_samples = int(len(dataset) * self.cfg.oversample_factor)
            sampler = WeightedRandomSampler(
                weights=weights, num_samples=n_samples, replacement=True
            )
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=True,
        )

    def val_selection_dataloader(self) -> DataLoader:
        return self._val_loader(self.cfg.val_selection_csv)

    def val_calibration_dataloader(self) -> DataLoader:
        return self._val_loader(self.cfg.val_calibration_csv)

    def _val_loader(self, csv_path: str) -> DataLoader:
        dataset = Rare26Dataset(
            csv_path=csv_path,
            transform=build_val_transforms(self.cfg),
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
