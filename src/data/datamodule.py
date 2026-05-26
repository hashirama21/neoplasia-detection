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
from sklearn.model_selection import StratifiedShuffleSplit
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

    def _prepare_val_splits(self) -> None:
        """
        If separate val CSVs don't exist, split the official validation set.
        70% → val_selection, 30% → val_calibration (stratified).
        """
        sel_path = Path(self.cfg.val_selection_csv)
        cal_path = Path(self.cfg.val_calibration_csv)

        if sel_path.exists() and cal_path.exists():
            logger.info("Using existing val_selection and val_calibration splits.")
            return

        # If only one val CSV provided, split it
        val_csvs = [
            p for p in Path(self.cfg.train_csv).parent.glob("val*.csv")
            if "selection" not in str(p) and "calibration" not in str(p)
        ]
        if not val_csvs:
            logger.warning("No validation CSV found. Splits will be empty.")
            return

        val_df = pd.read_csv(val_csvs[0])
        labels = val_df["label"].values

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1 - self.cfg.val_selection_ratio,
            random_state=self.cfg.val_split_seed,
        )
        sel_idx, cal_idx = next(splitter.split(np.zeros(len(labels)), labels))

        val_df.iloc[sel_idx].to_csv(sel_path, index=False)
        val_df.iloc[cal_idx].to_csv(cal_path, index=False)

        logger.info(
            "Val split created — selection: %d, calibration: %d",
            len(sel_idx), len(cal_idx),
        )

    def train_dataloader(self) -> DataLoader:
        dataset = Rare26Dataset(
            csv_path=self.cfg.train_csv,
            transform=build_train_transforms(self.cfg),
        )
        sampler = None
        if self.cfg.weighted_sampler:
            weights = dataset.get_class_weights()
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
