"""
Trainer for RARE26.
Supports:
- Differential learning rates (backbone vs head)
- Cross-validation for hyperparameter validation (gamma_neg sweep)
- Early stopping on val PPV@90Recall
- Mixed precision training
- Checkpoint saving (top-k by PPV@90Recall)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.cuda.amp as amp
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import Rare26Dataset, build_train_transforms, build_val_transforms
from src.losses.asymmetric_loss import AsymmetricLoss
from src.models.rare26_model import Rare26Model
from src.utils.metrics import bootstrap_ppv_at_recall

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = (
            value > self.best_value + self.min_delta
            if self.mode == "max"
            else value < self.best_value - self.min_delta
        )
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class CheckpointManager:
    def __init__(self, save_dir: str, top_k: int = 3, monitor: str = "val_ppv"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.monitor = monitor
        self.checkpoints: list[tuple[float, Path]] = []

    def save(self, model: torch.nn.Module, epoch: int, score: float, cfg: DictConfig) -> None:
        path = self.save_dir / f"epoch_{epoch:03d}_{self.monitor}_{score:.4f}.pt"
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "score": score}, path)
        self.checkpoints.append((score, path))
        self.checkpoints.sort(key=lambda x: x[0], reverse=True)
        while len(self.checkpoints) > self.top_k:
            _, worst_path = self.checkpoints.pop()
            if worst_path.exists():
                worst_path.unlink()
        logger.info("Saved checkpoint: %s (score=%.4f)", path.name, score)

    def best_checkpoint(self) -> Optional[Path]:
        return self.checkpoints[0][1] if self.checkpoints else None


class Rare26Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg.training
        self.data_cfg = cfg.data
        self.model_cfg = cfg.model
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    def build_optimizer(self, model: Rare26Model) -> torch.optim.Optimizer:
        param_groups = model.get_parameter_groups(
            backbone_lr=self.cfg.optimizer.backbone_lr,
            head_lr=self.cfg.optimizer.head_lr,
        )
        return torch.optim.AdamW(
            param_groups, weight_decay=self.cfg.optimizer.weight_decay
        )

    def train_one_epoch(
        self,
        model: Rare26Model,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scaler: amp.GradScaler,
        epoch: int,
    ) -> dict:
        model.train()
        total_loss = 0.0
        n_batches = 0
        accumulate = self.cfg.accumulate_grad_batches

        optimizer.zero_grad()
        with tqdm(loader, desc=f"Train E{epoch}", leave=False) as pbar:
            for step, batch in enumerate(pbar):
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                with amp.autocast(enabled=self.cfg.mixed_precision):
                    logits = model(images)
                    loss = criterion(logits, labels) / accumulate

                scaler.scale(loss).backward()

                if (step + 1) % accumulate == 0 or (step + 1) == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.cfg.gradient_clip_val
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                unscaled_loss = loss.item() * accumulate
                total_loss += unscaled_loss
                n_batches += 1
                pbar.set_postfix(loss=f"{unscaled_loss:.4f}")

        return {"train_loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def evaluate(
        self,
        model: Rare26Model,
        loader: DataLoader,
        threshold: float = 0.5,
    ) -> dict:
        model.eval()
        all_logits, all_labels = [], []

        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            logits = model(images).squeeze(-1)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"])

        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        probs = 1 / (1 + np.exp(-logits))

        result = bootstrap_ppv_at_recall(
            y_true=labels,
            y_score=probs,
            threshold=threshold,
            n_iterations=200,
        )
        return {
            "val_ppv_at_90recall": result["median_ppv"],
            "val_median_recall": result["median_recall"],
            "probs": probs,
            "labels": labels,
        }

    def run_cv_gamma_sweep(
        self,
        train_csv: str,
        gamma_values: list[float],
        n_splits: int = 5,
        seed: int = 42,
    ) -> dict:
        """
        Cross-validation sweep over gamma_neg values.
        Returns the gamma_neg that maximizes median PPV@90Recall.
        """
        full_dataset = Rare26Dataset(train_csv, transform=build_val_transforms(self.data_cfg))
        labels = full_dataset.labels

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv_results = {g: [] for g in gamma_values}

        logger.info("Starting CV gamma sweep: %s over %d folds", gamma_values, n_splits)

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            logger.info("Fold %d/%d", fold + 1, n_splits)
            train_sub = Subset(
                Rare26Dataset(train_csv, transform=build_train_transforms(self.data_cfg)),
                train_idx,
            )
            val_sub = Subset(
                Rare26Dataset(train_csv, transform=build_val_transforms(self.data_cfg)),
                val_idx,
            )

            for gamma_neg in gamma_values:
                model = Rare26Model(self.model_cfg).to(self.device)
                criterion = AsymmetricLoss(
                    gamma_neg=gamma_neg,
                    gamma_pos=self.cfg.loss.gamma_pos,
                    clip=self.cfg.loss.clip,
                )
                optimizer = self.build_optimizer(model)
                scaler = amp.GradScaler(enabled=self.cfg.mixed_precision)

                train_loader = DataLoader(train_sub, batch_size=self.data_cfg.batch_size,
                                          shuffle=True, num_workers=4)
                val_loader = DataLoader(val_sub, batch_size=self.data_cfg.batch_size * 2,
                                        shuffle=False, num_workers=4)

                for epoch in range(10):
                    self.train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch)

                metrics = self.evaluate(model, val_loader)
                cv_results[gamma_neg].append(metrics["val_ppv_at_90recall"])
                logger.info(
                    "Fold %d | gamma_neg=%.1f | PPV@90R=%.4f",
                    fold + 1, gamma_neg, metrics["val_ppv_at_90recall"]
                )

        summary = {
            g: {"mean": np.mean(v), "std": np.std(v)} for g, v in cv_results.items()
        }
        best_gamma = max(summary, key=lambda g: summary[g]["mean"])
        logger.info("Best gamma_neg: %.1f (mean PPV=%.4f)", best_gamma, summary[best_gamma]["mean"])
        return {"best_gamma_neg": best_gamma, "cv_summary": summary}

    def fit(
        self,
        model: Rare26Model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
    ) -> dict:
        criterion = AsymmetricLoss(
            gamma_neg=self.cfg.loss.gamma_neg,
            gamma_pos=self.cfg.loss.gamma_pos,
            clip=self.cfg.loss.clip,
        )
        optimizer = self.build_optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.epochs, eta_min=self.cfg.scheduler.eta_min
        )
        scaler = amp.GradScaler(enabled=self.cfg.mixed_precision)
        early_stopping = EarlyStopping(
            patience=self.cfg.early_stopping.patience,
            min_delta=self.cfg.early_stopping.min_delta,
            mode=self.cfg.early_stopping.mode,
        )
        ckpt_manager = CheckpointManager(
            save_dir=str(output_dir),
            top_k=self.cfg.logging.save_top_k,
        )

        history = []
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_metrics = self.train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, epoch
            )
            val_metrics = self.evaluate(model, val_loader)
            scheduler.step()

            ppv = val_metrics["val_ppv_at_90recall"]
            elapsed = time.time() - t0

            logger.info(
                "E%03d | loss=%.4f | PPV@90R=%.4f | recall=%.4f | %.1fs",
                epoch, train_metrics["train_loss"], ppv,
                val_metrics["val_median_recall"], elapsed,
            )

            ckpt_manager.save(model, epoch, ppv, self.cfg)
            history.append({**train_metrics, **val_metrics, "epoch": epoch})

            if early_stopping.step(ppv):
                logger.info("Early stopping at epoch %d", epoch)
                break

        return {"history": history, "best_checkpoint": str(ckpt_manager.best_checkpoint())}