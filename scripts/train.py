"""
Main training entry point for RARE26.

Usage:
    # Standard training
    python scripts/train.py

    # Custom config override
    python scripts/train.py training.loss.gamma_neg=6 training.epochs=40

    # Gamma sweep via Hydra multirun
    python scripts/train.py --multirun training.loss.gamma_neg=2,4,6

    # Different model
    python scripts/train.py model=dinov2_gastronet data.batch_size=8
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Reproducibility
    torch.manual_seed(cfg.project.seed)
    torch.cuda.manual_seed_all(cfg.project.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from src.data.datamodule import Rare26DataModule
    from src.models.rare26_model import Rare26Model
    from src.training.trainer import Rare26Trainer

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    # Step 1 (optional): CV gamma sweep to find best gamma_neg
    if cfg.training.cross_validation.enabled:
        log.info("Running CV gamma sweep...")
        trainer = Rare26Trainer(cfg)
        cv_result = trainer.run_cv_gamma_sweep(
            train_csv=cfg.data.train_csv,
            gamma_values=cfg.training.loss.gamma_neg_sweep,
            n_splits=cfg.training.cross_validation.n_splits,
            seed=cfg.training.cross_validation.seed,
        )
        best_gamma = cv_result["best_gamma_neg"]
        log.info("Using gamma_neg=%.1f from CV", best_gamma)
        cfg.training.loss.gamma_neg = best_gamma

    # Step 2: Build datamodule and model
    dm = Rare26DataModule(cfg.data)
    model = Rare26Model(cfg.model).to(device)
    trainer = Rare26Trainer(cfg)

    # Step 3: Train
    log.info("Starting training...")
    result = trainer.fit(
        model=model,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_selection_dataloader(),
        output_dir=cfg.paths.checkpoint_dir,
    )
    log.info("Training complete. Best checkpoint: %s", result["best_checkpoint"])

    # Step 4: Load best checkpoint and calibrate on val_calibration
    log.info("Running post-training calibration on val_calibration set...")
    from src.calibration.calibrator import PPVCalibrator

    best_ckpt = result["best_checkpoint"]
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Collect raw probabilities on val_calibration (STRICT — never val_selection)
    import numpy as np
    cal_loader = dm.val_calibration_dataloader()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in cal_loader:
            images = batch["image"].to(device)
            logits = model(images).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch["label"].numpy())

    calibrator = PPVCalibrator(cfg.calibration)
    cal_results = calibrator.fit_and_optimize(
        raw_probs=np.array(all_probs),
        labels=np.array(all_labels),
    )
    calibrator.save(cfg.paths.results_dir)

    log.info(
        "Final calibrated score | threshold=%.4f | median PPV@90R=%.4f",
        cal_results["optimal_threshold"],
        cal_results["median_ppv"],
    )

    # Return median PPV for Hydra multirun optimization
    return cal_results["median_ppv"]


if __name__ == "__main__":
    main()
