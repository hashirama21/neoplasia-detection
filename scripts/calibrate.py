"""
Standalone calibration script.
Run after training to recalibrate with a different method or on a different set.

Usage:
    python scripts/calibrate.py \
        paths.checkpoint_dir=outputs/.../checkpoints \
        calibration.method=isotonic
"""

import glob
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def _checkpoint_score(path: str) -> float:
    try:
        return float(Path(path).stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1.0


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from src.calibration.calibrator import PPVCalibrator
    from src.data.datamodule import Rare26DataModule
    from src.models.rare26_model import Rare26Model

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    checkpoints = [
        p for p in sorted(
            glob.glob(f"{cfg.paths.checkpoint_dir}/*.pt"),
            key=_checkpoint_score,
            reverse=True,
        )
        if _checkpoint_score(p) >= 0
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {cfg.paths.checkpoint_dir}")

    log.info("Loading checkpoint: %s", checkpoints[0])
    model = Rare26Model(cfg.model).to(device)
    ckpt = torch.load(checkpoints[0], map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dm = Rare26DataModule(cfg.data)
    cal_loader = dm.val_calibration_dataloader()

    log.info("Collecting probabilities on val_calibration set...")
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in cal_loader:
            images = batch["image"].to(device)
            logits = model(images).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch["label"].numpy())

    if not all_probs:
        raise RuntimeError(
            f"Calibration set is empty — check {cfg.data.val_calibration_csv}"
        )

    calibrator = PPVCalibrator(cfg.calibration)
    results = calibrator.fit_and_optimize(
        raw_probs=np.array(all_probs),
        labels=np.array(all_labels),
    )

    log.info("Calibration complete.")
    log.info("  Optimal threshold : %.4f", results["optimal_threshold"])
    log.info("  Median PPV@90R    : %.4f", results["median_ppv"])
    log.info("  Median Recall     : %.4f", results["median_recall"])
    log.info("  PPV std           : %.4f", results["std_ppv"])

    calibrator.save(cfg.paths.results_dir)
    log.info("Results saved to %s", cfg.paths.results_dir)


if __name__ == "__main__":
    main()
