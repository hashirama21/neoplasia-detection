"""
Evaluation script — reproduces official RARE26 bootstrap procedure locally.
Use before final submission to estimate expected leaderboard score.

Usage:
    python scripts/evaluate.py \
        paths.results_dir=outputs/.../results \
        paths.checkpoint_dir=outputs/.../checkpoints
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from src.calibration.calibrator import PPVCalibrator
    from src.data.datamodule import Rare26DataModule
    from src.inference.predictor import EnsemblePredictor
    from src.models.rare26_model import Rare26Model
    from src.utils.metrics import bootstrap_ppv_at_recall

    import glob
    from pathlib import Path

    def _ckpt_score(p: str) -> float:
        try:
            return float(Path(p).stem.rsplit("_", 1)[-1])
        except ValueError:
            return -1.0

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Build ensemble from top checkpoints
    predictor = EnsemblePredictor(cfg.inference, device)
    checkpoints = [
        p for p in sorted(
            glob.glob(f"{cfg.paths.checkpoint_dir}/*.pt"),
            key=_ckpt_score,
            reverse=True,
        )
        if _ckpt_score(p) >= 0
    ][: cfg.inference.ensemble.n_models]

    log.info("Loading %d checkpoints for ensemble...", len(checkpoints))
    for ckpt_path in checkpoints:
        predictor.load_model(ckpt_path, cfg.model)

    # Load calibrator
    calibrator = PPVCalibrator(cfg.calibration)
    calibrator.load(cfg.paths.results_dir)
    predictor.set_calibrator(calibrator)

    # Evaluate on val_selection (not val_calibration — that was used for threshold)
    dm = Rare26DataModule(cfg.data)
    val_loader = dm.val_selection_dataloader()

    log.info("Running ensemble inference on val_selection set...")
    result = predictor.predict_loader(val_loader)

    # Collect labels
    all_labels = []
    for batch in val_loader:
        all_labels.extend(batch["label"].numpy())
    labels = np.array(all_labels)

    # Official bootstrap evaluation
    log.info("Running official bootstrap evaluation (1000 iterations)...")
    bootstrap = bootstrap_ppv_at_recall(
        y_true=labels,
        y_score=result["calibrated_probs"],
        threshold=predictor.threshold,
        n_iterations=1000,
        prevalence=0.01,
        target_recall=0.90,
    )

    log.info("=" * 50)
    log.info("EVALUATION RESULTS (val_selection set)")
    log.info("=" * 50)
    log.info("  Median PPV@90Recall : %.4f", bootstrap["median_ppv"])
    log.info("  Mean PPV            : %.4f", bootstrap["mean_ppv"])
    log.info("  Std PPV             : %.4f", bootstrap["std_ppv"])
    log.info("  P10 PPV             : %.4f", bootstrap["p10_ppv"])
    log.info("  P90 PPV             : %.4f", bootstrap["p90_ppv"])
    log.info("  Median Recall       : %.4f", bootstrap["median_recall"])
    log.info("  Threshold used      : %.4f", predictor.threshold)
    log.info("=" * 50)

    # Save results
    import json
    out_path = Path(cfg.paths.results_dir) / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {k: float(v) for k, v in bootstrap.items() if k not in ("ppv_values", "recall_values")},
            f, indent=2,
        )
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
