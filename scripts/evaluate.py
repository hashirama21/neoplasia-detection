"""
Evaluation script — reproduces official RARE26 bootstrap procedure locally.
Use before final submission to estimate expected leaderboard score.

Supports two modes:
  1. Multi-seed ensemble (default): auto-discovers seed_*/checkpoints/ under
     project.output_dir, loads the best checkpoint per seed, recalibrates on
     val_calibration using the ENSEMBLE outputs (calibrate_after_ensemble).
  2. Single-seed: pass paths.checkpoint_dir and paths.results_dir explicitly.

Usage (multi-seed):
    python scripts/evaluate.py \
        project.output_dir=/root/outputs \
        paths.data_dir=/root/data \
        paths.weights_dir=/root/rare26/weights \
        device=cuda

Usage (single-seed):
    python scripts/evaluate.py \
        paths.checkpoint_dir=/root/outputs/seed_42/checkpoints \
        paths.results_dir=/root/outputs/seed_42/results \
        paths.data_dir=/root/data \
        paths.weights_dir=/root/rare26/weights \
        device=cuda
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import glob
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def _ckpt_score(p: str) -> float:
    try:
        return float(Path(p).stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1.0


def _discover_seed_checkpoints(output_dir: str) -> list[str]:
    """Return the best checkpoint from each seed_* subdirectory."""
    seed_dirs = sorted(glob.glob(f"{output_dir}/seed_*/checkpoints"))
    best_ckpts = []
    for seed_dir in seed_dirs:
        candidates = [
            p for p in glob.glob(f"{seed_dir}/*.pt")
            if _ckpt_score(p) >= 0
        ]
        if candidates:
            best = max(candidates, key=_ckpt_score)
            best_ckpts.append(best)
            log.info("Seed ckpt: %s (score=%.4f)", Path(best).name, _ckpt_score(best))
    return best_ckpts


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from src.calibration.calibrator import PPVCalibrator
    from src.data.datamodule import Rare26DataModule
    from src.inference.predictor import EnsemblePredictor
    from src.utils.metrics import bootstrap_ppv_at_recall

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    predictor = EnsemblePredictor(cfg.inference, device)

    # Try multi-seed layout first (seed_*/checkpoints/*.pt under output_dir)
    checkpoints = _discover_seed_checkpoints(cfg.project.output_dir)

    if not checkpoints:
        # Fallback: single checkpoint_dir
        checkpoints = sorted(
            [p for p in glob.glob(f"{cfg.paths.checkpoint_dir}/*.pt")
             if _ckpt_score(p) >= 0],
            key=_ckpt_score,
            reverse=True,
        )[: cfg.inference.ensemble.n_models]

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found under {cfg.project.output_dir}/seed_*/checkpoints/ "
            f"or {cfg.paths.checkpoint_dir}/"
        )

    log.info("Loading %d checkpoints for ensemble...", len(checkpoints))
    for ckpt_path in checkpoints:
        predictor.load_model(ckpt_path, cfg.model)

    dm = Rare26DataModule(cfg.data)

    log.info("Recalibrating ensemble on val_calibration set...")
    # Collect labels in a fast dedicated pass first, then run expensive model inference.
    # Both loaders use shuffle=False so order is guaranteed identical.
    cal_labels = np.concatenate([
        batch["label"].numpy() for batch in dm.val_calibration_dataloader()
    ])
    cal_result = predictor.predict_loader(dm.val_calibration_dataloader())

    calibrator = PPVCalibrator(cfg.calibration)
    calibrator.fit_and_optimize(cal_result["raw_probs"], cal_labels)
    predictor.set_calibrator(calibrator)

    # Save ensemble calibration artifacts
    ensemble_results_dir = Path(cfg.project.output_dir) / "ensemble" / "results"
    calibrator.save(str(ensemble_results_dir))
    log.info(
        "Calibration artifacts saved to %s — copy to WEIGHTS_DIR/calibration/ "
        "before building the Docker image.",
        ensemble_results_dir,
    )

    # Collect val_selection labels before inference (same rationale as above)
    val_labels = np.concatenate([
        batch["label"].numpy() for batch in dm.val_selection_dataloader()
    ])
    log.info("Running ensemble inference on val_selection set...")
    result = predictor.predict_loader(dm.val_selection_dataloader())

    log.info("Running official bootstrap evaluation (1000 iterations, 1%% prevalence)...")
    bootstrap = bootstrap_ppv_at_recall(
        y_true=val_labels,
        y_score=result["calibrated_probs"],
        threshold=predictor.threshold,
        n_iterations=1000,
        prevalence=0.01,
        target_recall=0.90,
    )

    log.info("=" * 50)
    log.info("ENSEMBLE EVALUATION RESULTS (val_selection)")
    log.info("=" * 50)
    log.info("  Models in ensemble  : %d", len(checkpoints))
    log.info("  Median PPV@90Recall : %.4f", bootstrap["median_ppv"])
    log.info("  Mean PPV            : %.4f", bootstrap["mean_ppv"])
    log.info("  Std PPV             : %.4f", bootstrap["std_ppv"])
    log.info("  P10 PPV             : %.4f", bootstrap["p10_ppv"])
    log.info("  P90 PPV             : %.4f", bootstrap["p90_ppv"])
    log.info("  Median Recall       : %.4f", bootstrap["median_recall"])
    log.info("  Threshold used      : %.4f", predictor.threshold)
    log.info("=" * 50)

    out_path = ensemble_results_dir / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "n_models": len(checkpoints),
                "checkpoints": [str(p) for p in checkpoints],
                **{k: float(v) for k, v in bootstrap.items()
                   if k not in ("ppv_values", "recall_values")},
            },
            f, indent=2,
        )
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()