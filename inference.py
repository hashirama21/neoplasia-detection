"""
inference.py — Grand Challenge Docker submission entry point for RARE26.

Requirements:
- Self-contained (no internet access at runtime)
- All weights must be bundled in the Docker image
- Reads from /input/, writes to /output/
- Grand Challenge I/O format: images as PNG/JPEG, output as JSON probability scores

Usage (local test):
    ./do_test_run.sh
    # or
    docker run --network=none --gpus all rare26:latest

CRITICAL: threshold is HARDCODED from calibration step.
          Do NOT recompute at runtime.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", "/opt/ml/weights"))

OPTIMAL_THRESHOLD = float(os.environ.get("RARE26_THRESHOLD", "0.42"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_val_transform(img_size: int = 392, resize_size: int = 448) -> T.Compose:
    return T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from bundled checkpoint. No network calls."""
    sys.path.insert(0, str(Path(__file__).parent))
    from src.models.rare26_model import Rare26Model
    from omegaconf import OmegaConf

    model_cfg = OmegaConf.load(Path(__file__).parent / "configs" / "model" / "dinov2_gastronet.yaml")
    model_cfg.checkpoint_path = checkpoint_path
    model = Rare26Model(model_cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_calibrator(calibration_dir: str):
    """Load isotonic calibrator from bundled artifacts."""
    from src.calibration.calibrator import IsotonicCalibrator
    cal = IsotonicCalibrator()
    cal.load(str(Path(calibration_dir) / "isotonic_calibrator.pkl"))
    return cal


def tta_predict(
    model,
    image: Image.Image,
    transform: T.Compose,
    device: torch.device,
    n_views: int = 8,
) -> float:
    """TTA prediction — deterministic 8 views."""
    import torchvision.transforms.functional as TF

    views = [
        transform(image),
        transform(TF.hflip(image)),
        transform(TF.vflip(image)),
        transform(TF.hflip(TF.vflip(image))),
        transform(TF.rotate(image, 15)),
        transform(TF.rotate(image, -15)),
        transform(TF.adjust_saturation(image, 1.2)),
        transform(TF.adjust_contrast(image, 1.15)),
    ][:n_views]

    batch = torch.stack(views).to(device)
    with torch.no_grad():
        logits = model(batch).squeeze(-1)
    return float(logits.mean().cpu().item())


def run_inference() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s | Threshold: %.4f", device, OPTIMAL_THRESHOLD)

    # Find all model checkpoints (ensemble)
    checkpoint_paths = sorted(WEIGHTS_DIR.glob("*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No .pt checkpoints found in {WEIGHTS_DIR}")
    log.info("Loading %d model(s) for ensemble.", len(checkpoint_paths))

    models = [load_model(str(p), device) for p in checkpoint_paths]
    transform = build_val_transform(img_size=392)

    # Load calibrator
    cal_dir = WEIGHTS_DIR / "calibration"
    calibrator = load_calibrator(str(cal_dir)) if (cal_dir / "isotonic_calibrator.pkl").exists() else None
    if calibrator is None:
        log.warning("No calibrator found — using raw sigmoid probabilities.")

    # Find input images
    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_paths = [
        p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in image_extensions
    ]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {INPUT_DIR}")
    log.info("Processing %d image(s)...", len(image_paths))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.error("Cannot open %s: %s", img_path, e)
            continue

        # Ensemble + TTA
        logits = [tta_predict(m, image, transform, device) for m in models]
        mean_logit = sum(logits) / len(logits)
        raw_prob = float(1 / (1 + np.exp(-mean_logit)))

        if calibrator is not None:
            cal_prob = float(calibrator.transform(np.array([raw_prob]))[0])
        else:
            cal_prob = raw_prob

        prediction = int(cal_prob >= OPTIMAL_THRESHOLD)
        results[img_path.name] = {
            "probability": round(cal_prob, 6),
            "prediction": prediction,
            "neoplasia_detected": bool(prediction),
        }
        log.info(
            "%s → prob=%.4f | pred=%d", img_path.name, cal_prob, prediction
        )

    # Write output — Grand Challenge format
    output_file = OUTPUT_DIR / "predictions.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Also write individual score files (some GC evaluators expect per-image files)
    scores_dir = OUTPUT_DIR / "scores"
    scores_dir.mkdir(exist_ok=True)
    for name, res in results.items():
        score_file = scores_dir / f"{Path(name).stem}.json"
        with open(score_file, "w") as f:
            json.dump({"neoplasia-score": res["probability"]}, f)

    log.info("Inference complete. Results saved to %s", OUTPUT_DIR)
    log.info(
        "Summary: %d positive / %d total",
        sum(r["prediction"] for r in results.values()),
        len(results),
    )


if __name__ == "__main__":
    run_inference()
