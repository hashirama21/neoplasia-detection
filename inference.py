"""
inference.py — Grand Challenge RARE26 submission entry point.

I/O contract (Grand Challenge):
  Input:  /input/images/stacked-barretts-esophagus-endoscopy/*.tiff
          /input/inputs.json  (interface descriptor)
  Output: /output/stacked-neoplastic-lesion-likelihoods.json
          (JSON array of floats — one calibrated likelihood per frame)

Runtime: --network none, all artefacts in WEIGHTS_DIR.
"""

from __future__ import annotations

import json
import logging
import sys
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

INPUT_PATH   = Path("/input")
OUTPUT_PATH  = Path("/output")
WEIGHTS_DIR  = Path("/opt/ml/weights")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def _load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(content, f, indent=4)


def get_interface_key() -> tuple[str, ...]:
    inputs = _load_json(INPUT_PATH / "inputs.json")
    return tuple(sorted(sv["interface"]["slug"] for sv in inputs))



def load_stacked_tiff(location: Path) -> list[np.ndarray]:
    """Stacked TIFF → list of HxWx3 uint8 arrays (one per frame)."""
    files = glob(str(location / "*.tiff")) + glob(str(location / "*.tif"))
    if not files:
        raise FileNotFoundError(f"No TIFF files in {location}")

    arr = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(files[0]))

    if arr.ndim == 2:
        frames = [np.stack([arr, arr, arr], axis=-1)]
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        frames = [arr[..., :3]]
    elif arr.ndim == 3:
        frames = [np.stack([arr[i], arr[i], arr[i]], axis=-1) for i in range(arr.shape[0])]
    elif arr.ndim == 4:
        frames = [arr[i, ..., :3] for i in range(arr.shape[0])]
    else:
        raise ValueError(f"Unexpected TIFF shape: {arr.shape}")

    result = []
    for f in frames:
        if f.dtype != np.uint8:
            lo, hi = float(f.min()), float(f.max())
            f = ((f - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8) if hi > lo \
                else np.zeros_like(f, dtype=np.uint8)
        result.append(f)
    return result



def build_val_transform(img_size: int = 392, resize_size: int = 448) -> T.Compose:
    return T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_ensemble(device: torch.device) -> list:
    sys.path.insert(0, str(Path(__file__).parent))
    from src.models.rare26_model import Rare26Model
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path(__file__).parent / "configs" / "model" / "dinov2_gastronet.yaml")
    cfg.checkpoint_path = ""  # skip GastroNet init — full weights restored from ckpt below

    checkpoints = sorted(WEIGHTS_DIR.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoints in {WEIGHTS_DIR}")

    models = []
    for ckpt_path in checkpoints:
        model = Rare26Model(cfg).to(device)
        ckpt  = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
        model.eval()
        models.append(model)
        log.info("Loaded: %s", ckpt_path.name)

    log.info("Ensemble ready — %d model(s)", len(models))
    return models


def load_calibrator():
    from src.calibration.calibrator import IsotonicCalibrator
    path = WEIGHTS_DIR / "calibration" / "isotonic_calibrator.pkl"
    if not path.exists():
        log.warning("Calibrator not found — using raw probabilities.")
        return None
    cal = IsotonicCalibrator()
    cal.load(str(path))
    return cal


def load_threshold() -> float:
    path = WEIGHTS_DIR / "calibration" / "calibration_results.json"
    if path.exists():
        t = float(_load_json(path).get("optimal_threshold", 0.5))
        log.info("Threshold loaded from calibration_results.json: %.4f", t)
        return t
    log.warning("calibration_results.json not found — using 0.5")
    return 0.5


@torch.no_grad()
def ensemble_tta_predict(
    models: list,
    image: Image.Image,
    transform: T.Compose,
    device: torch.device,
) -> float:
    """8 deterministic TTA views × N models → mean logit."""
    views = [
        transform(image),
        transform(TF.hflip(image)),
        transform(TF.vflip(image)),
        transform(TF.hflip(TF.vflip(image))),
        transform(TF.rotate(image, 15)),
        transform(TF.rotate(image, -15)),
        transform(TF.adjust_saturation(image, 1.2)),
        transform(TF.adjust_contrast(image, 1.15)),
    ]
    all_logits = []
    for model in models:
        batch  = torch.stack(views).to(device)
        logits = model(batch).squeeze(-1)
        all_logits.append(float(logits.mean().cpu().item()))
    return float(np.mean(all_logits))



def interface_0_handler() -> int:
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    frames     = load_stacked_tiff(INPUT_PATH / "images" / "stacked-barretts-esophagus-endoscopy")
    models     = load_ensemble(device)
    calibrator = load_calibrator()
    threshold  = load_threshold()
    transform  = build_val_transform()

    log.info("Processing %d frame(s)...", len(frames))

    likelihoods: list[float] = []
    for i, frame in enumerate(frames):
        pil_img  = Image.fromarray(frame)
        logit    = ensemble_tta_predict(models, pil_img, transform, device)
        raw_prob = float(1 / (1 + np.exp(-logit)))
        cal_prob = float(calibrator.transform(np.array([raw_prob]))[0]) \
                   if calibrator else raw_prob
        likelihoods.append(round(cal_prob, 6))
        log.info("Frame %d/%d  raw=%.4f  cal=%.4f  pred=%d",
                 i + 1, len(frames), raw_prob, cal_prob, int(cal_prob >= threshold))

    _write_json(
        OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json",
        likelihoods,
    )

    n_pos = sum(p >= threshold for p in likelihoods)
    log.info("Done — %d/%d frame(s) neoplastic (thr=%.4f)", n_pos, len(frames), threshold)
    return 0


def run() -> int:
    key = get_interface_key()
    log.info("Interface: %s", key)
    handler = {
        ("stacked-barretts-esophagus-endoscopy-images",): interface_0_handler,
    }.get(key)
    if handler is None:
        raise ValueError(f"Unknown interface key: {key}")
    return handler()


if __name__ == "__main__":
    raise SystemExit(run())
