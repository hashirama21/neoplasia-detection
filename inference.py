"""
inference.py — Grand Challenge RARE26 submission entry point.

I/O contract (Grand Challenge):
  Input:  /input/images/stacked-barretts-esophagus-endoscopy/*.tiff
          /input/inputs.json  (interface descriptor)
  Output: /output/stacked-neoplastic-lesion-likelihoods.json
          (JSON array of floats — one calibrated likelihood per frame)

Runtime: --network none, all artefacts in WEIGHTS_DIR.

Inference modes (auto-detected from WEIGHTS_DIR contents):
  Ensemble — default. Scores each frame independently with N models + TTA.
             Noisy-OR aggregation across models.
  MIL      — activated when WEIGHTS_DIR/mil_head.pt is present.
             Treats the full TIFF as a bag; GatedAttentionMIL produces a
             single bag-level score broadcast to all frames.

Calibration (auto-detected from WEIGHTS_DIR/calibration/):
  affine_calibrator.pt   → AffineCalibrator (preferred, encodes 1% prior)
  isotonic_calibrator.pkl → IsotonicCalibrator (fallback)
  If neither present: raw model probabilities are used.

Multi-architecture ensemble:
  WEIGHTS_DIR/manifest.json controls which model config to use per checkpoint.
  Fallback (no manifest): all *.pt files → dinov2_gastronet config.
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
from PIL import Image

INPUT_PATH  = Path("/input")
OUTPUT_PATH = Path("/output")
WEIGHTS_DIR = Path("/opt/ml/weights")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))



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
    for frame in frames:
        if frame.dtype != np.uint8:
            lo, hi = float(frame.min()), float(frame.max())
            frame = (
                ((frame - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
                if hi > lo
                else np.zeros_like(frame, dtype=np.uint8)
            )
        result.append(frame)
    return result


def build_val_transform(img_size: int = 392, resize_size: int = 448) -> T.Compose:
    return T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _build_predictor_cfg():
    """Inline OmegaConf config for EnsemblePredictor (no Hydra resolver needed)."""
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "tta": {"enabled": True},
        "ensemble": {
            "aggregation": "noisy_or",
            "uncertainty_penalty": 0.0,
        },
        "threshold": 0.5,
    })


def _load_manifest() -> list[dict]:
    """
    Read WEIGHTS_DIR/manifest.json or infer from *.pt filenames.
    manifest.json format:
      [{"checkpoint": "epoch_002.pt", "model_config": "dinov2_gastronet"}, ...]
    """
    manifest_path = WEIGHTS_DIR / "manifest.json"
    if manifest_path.exists():
        entries = _load_json(manifest_path)
        log.info("Loaded manifest: %d checkpoint(s)", len(entries))
        return entries

    checkpoints = sorted(WEIGHTS_DIR.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoints in {WEIGHTS_DIR}")

    # Naming convention fallback: rn50_*.pt → rn50_gastronet, else dinov2_gastronet
    entries = []
    for p in checkpoints:
        cfg_name = "rn50_gastronet" if p.name.startswith("rn50_") else "dinov2_gastronet"
        entries.append({"checkpoint": p.name, "model_config": cfg_name})
        log.info("Manifest inferred: %s → %s", p.name, cfg_name)
    return entries


def load_ensemble_predictor(device: torch.device):
    """Load all checkpoints into a configured EnsemblePredictor."""
    from omegaconf import OmegaConf
    from src.inference.predictor import EnsemblePredictor

    predictor = EnsemblePredictor(_build_predictor_cfg(), device)
    configs_dir = Path(__file__).parent / "configs" / "model"

    for entry in _load_manifest():
        ckpt_path = WEIGHTS_DIR / entry["checkpoint"]
        model_cfg = OmegaConf.load(configs_dir / f"{entry['model_config']}.yaml")
        # Skip pretrained init — full weights come from the fine-tuned checkpoint
        model_cfg.checkpoint_path = ""
        predictor.load_model(str(ckpt_path), model_cfg)

    log.info("Ensemble ready — %d model(s), aggregation=noisy_or", len(predictor.models))
    return predictor


def _load_calibrator_raw():
    """
    Load and return the raw calibrator object (AffineCalibrator or IsotonicCalibrator).
    Returns (calibrator, threshold). calibrator may be None.
    Single source of truth used by both load_calibration_artifacts and _predict_mil.
    """
    cal_dir      = WEIGHTS_DIR / "calibration"
    results_path = cal_dir / "calibration_results.json"

    threshold = 0.5
    if results_path.exists():
        threshold = float(_load_json(results_path).get("optimal_threshold", 0.5))
        log.info("Threshold loaded: %.4f", threshold)
    else:
        log.warning("calibration_results.json not found — using threshold=0.5")

    affine_path   = cal_dir / "affine_calibrator.pt"
    isotonic_path = cal_dir / "isotonic_calibrator.pkl"

    if affine_path.exists():
        from src.calibration.calibrator import AffineCalibrator
        cal = AffineCalibrator()
        cal.load(str(affine_path))
        log.info("Affine calibrator loaded.")
        return cal, threshold
    if isotonic_path.exists():
        from src.calibration.calibrator import IsotonicCalibrator
        cal = IsotonicCalibrator()
        cal.load(str(isotonic_path))
        log.info("Isotonic calibrator loaded.")
        return cal, threshold

    log.warning("No calibrator found — using raw probabilities.")
    return None, threshold


def load_calibration_artifacts(predictor) -> None:
    """Attach calibrator and threshold to an EnsemblePredictor."""
    cal, threshold = _load_calibrator_raw()
    predictor.calibrator = cal
    predictor.threshold  = threshold


def load_patchcore():
    """
    Load PatchCore + HybridScorer if present in WEIGHTS_DIR.
    Returns (PatchCoreDetector, HybridScorer) or (None, None).
    Hybrid Noisy-OR scoring activates automatically when patchcore.pkl is present.
    """
    pc_path     = WEIGHTS_DIR / "patchcore.pkl"
    hybrid_path = WEIGHTS_DIR / "hybrid_config.json"

    if not pc_path.exists():
        return None, None

    from src.models.patchcore import HybridScorer, PatchCoreDetector
    detector = PatchCoreDetector()
    detector.load(str(pc_path))

    hybrid = HybridScorer()
    if hybrid_path.exists():
        hybrid.load(str(hybrid_path))
    else:
        log.warning("hybrid_config.json not found — using default alpha=0.9, beta=0.3")

    log.info("PatchCore loaded — memory bank: %d patches", len(detector.memory_bank))
    return detector, hybrid


def load_mil_predictor(device: torch.device):
    """
    Load the GatedAttentionMIL head if present in WEIGHTS_DIR.
    Returns (backbone, mil_head) or None when MIL is not available.
    """
    mil_path = WEIGHTS_DIR / "mil_head.pt"
    if not mil_path.exists():
        return None

    log.info("mil_head.pt found — activating MIL inference mode.")

    # Use the first checkpoint in the manifest as the frozen backbone
    manifest = _load_manifest()
    if not manifest:
        log.warning("MIL: no backbone checkpoint found, falling back to ensemble.")
        return None

    from omegaconf import OmegaConf
    from src.models.mil import GatedAttentionMIL
    from src.models.rare26_model import Rare26Model

    configs_dir = Path(__file__).parent / "configs" / "model"
    first_entry = manifest[0]
    model_cfg = OmegaConf.load(configs_dir / f"{first_entry['model_config']}.yaml")
    model_cfg.checkpoint_path = ""

    backbone_model = Rare26Model(model_cfg).to(device)
    ckpt = torch.load(
        str(WEIGHTS_DIR / first_entry["checkpoint"]),
        map_location=device,
        weights_only=True,
    )
    backbone_model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    backbone_model.eval()

    embed_dim = model_cfg.head.embed_dim
    mil_head = GatedAttentionMIL(feature_dim=embed_dim).to(device)
    mil_head.load(str(mil_path), device=device)
    mil_head.eval()

    log.info("MIL predictor ready (embed_dim=%d).", embed_dim)
    return backbone_model, mil_head


def interface_0_handler() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    frames = load_stacked_tiff(
        INPUT_PATH / "images" / "stacked-barretts-esophagus-endoscopy"
    )
    log.info("Loaded %d frame(s) from TIFF.", len(frames))

    transform = build_val_transform()

    mil_result = load_mil_predictor(device)
    if mil_result is not None:
        return _predict_mil(frames, mil_result, transform, device)

    return _predict_ensemble(frames, transform, device)


def _predict_ensemble(
    frames: list[np.ndarray],
    transform: T.Compose,
    device: torch.device,
) -> int:
    """
    Score each frame independently via ensemble + TTA.
    If patchcore.pkl is present, fuses ensemble probability with PatchCore
    anomaly score using Noisy-OR (HybridScorer).
    """
    predictor = load_ensemble_predictor(device)
    load_calibration_artifacts(predictor)
    detector, hybrid = load_patchcore()

    likelihoods: list[float] = []
    for i, frame in enumerate(frames):
        pil    = Image.fromarray(frame)
        result = predictor.predict_single(pil, transform)
        cal_prob = result["calibrated_prob"]

        if detector is not None and hybrid is not None:
            img_tensor = transform(pil).unsqueeze(0).to(device)
            anom_score = detector.score(predictor.models[0], img_tensor, device)
            cal_prob   = hybrid.combine(cal_prob, anom_score)
            log.info(
                "Frame %d/%d  ens=%.4f  anom=%.4f  hybrid=%.4f  pred=%d",
                i + 1, len(frames), result["calibrated_prob"], anom_score, cal_prob,
                int(cal_prob >= predictor.threshold),
            )
        else:
            log.info(
                "Frame %d/%d  cal=%.4f  unc=%.4f  pred=%d",
                i + 1, len(frames), cal_prob, result["uncertainty"],
                result["prediction"],
            )

        likelihoods.append(round(cal_prob, 6))

    n_pos = sum(p >= predictor.threshold for p in likelihoods)
    log.info(
        "Done — %d/%d frame(s) neoplastic (thr=%.4f, patchcore=%s)",
        n_pos, len(frames), predictor.threshold, detector is not None,
    )
    _write_json(
        OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json",
        likelihoods,
    )
    return 0


def _predict_mil(
    frames: list[np.ndarray],
    mil_result: tuple,
    transform: T.Compose,
    device: torch.device,
) -> int:
    """
    Bag-level MIL prediction.
    FrameQualityFilter selects the most informative frames.
    Bag-level score is broadcast to all input frames (competition format).
    """
    from src.data.frame_filter import FrameQualityConfig, FrameQualityFilter
    from src.models.mil import extract_bag_features

    backbone, mil_head = mil_result

    filt       = FrameQualityFilter(FrameQualityConfig(top_k=16))
    bag_frames = filt.filter(frames)
    log.info("MIL: using %d / %d frame(s) after quality filter.", len(bag_frames), len(frames))

    H = extract_bag_features(backbone.backbone, bag_frames, transform, device)
    with torch.no_grad():
        logit, _ = mil_head(H.to(device))
        raw_prob = float(torch.sigmoid(logit).item())

    cal, threshold = _load_calibrator_raw()
    cal_prob = float(cal.transform(np.array([raw_prob]))[0]) if cal is not None else raw_prob

    log.info(
        "MIL done — raw=%.4f  cal=%.4f  pred=%d  thr=%.4f",
        raw_prob, cal_prob, int(cal_prob >= threshold), threshold,
    )
    likelihoods = [round(cal_prob, 6)] * len(frames)
    _write_json(
        OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json",
        likelihoods,
    )
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
