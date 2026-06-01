"""
FrameQualityFilter — deterministic frame selection for endoscopic stacks.

Rejects frames with two physical defects that cause false positives:
  - Motion blur: Laplacian variance below threshold → camera movement artifact.
  - Overexposure: specular highlights cover too much of the frame → light reflection.

Usage:
    filt = FrameQualityFilter(top_k=12)
    clean_frames = filt.filter(raw_frames)  # list[np.ndarray HxWx3]

Setting top_k=None keeps all frames that pass the quality thresholds.
Setting top_k=K always returns the K best frames (sorted by score), useful when
  the number of frames varies and the downstream model expects a fixed budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameQualityConfig:
    top_k: int | None = None
    min_sharpness_score: float = 0.0   # [0, 1] — 0 disables the sharpness gate
    max_overexposed_ratio: float = 1.0  # [0, 1] — 1.0 disables the overexposure gate


class FrameQualityFilter:
    """
    Scores endoscopic frames by sharpness and overexposure, then filters.

    Score formula (∈ [0, 1]):
        sharpness_score  = var_laplacian / (var_laplacian + 50)   # asymptotic
        overexposed_frac = fraction of pixels with gray > 240
        score            = sharpness_score × (1 − overexposed_frac)

    The constant 50 was chosen so typical in-focus endoscopic frames (var ≈ 200–800)
    score above 0.75, while heavily blurred frames (var < 20) score below 0.3.
    """

    def __init__(self, cfg: FrameQualityConfig | None = None):
        self.cfg = cfg or FrameQualityConfig()

    @staticmethod
    def score(frame: np.ndarray) -> float:
        """Compute quality score ∈ [0, 1] for a single HxWx3 uint8 frame."""
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-python-headless is required for FrameQualityFilter. "
                "Install with: pip install opencv-python-headless"
            ) from exc

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        sharpness = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var()
        overexposed = float((gray > 240).mean())
        return float(sharpness / (sharpness + 50.0)) * (1.0 - overexposed)

    def filter(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Filter and rank frames by quality.

        Fallback: if no frame passes the thresholds (e.g. all overexposed),
        returns all frames sorted by score so inference is never blocked.
        """
        if not frames:
            return frames

        scored = [(self.score(f), f) for f in frames]

        cfg = self.cfg
        passed = [
            (s, f) for s, f in scored
            if s >= cfg.min_sharpness_score
            and (1.0 - s) <= cfg.max_overexposed_ratio
        ]

        if not passed:
            logger.warning(
                "FrameQualityFilter: no frame passed quality gates "
                "(min_sharpness=%.2f, max_overexposed=%.2f). "
                "Falling back to all %d frames.",
                cfg.min_sharpness_score,
                cfg.max_overexposed_ratio,
                len(frames),
            )
            passed = scored

        passed.sort(key=lambda x: x[0], reverse=True)

        if cfg.top_k is not None:
            passed = passed[: cfg.top_k]

        logger.debug(
            "FrameQualityFilter: kept %d / %d frames (top_k=%s)",
            len(passed), len(frames), cfg.top_k,
        )
        return [f for _, f in passed]
