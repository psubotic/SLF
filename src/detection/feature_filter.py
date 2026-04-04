# MIT License
# Copyright (c) 2026 Pavle Subotic

"""
feature_filter.py
─────────────────
Stage 3a: Heuristic feature-based filtering of region proposals.

Uses species-specific visual features of *Lycorma delicatula* adults:

  Forewings:  mottled grey-brown with black spots
  Hindwings:  bright red with black spots and white band
  Body shape: broadly oval, wider than tall, folded wings
  Aspect ratio (wing spread): ~1.5–3.5 × wider than tall

This stage runs entirely with NumPy/OpenCV — no ML model required —
and provides a fast pre-filter before the descriptor classifier.

Each proposal receives a float score in [0, 1] combining:
  - HSV colour overlap with SLF colour ranges
  - Aspect ratio compatibility
  - Contour solidity (not too fragmented)

Proposals below `min_score` are discarded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .region_proposer import RegionProposal

logger = logging.getLogger(__name__)


@dataclass
class FeatureFilterConfig:
    # HSV ranges for SLF forewings (grey-brown) and hindwings (red)
    # OpenCV uses H: 0-180, S: 0-255, V: 0-255
    hsv_ranges: List[Tuple[List[int], List[int]]] = None

    min_aspect_ratio: float = 1.1
    max_aspect_ratio: float = 5.0
    min_solidity: float = 0.25
    max_solidity: float = 0.98
    min_color_fraction: float = 0.06
    min_score: float = 0.15  # Proposals below this are discarded

    def __post_init__(self):
        if self.hsv_ranges is None:
            # Red-brown range (forewing)
            red_brown_lo = [0, 40, 35]
            red_brown_hi = [22, 255, 220]
            # Red range (hindwing), wraps around hue wheel
            red_lo = [155, 50, 40]
            red_hi = [180, 255, 255]
            # Grey range (forewing base)
            grey_lo = [0, 0, 60]
            grey_hi = [180, 55, 200]
            self.hsv_ranges = [
                (red_brown_lo, red_brown_hi),
                (red_lo, red_hi),
                (grey_lo, grey_hi),
            ]


class FeatureFilter:
    """
    Score and filter region proposals using hand-crafted visual features.

    Usage
    -----
    >>> ffilter = FeatureFilter()
    >>> scored = ffilter(image_bgr, proposals)
    >>> top = [p for p in scored if p.score > 0.3]
    """

    def __init__(self, config: FeatureFilterConfig | None = None) -> None:
        self.cfg = config or FeatureFilterConfig()

    def __call__(
        self,
        image_bgr: np.ndarray,
        proposals: List[RegionProposal],
    ) -> List[RegionProposal]:
        return self.score_and_filter(image_bgr, proposals)

    def score_and_filter(
        self,
        image_bgr: np.ndarray,
        proposals: List[RegionProposal],
    ) -> List[RegionProposal]:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        scored = []
        for proposal in proposals:
            score = self._score(image_bgr, hsv, proposal)
            proposal.score = score
            if score >= self.cfg.min_score:
                scored.append(proposal)

        # Sort by descending score
        scored.sort(key=lambda p: p.score, reverse=True)
        logger.debug(
            "FeatureFilter: %d → %d proposals (min_score=%.2f)",
            len(proposals),
            len(scored),
            self.cfg.min_score,
        )
        return scored

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        bgr: np.ndarray,
        hsv: np.ndarray,
        proposal: RegionProposal,
    ) -> float:
        x, y, w, h = proposal.bbox
        # Guard against out-of-bounds
        H, W = bgr.shape[:2]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x or y2 <= y:
            return 0.0

        crop_hsv = hsv[y:y2, x:x2]
        crop_bgr = bgr[y:y2, x:x2]

        color_score = self._color_score(crop_hsv)
        shape_score = self._shape_score(crop_bgr, w, h)

        # Weighted combination
        return 0.55 * color_score + 0.45 * shape_score

    def _color_score(self, crop_hsv: np.ndarray) -> float:
        """Fraction of pixels matching any SLF HSV range, clipped to [0,1]."""
        total = crop_hsv.shape[0] * crop_hsv.shape[1]
        if total == 0:
            return 0.0

        combined_mask = np.zeros((crop_hsv.shape[0], crop_hsv.shape[1]), dtype=np.uint8)
        for lo, hi in self.cfg.hsv_ranges:
            lo_arr = np.array(lo, dtype=np.uint8)
            hi_arr = np.array(hi, dtype=np.uint8)
            mask = cv2.inRange(crop_hsv, lo_arr, hi_arr)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        fraction = combined_mask.sum() / 255.0 / total
        # Normalise: we expect 10-70% match for a real specimen
        score = np.clip((fraction - self.cfg.min_color_fraction) / 0.45, 0.0, 1.0)
        return float(score)

    def _shape_score(self, crop_bgr: np.ndarray, w: int, h: int) -> float:
        """Score based on aspect ratio and contour solidity."""
        aspect = w / max(h, 1)
        aspect_score = self._aspect_score(aspect)

        solidity_score = self._solidity_score(crop_bgr)
        return 0.5 * aspect_score + 0.5 * solidity_score

    def _aspect_score(self, aspect: float) -> float:
        lo, hi = self.cfg.min_aspect_ratio, self.cfg.max_aspect_ratio
        if aspect < lo or aspect > hi:
            return 0.0
        # Peak around 2.0 (typical SLF resting aspect ratio)
        optimal = 2.0
        spread = (hi - lo) / 2.0
        return float(np.exp(-((aspect - optimal) ** 2) / (2 * spread**2)))

    def _solidity_score(self, crop_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return 0.0
        solidity = area / hull_area
        if solidity < self.cfg.min_solidity or solidity > self.cfg.max_solidity:
            return 0.0
        # Prefer moderate solidity (0.5–0.85) — fragmented = debris, perfect = simple shape
        return float(np.clip((solidity - 0.3) / 0.5, 0.0, 1.0))
