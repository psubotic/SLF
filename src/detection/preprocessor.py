from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessorConfig:
    target_size: Tuple[int, int] = (1024, 1024)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    glare_percentile: float = 97.0
    glare_inpaint_radius: int = 7


class TrapImagePreprocessor:
    def __init__(self, config: PreprocessorConfig | None = None) -> None:
        self.cfg = config or PreprocessorConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip_limit,
            tileGridSize=self.cfg.clahe_tile_grid,
        )

    def __call__(self, image_bgr: np.ndarray) -> "PreprocessResult":
        return self.process(image_bgr)

    def process(self, image_bgr: np.ndarray) -> "PreprocessResult":
        """Full preprocessing chain. Returns a PreprocessResult dataclass."""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Received empty image.")

        img = self._resize(image_bgr)
        glare_mask = self._detect_glare(img)
        img = self._remove_glare(img, glare_mask)
        img = self._enhance_contrast(img)

        logger.debug(
            "Preprocessed image: %s → %s | glare pixels: %d",
            image_bgr.shape,
            img.shape,
            int(glare_mask.sum()),
        )
        return PreprocessResult(image=img, glare_mask=glare_mask)

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = self.cfg.target_size
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    def _detect_glare(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray, self.cfg.glare_percentile)
        mask = (gray >= threshold).astype(np.uint8) * 255
        # Dilate slightly so inpainting covers the halo
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.dilate(mask, kernel, iterations=1)

    def _remove_glare(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.sum() == 0:
            return img
        return cv2.inpaint(img, mask, self.cfg.glare_inpaint_radius, cv2.INPAINT_TELEA)

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    @staticmethod
    def correct_perspective(
        img: np.ndarray,
        src_points: np.ndarray,
        dst_size: Tuple[int, int] = (1024, 1024),
    ) -> np.ndarray:
        w, h = dst_size
        dst_points = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )
        M = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points)
        return cv2.warpPerspective(img, M, (w, h))


@dataclass
class PreprocessResult:
    image: np.ndarray  # Processed BGR image
    glare_mask: np.ndarray  # Binary glare mask (uint8, 255=glare)
