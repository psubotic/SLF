from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class RegionProposerConfig:
    min_area_px: int = 300
    max_area_px: int = 18000
    mser_delta: int = 5
    mser_min_area: int = 200
    mser_max_area: int = 14400
    mser_max_variation: float = 0.25
    contour_canny_low: int = 30
    contour_canny_high: int = 100
    nms_iou_threshold: float = 0.4


@dataclass
class RegionProposal:
    bbox: BBox  # (x, y, w, h)
    source: str  # "mser" | "contour"
    score: float = 1.0  # placeholder; filled by feature_filter


class RegionProposer:
    def __init__(self, config: RegionProposerConfig | None = None) -> None:
        self.cfg = config or RegionProposerConfig()
        self._mser = self._build_mser()

    def __call__(self, image_bgr: np.ndarray) -> List[RegionProposal]:
        return self.propose(image_bgr)

    def propose(self, image_bgr: np.ndarray) -> List[RegionProposal]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        mser_boxes = self._mser_proposals(gray)
        contour_boxes = self._contour_proposals(gray)

        all_proposals = mser_boxes + contour_boxes
        filtered = self._size_filter(all_proposals)
        merged = self._nms(filtered)

        logger.debug(
            "RegionProposer: mser=%d contour=%d after_size=%d after_nms=%d",
            len(mser_boxes),
            len(contour_boxes),
            len(filtered),
            len(merged),
        )
        return merged

    def _build_mser(self) -> cv2.MSER:
        return cv2.MSER_create(
            self.cfg.mser_delta,
            self.cfg.mser_min_area,
            self.cfg.mser_max_area,
            self.cfg.mser_max_variation,
        )

    def _mser_proposals(self, gray: np.ndarray) -> List[RegionProposal]:
        regions, bboxes = self._mser.detectRegions(gray)
        proposals = []
        for bbox in bboxes:
            x, y, w, h = bbox
            proposals.append(RegionProposal(bbox=(x, y, w, h), source="mser"))
        return proposals

    def _contour_proposals(self, gray: np.ndarray) -> List[RegionProposal]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(
            blurred, self.cfg.contour_canny_low, self.cfg.contour_canny_high
        )

        # Close small gaps so wing outlines form closed contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        proposals = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            proposals.append(RegionProposal(bbox=(x, y, w, h), source="contour"))
        return proposals

    def _size_filter(self, proposals: List[RegionProposal]) -> List[RegionProposal]:
        out = []
        for p in proposals:
            _, _, w, h = p.bbox
            area = w * h
            if self.cfg.min_area_px <= area <= self.cfg.max_area_px:
                out.append(p)
        return out

    def _nms(self, proposals: List[RegionProposal]) -> List[RegionProposal]:
        if not proposals:
            return []

        boxes = np.array([p.bbox for p in proposals], dtype=np.float32)
        areas = boxes[:, 2] * boxes[:, 3]
        order = np.argsort(-areas)  # descending by area

        keep: List[int] = []
        suppressed = np.zeros(len(proposals), dtype=bool)

        for i in order:
            if suppressed[i]:
                continue
            keep.append(i)
            x1, y1, w1, h1 = boxes[i]
            for j in order:
                if suppressed[j] or j == i:
                    continue
                x2, y2, w2, h2 = boxes[j]
                iou = self._iou(
                    (x1, y1, x1 + w1, y1 + h1),
                    (x2, y2, x2 + w2, y2 + h2),
                )
                if iou > self.cfg.nms_iou_threshold:
                    suppressed[j] = True

        return [proposals[i] for i in keep]

    @staticmethod
    def _iou(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0
