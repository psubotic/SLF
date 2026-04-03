from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from .descriptor_classifier import DescriptorClassifier, DescriptorClassifierConfig
from .feature_filter import FeatureFilter, FeatureFilterConfig
from .preprocessor import PreprocessorConfig, TrapImagePreprocessor
from .region_proposer import RegionProposal, RegionProposer, RegionProposerConfig

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in processed image space
    heuristic_score: float  # Score from FeatureFilter [0,1]
    classifier_score: float  # Score from DescriptorClassifier [0,1]
    final_score: float  # Combined or direct score
    label: int  # 1 = SLF, 0 = not SLF (post-threshold)

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        x, y, w, h = self.bbox
        return x, y, x + w, y + h


@dataclass
class PipelineResult:
    detections: List[Detection]
    slf_count: int
    processed_image: np.ndarray
    elapsed_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "slf_count": int(self.slf_count),
            "elapsed_sec": round(self.elapsed_sec, 3),
            "detections": [
                {
                    "bbox_xywh": [int(v) for v in d.bbox],
                    "heuristic_score": round(float(d.heuristic_score), 4),
                    "classifier_score": round(float(d.classifier_score), 4),
                    "final_score": round(float(d.final_score), 4),
                    "label": int(d.label),
                }
                for d in self.detections
            ],
            "metadata": self.metadata,
        }


class SLFDetectionPipeline:
    def __init__(
        self,
        preprocessor: TrapImagePreprocessor,
        proposer: RegionProposer,
        feature_filter: FeatureFilter,
        classifier: DescriptorClassifier,
        max_proposals: int = 200,
    ):
        self.preprocessor = preprocessor
        self.proposer = proposer
        self.feature_filter = feature_filter
        self.classifier = classifier
        self.max_proposals = max_proposals

    @classmethod
    def from_config(cls, config_path: str | Path) -> "SLFDetectionPipeline":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        pre_cfg = PreprocessorConfig(
            target_size=tuple(cfg["preprocessing"]["target_size"]),
            clahe_clip_limit=cfg["preprocessing"]["clahe_clip_limit"],
            clahe_tile_grid=tuple(cfg["preprocessing"]["clahe_tile_grid"]),
            glare_percentile=cfg["preprocessing"]["glare_percentile"],
            glare_inpaint_radius=cfg["preprocessing"]["glare_inpaint_radius"],
        )

        prop_cfg = RegionProposerConfig(
            min_area_px=cfg["region_proposal"]["min_area_px"],
            max_area_px=cfg["region_proposal"]["max_area_px"],
            mser_delta=cfg["region_proposal"]["mser_delta"],
            nms_iou_threshold=cfg["region_proposal"]["nms_iou_threshold"],
        )

        feat_cfg = FeatureFilterConfig(
            min_aspect_ratio=cfg["feature_filter"]["min_aspect_ratio"],
            max_aspect_ratio=cfg["feature_filter"]["max_aspect_ratio"],
            min_color_fraction=cfg["feature_filter"]["min_color_fraction"],
            min_score=cfg["feature_filter"]["min_score"],
        )

        clf_cfg = cfg["classifier"]
        descriptor_cfg = DescriptorClassifierConfig(
            mode=clf_cfg["mode"],
            threshold=clf_cfg["threshold"],
            patch_size=tuple(clf_cfg["patch_size"]),
            hog_orientations=clf_cfg["hog_orientations"],
            hog_pixels_per_cell=tuple(clf_cfg["hog_pixels_per_cell"]),
            hog_cells_per_block=tuple(clf_cfg["hog_cells_per_block"]),
            lbp_radius=clf_cfg["lbp_radius"],
            lbp_n_points=clf_cfg["lbp_n_points"],
            lbp_n_bins=clf_cfg["lbp_n_bins"],
            hsv_h_bins=clf_cfg["hsv_h_bins"],
            hsv_s_bins=clf_cfg["hsv_s_bins"],
            hsv_v_bins=clf_cfg["hsv_v_bins"],
            ocsvm_nu=clf_cfg["ocsvm_nu"],
            ocsvm_kernel=clf_cfg["ocsvm_kernel"],
            ocsvm_gamma=clf_cfg["ocsvm_gamma"],
            rf_n_estimators=clf_cfg["rf_n_estimators"],
            rf_max_depth=clf_cfg["rf_max_depth"],
            rf_class_weight=clf_cfg["rf_class_weight"],
            auto_train_n_synthetic=clf_cfg["auto_train_n_synthetic"],
            model_cache_path=clf_cfg["model_cache_path"],
        )

        return cls(
            preprocessor=TrapImagePreprocessor(pre_cfg),
            proposer=RegionProposer(prop_cfg),
            feature_filter=FeatureFilter(feat_cfg),
            classifier=DescriptorClassifier(descriptor_cfg),
        )

    def run(self, image_bgr: np.ndarray, image_id: str = "") -> PipelineResult:
        t0 = time.perf_counter()

        # Stage 1: Preprocess
        prep_result = self.preprocessor(image_bgr)
        processed = prep_result.image

        # Stage 2: Region proposals
        proposals: List[RegionProposal] = self.proposer(processed)
        if not proposals:
            logger.info("[%s] No proposals found.", image_id)
            return PipelineResult(
                detections=[],
                slf_count=0,
                processed_image=processed,
                elapsed_sec=time.perf_counter() - t0,
                metadata={"image_id": image_id, "n_proposals": 0},
            )

        # Stage 3a: Heuristic feature filter
        proposals = self.feature_filter(processed, proposals)
        # Limit to top N before expensive descriptor extraction
        proposals = proposals[: self.max_proposals]

        if not proposals:
            return PipelineResult(
                detections=[],
                slf_count=0,
                processed_image=processed,
                elapsed_sec=time.perf_counter() - t0,
                metadata={"image_id": image_id, "n_proposals_post_filter": 0},
            )

        # Stage 3b: Descriptor classification
        patches = [self._crop_patch(processed, p.bbox) for p in proposals]
        scores = self.classifier.score_batch(patches)

        # Apply threshold and create detections
        detections = []
        for proposal, score in zip(proposals, scores):
            label = int(score >= self.classifier.cfg.threshold)
            detections.append(
                Detection(
                    bbox=proposal.bbox,
                    heuristic_score=proposal.score,
                    classifier_score=score,
                    final_score=score,  # Use classifier score directly
                    label=label,
                )
            )

        positive = [d for d in detections if d.label == 1]

        elapsed = time.perf_counter() - t0
        logger.info(
            "[%s] Proposals: %d | Post-filter: %d | Detections (SLF): %d | %.2fs",
            image_id,
            len(proposals),
            len(proposals),
            len(positive),
            elapsed,
        )

        return PipelineResult(
            detections=detections,
            slf_count=len(positive),
            processed_image=processed,
            elapsed_sec=elapsed,
            metadata={
                "image_id": image_id,
                "n_proposals_raw": len(proposals),
                "n_clf_positive": len(positive),
            },
        )

    @staticmethod
    def _crop_patch(image: np.ndarray, bbox: Tuple) -> np.ndarray:
        x, y, w, h = bbox
        H, W = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return patch
