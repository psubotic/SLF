import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augmentation.synthetic_generator import (SyntheticConfig,
                                                  SyntheticTrapGenerator)
from src.detection.descriptor_classifier import (DescriptorClassifier,
                                                 DescriptorClassifierConfig)
from src.detection.feature_filter import FeatureFilter, FeatureFilterConfig
from src.detection.pipeline import PipelineResult, SLFDetectionPipeline
from src.detection.preprocessor import (PreprocessorConfig,
                                        TrapImagePreprocessor)
from src.detection.region_proposer import RegionProposer, RegionProposerConfig


@pytest.fixture(scope="module")
def synthetic_image():
    gen = SyntheticTrapGenerator(
        SyntheticConfig(insects_per_image_range=(2, 4), seed=99)
    )
    img, ann = gen.generate_one(image_id=0)
    return img, ann


@pytest.fixture(scope="module")
def pipeline():
    return SLFDetectionPipeline(
        preprocessor=TrapImagePreprocessor(PreprocessorConfig(target_size=(512, 512))),
        proposer=RegionProposer(
            RegionProposerConfig(min_area_px=200, max_area_px=20000)
        ),
        feature_filter=FeatureFilter(FeatureFilterConfig()),
        classifier=DescriptorClassifier(
            DescriptorClassifierConfig(
                model_cache_path=None,  # Don't cache during tests
                auto_train_n_synthetic=100,  # Faster training for tests
            )
        ),
    )


class TestPipelineIntegration:
    def test_pipeline_returns_result(self, pipeline, synthetic_image):
        img, _ = synthetic_image
        result = pipeline.run(img, image_id="test_0")
        assert isinstance(result, PipelineResult)

    def test_result_has_count(self, pipeline, synthetic_image):
        img, _ = synthetic_image
        result = pipeline.run(img)
        assert isinstance(result.slf_count, int)
        assert result.slf_count >= 0

    def test_processed_image_shape(self, pipeline, synthetic_image):
        img, _ = synthetic_image
        result = pipeline.run(img)
        assert result.processed_image.shape == (512, 512, 3)

    def test_elapsed_positive(self, pipeline, synthetic_image):
        img, _ = synthetic_image
        result = pipeline.run(img)
        assert result.elapsed_sec > 0.0

    def test_to_dict_structure(self, pipeline, synthetic_image):
        img, _ = synthetic_image
        result = pipeline.run(img)
        d = result.to_dict()
        assert "slf_count" in d
        assert "detections" in d
        assert "elapsed_sec" in d
        for det in d["detections"]:
            assert "bbox_xywh" in det
            assert "final_score" in det
            assert "label" in det

    def test_empty_image_handled(self, pipeline):
        """Pipeline should handle an image with no plausible insect regions gracefully."""
        # Pure solid colour image — should produce zero detections
        blank = np.full((512, 512, 3), [60, 180, 230], dtype=np.uint8)
        result = pipeline.run(blank, image_id="blank")
        assert isinstance(result, PipelineResult)
        assert result.slf_count >= 0  # May or may not find anything; should not crash

    def test_detection_bboxes_within_image(self, pipeline, synthetic_image):
        img, _ = synthetic_image
        result = pipeline.run(img)
        H, W = result.processed_image.shape[:2]
        for det in result.detections:
            x, y, w, h = det.bbox
            assert x >= 0 and y >= 0
            assert x + w <= W + 5
            assert y + h <= H + 5
