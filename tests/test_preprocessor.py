import sys
from pathlib import Path

import numpy as np
import pytest

from src.detection.preprocessor import PreprocessorConfig, TrapImagePreprocessor


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def preprocessor():
    return TrapImagePreprocessor(PreprocessorConfig(target_size=(512, 512)))


def make_random_image(h=480, w=640, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


class TestPreprocessor:
    def test_output_shape(self, preprocessor):
        img = make_random_image()
        result = preprocessor(img)
        assert result.image.shape == (512, 512, 3)

    def test_output_dtype(self, preprocessor):
        img = make_random_image()
        result = preprocessor(img)
        assert result.image.dtype == np.uint8

    def test_glare_mask_shape(self, preprocessor):
        img = make_random_image()
        result = preprocessor(img)
        assert result.glare_mask.shape == (512, 512)

    def test_glare_detected_on_bright_image(self, preprocessor):
        """An image with a bright white patch should have glare detected."""
        img = make_random_image(seed=1)
        img[100:200, 100:200] = 255  # Bright patch
        result = preprocessor(img)
        assert result.glare_mask.sum() > 0

    def test_empty_image_raises(self, preprocessor):
        with pytest.raises(ValueError):
            preprocessor(np.array([]))

    def test_non_square_input(self, preprocessor):
        img = make_random_image(h=300, w=800)
        result = preprocessor(img)
        assert result.image.shape == (512, 512, 3)

    def test_perspective_correction_shape(self):
        img = make_random_image(h=600, w=800)
        src = np.array([[50, 50], [750, 50], [750, 550], [50, 550]], dtype=np.float32)
        out = TrapImagePreprocessor.correct_perspective(img, src, dst_size=(512, 512))
        assert out.shape == (512, 512, 3)
