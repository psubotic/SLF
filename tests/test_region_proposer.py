import sys
from pathlib import Path

import numpy as np
import pytest

from src.detection.region_proposer import RegionProposer, RegionProposerConfig


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def proposer():
    return RegionProposer(RegionProposerConfig(min_area_px=100, max_area_px=50000))


def make_image_with_blobs(n_blobs=3, size=512, seed=7):
    rng = np.random.default_rng(seed)
    # Yellow background (typical trap colour)
    img = np.full((size, size, 3), [50, 180, 220], dtype=np.uint8)  # BGR ~yellow
    for _ in range(n_blobs):
        cx, cy = rng.integers(80, size - 80, size=2)
        rx = rng.integers(20, 60)
        ry = rng.integers(12, 35)
        color = tuple(rng.integers(20, 80, size=3).tolist())
        import cv2

        cv2.ellipse(img, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, color, -1)
    return img


class TestRegionProposer:
    def test_returns_list(self, proposer):
        img = make_image_with_blobs()
        proposals = proposer(img)
        assert isinstance(proposals, list)

    def test_finds_proposals_with_blobs(self, proposer):
        img = make_image_with_blobs(n_blobs=5)
        proposals = proposer(img)
        assert len(proposals) >= 1, "Should find at least one proposal"

    def test_proposal_bbox_within_image(self, proposer):
        size = 512
        img = make_image_with_blobs(size=size)
        proposals = proposer(img)
        for p in proposals:
            x, y, w, h = p.bbox
            assert x >= 0 and y >= 0
            assert x + w <= size + 5  # small tolerance for rounding
            assert y + h <= size + 5

    def test_source_labels(self, proposer):
        img = make_image_with_blobs()
        proposals = proposer(img)
        for p in proposals:
            assert p.source in {"mser", "contour"}

    def test_area_filter(self):
        # With very tight area limits, should return fewer proposals
        tight = RegionProposer(RegionProposerConfig(min_area_px=5000, max_area_px=8000))
        img = make_image_with_blobs()
        proposals_tight = tight(img)

        wide = RegionProposer(RegionProposerConfig(min_area_px=50, max_area_px=100000))
        proposals_wide = wide(img)

        assert len(proposals_wide) >= len(proposals_tight)

    def test_iou_calculation(self):
        from src.detection.region_proposer import RegionProposer

        iou = RegionProposer._iou((0, 0, 10, 10), (0, 0, 10, 10))
        assert abs(iou - 1.0) < 1e-5

        iou_no_overlap = RegionProposer._iou((0, 0, 5, 5), (10, 10, 20, 20))
        assert iou_no_overlap == 0.0
