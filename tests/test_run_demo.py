# MIT License
# Copyright (c) 2026 Pavle Subotic
"""
Tests for scripts/run_demo.py

The pipeline is mocked throughout so tests run fast without ML training.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_demo import (
    build_pipeline,
    main,
    run_batch_demo,
    run_image_demo,
    run_on_image,
    run_synthetic_demo,
)
from src.detection.pipeline import Detection, PipelineResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(h: int = 512, w: int = 512) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_detection(x=10, y=10, w=30, h=30, label=1) -> Detection:
    return Detection(
        bbox=(x, y, w, h),
        heuristic_score=0.85,
        classifier_score=0.80,
        final_score=0.80,
        label=label,
    )


def _make_result(
    detections=None,
    slf_count=1,
    h=512,
    w=512,
    elapsed_sec=0.1,
) -> PipelineResult:
    if detections is None:
        detections = [_make_detection()]
    return PipelineResult(
        detections=detections,
        slf_count=slf_count,
        processed_image=_make_image(h, w),
        elapsed_sec=elapsed_sec,
    )


def _mock_pipeline(result: PipelineResult = None) -> MagicMock:
    mock = MagicMock()
    mock.run.return_value = result or _make_result()
    return mock


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def test_returns_pipeline_from_valid_config(self, tmp_path):
        """build_pipeline delegates to SLFDetectionPipeline.from_config when file exists."""
        fake_config = tmp_path / "cfg.yaml"
        fake_config.write_text("")  # content doesn't matter; we mock from_config

        with patch(
            "scripts.run_demo.SLFDetectionPipeline.from_config"
        ) as mock_from_config:
            mock_from_config.return_value = MagicMock()
            pipeline = build_pipeline(str(fake_config))
            mock_from_config.assert_called_once_with(str(fake_config))
            assert pipeline is mock_from_config.return_value

    def test_falls_back_to_defaults_when_config_missing(self, tmp_path):
        """build_pipeline constructs a default pipeline when config file is absent."""
        missing = str(tmp_path / "nonexistent.yaml")
        with patch("scripts.run_demo.SLFDetectionPipeline") as MockPipeline:
            MockPipeline.from_config.side_effect = FileNotFoundError
            MockPipeline.return_value = MagicMock()
            pipeline = build_pipeline(missing)
            assert pipeline is MockPipeline.return_value


# ---------------------------------------------------------------------------
# run_on_image
# ---------------------------------------------------------------------------

class TestRunOnImage:
    def test_returns_dict_with_expected_keys(self, tmp_path):
        pipeline = _mock_pipeline()
        result = run_on_image(_make_image(), pipeline, tmp_path / "out.jpg")
        assert "slf_count" in result
        assert "elapsed_sec" in result
        assert "detections" in result

    def test_saves_output_file(self, tmp_path):
        pipeline = _mock_pipeline()
        out = tmp_path / "sub" / "out.jpg"
        run_on_image(_make_image(), pipeline, out)
        assert out.exists()

    def test_creates_parent_directories(self, tmp_path):
        pipeline = _mock_pipeline()
        deep_out = tmp_path / "a" / "b" / "c" / "out.jpg"
        run_on_image(_make_image(), pipeline, deep_out)
        assert deep_out.parent.exists()

    def test_show_proposals_draws_bboxes(self, tmp_path):
        """When show_proposals=True, cv2.rectangle is called once per detection by run_on_image.
        Both draw_detections and add_summary_overlay are mocked to isolate those calls."""
        pipeline = _mock_pipeline()
        out = tmp_path / "out.jpg"
        with patch("scripts.run_demo.draw_detections", return_value=_make_image()), \
             patch("scripts.run_demo.add_summary_overlay", return_value=_make_image()), \
             patch("scripts.run_demo.cv2.rectangle") as mock_rect:
            run_on_image(_make_image(), pipeline, out, show_proposals=True)
            assert mock_rect.call_count == len(pipeline.run.return_value.detections)

    def test_show_proposals_false_skips_rectangles(self, tmp_path):
        pipeline = _mock_pipeline()
        out = tmp_path / "out.jpg"
        with patch("scripts.run_demo.draw_detections", return_value=_make_image()), \
             patch("scripts.run_demo.add_summary_overlay", return_value=_make_image()), \
             patch("scripts.run_demo.cv2.rectangle") as mock_rect:
            run_on_image(_make_image(), pipeline, out, show_proposals=False)
            mock_rect.assert_not_called()

    def test_result_slf_count_matches_pipeline(self, tmp_path):
        result = _make_result(slf_count=3)
        pipeline = _mock_pipeline(result)
        d = run_on_image(_make_image(), pipeline, tmp_path / "out.jpg")
        assert d["slf_count"] == 3

    def test_image_id_forwarded_to_pipeline(self, tmp_path):
        pipeline = _mock_pipeline()
        run_on_image(_make_image(), pipeline, tmp_path / "out.jpg", image_id="my_id")
        pipeline.run.assert_called_once()
        _, kwargs = pipeline.run.call_args
        assert kwargs.get("image_id") == "my_id"


# ---------------------------------------------------------------------------
# run_synthetic_demo
# ---------------------------------------------------------------------------

class TestRunSyntheticDemo:
    def _run(self, tmp_path, **kwargs):
        with patch("scripts.run_demo.build_pipeline", return_value=_mock_pipeline()):
            run_synthetic_demo(tmp_path, **kwargs)

    def test_creates_input_image(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "input_synthetic.jpg").exists()

    def test_creates_result_image(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "result_synthetic.jpg").exists()

    def test_creates_result_json(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "result_synthetic.json").exists()

    def test_creates_ground_truth_json(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "ground_truth.json").exists()

    def test_result_json_has_expected_keys(self, tmp_path):
        self._run(tmp_path)
        data = json.loads((tmp_path / "result_synthetic.json").read_text())
        assert "slf_count" in data
        assert "detections" in data
        assert "elapsed_sec" in data

    def test_ground_truth_json_has_annotations(self, tmp_path):
        self._run(tmp_path)
        data = json.loads((tmp_path / "ground_truth.json").read_text())
        assert "annotations" in data

    def test_show_proposals_passed_through(self, tmp_path):
        with patch("scripts.run_demo.build_pipeline", return_value=_mock_pipeline()):
            with patch("scripts.run_demo.run_on_image") as mock_run:
                mock_run.return_value = _make_result().to_dict()
                run_synthetic_demo(tmp_path, show_proposals=True)
                _, kwargs = mock_run.call_args
                assert kwargs.get("show_proposals") is True


# ---------------------------------------------------------------------------
# run_image_demo
# ---------------------------------------------------------------------------

class TestRunImageDemo:
    def _make_image_file(self, directory: Path) -> Path:
        img_path = directory / "trap.jpg"
        cv2.imwrite(str(img_path), _make_image())
        return img_path

    def test_raises_file_not_found_for_missing_image(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_image_demo(tmp_path / "no_such.jpg", tmp_path)

    def test_raises_value_error_for_unreadable_image(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not an image")
        with pytest.raises(ValueError):
            run_image_demo(bad, tmp_path)

    def test_creates_result_image(self, tmp_path):
        img_path = self._make_image_file(tmp_path)
        with patch("scripts.run_demo.build_pipeline", return_value=_mock_pipeline()):
            run_image_demo(img_path, tmp_path)
        assert (tmp_path / f"result_{img_path.stem}.jpg").exists()

    def test_creates_result_json(self, tmp_path):
        img_path = self._make_image_file(tmp_path)
        with patch("scripts.run_demo.build_pipeline", return_value=_mock_pipeline()):
            run_image_demo(img_path, tmp_path)
        assert (tmp_path / f"result_{img_path.stem}.json").exists()

    def test_result_json_valid(self, tmp_path):
        img_path = self._make_image_file(tmp_path)
        with patch("scripts.run_demo.build_pipeline", return_value=_mock_pipeline()):
            run_image_demo(img_path, tmp_path)
        data = json.loads((tmp_path / f"result_{img_path.stem}.json").read_text())
        assert "slf_count" in data

    def test_show_proposals_passed_through(self, tmp_path):
        img_path = self._make_image_file(tmp_path)
        with patch("scripts.run_demo.build_pipeline", return_value=_mock_pipeline()):
            with patch("scripts.run_demo.run_on_image") as mock_run:
                mock_run.return_value = _make_result().to_dict()
                run_image_demo(img_path, tmp_path, show_proposals=True)
                _, kwargs = mock_run.call_args
                assert kwargs.get("show_proposals") is True


# ---------------------------------------------------------------------------
# run_batch_demo
# ---------------------------------------------------------------------------

class TestRunBatchDemo:
    def _populate_dir(self, directory: Path, count: int = 2) -> list[Path]:
        paths = []
        for i in range(count):
            p = directory / f"img_{i}.jpg"
            cv2.imwrite(str(p), _make_image())
            paths.append(p)
        return paths

    def test_raises_for_empty_directory(self, tmp_path):
        with pytest.raises(ValueError, match="No images found"):
            run_batch_demo(tmp_path, tmp_path / "out")

    def test_creates_batch_summary_json(self, tmp_path):
        self._populate_dir(tmp_path)
        out_dir = tmp_path / "out"
        with patch("scripts.run_demo.build_pipeline", return_value=_mock_pipeline()):
            run_batch_demo(tmp_path, out_dir)
        assert (out_dir / "batch_summary.json").exists()

    def test_batch_summary_contains_all_processed(self, tmp_path):
        self._populate_dir(tmp_path, count=3)
        out_dir = tmp_path / "out"
        with patch("scripts.run_demo.build_pipeline", return_value=_mock_pipeline()):
            run_batch_demo(tmp_path, out_dir)
        summary = json.loads((out_dir / "batch_summary.json").read_text())
        assert len(summary) == 3

    def test_batch_summary_entry_structure(self, tmp_path):
        self._populate_dir(tmp_path, count=1)
        out_dir = tmp_path / "out"
        with patch("scripts.run_demo.build_pipeline", return_value=_mock_pipeline()):
            run_batch_demo(tmp_path, out_dir)
        entry = json.loads((out_dir / "batch_summary.json").read_text())[0]
        assert "filename" in entry
        assert "slf_count" in entry
        assert "elapsed_sec" in entry
        assert "total_detections" in entry

    def test_skips_unreadable_images(self, tmp_path):
        self._populate_dir(tmp_path, count=2)
        bad = tmp_path / "corrupt.jpg"
        bad.write_bytes(b"not an image")
        out_dir = tmp_path / "out"
        with patch("scripts.run_demo.build_pipeline", return_value=_mock_pipeline()):
            run_batch_demo(tmp_path, out_dir)
        summary = json.loads((out_dir / "batch_summary.json").read_text())
        filenames = [e["filename"] for e in summary]
        assert "corrupt.jpg" not in filenames

    def test_no_division_error_when_all_images_fail(self, tmp_path):
        """If every image fails to process, batch_summary.json should be empty without crashing."""
        for i in range(2):
            (tmp_path / f"bad_{i}.jpg").write_bytes(b"corrupt")
        out_dir = tmp_path / "out"
        with patch("scripts.run_demo.build_pipeline", return_value=_mock_pipeline()):
            run_batch_demo(tmp_path, out_dir)  # should not raise ZeroDivisionError
        summary = json.loads((out_dir / "batch_summary.json").read_text())
        assert summary == []

    def test_aggregates_slf_counts(self, tmp_path):
        self._populate_dir(tmp_path, count=3)
        result_with_2 = _make_result(
            detections=[_make_detection(), _make_detection()], slf_count=2
        )
        pipeline = _mock_pipeline(result_with_2)
        out_dir = tmp_path / "out"
        with patch("scripts.run_demo.build_pipeline", return_value=pipeline):
            run_batch_demo(tmp_path, out_dir)
        summary = json.loads((out_dir / "batch_summary.json").read_text())
        assert sum(e["slf_count"] for e in summary) == 6

    def test_pipeline_built_once(self, tmp_path):
        self._populate_dir(tmp_path, count=3)
        out_dir = tmp_path / "out"
        with patch("scripts.run_demo.build_pipeline", return_value=_mock_pipeline()) as mock_build:
            run_batch_demo(tmp_path, out_dir)
            mock_build.assert_called_once()


# ---------------------------------------------------------------------------
# main (CLI entry point)
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_0_on_synthetic_success(self, tmp_path):
        with patch("scripts.run_demo.run_synthetic_demo") as mock_fn:
            with patch(
                "sys.argv",
                ["run_demo.py", "--mode", "synthetic", "--output-dir", str(tmp_path)],
            ):
                tmp_path.mkdir(parents=True, exist_ok=True)
                assert main() == 0
                mock_fn.assert_called_once()

    def test_returns_1_on_exception(self, tmp_path):
        with patch("scripts.run_demo.run_synthetic_demo", side_effect=RuntimeError("boom")):
            with patch(
                "sys.argv",
                ["run_demo.py", "--mode", "synthetic", "--output-dir", str(tmp_path)],
            ):
                tmp_path.mkdir(parents=True, exist_ok=True)
                assert main() == 1

    def test_image_mode_errors_without_image_arg(self, tmp_path):
        with patch(
            "sys.argv",
            ["run_demo.py", "--mode", "image", "--output-dir", str(tmp_path)],
        ):
            with pytest.raises(SystemExit):
                main()

    def test_batch_mode_errors_without_input_dir(self, tmp_path):
        with patch(
            "sys.argv",
            ["run_demo.py", "--mode", "batch", "--output-dir", str(tmp_path)],
        ):
            with pytest.raises(SystemExit):
                main()

    def test_proposals_flag_passed_to_synthetic(self, tmp_path):
        with patch("scripts.run_demo.run_synthetic_demo") as mock_fn:
            with patch(
                "sys.argv",
                [
                    "run_demo.py",
                    "--mode", "synthetic",
                    "--output-dir", str(tmp_path),
                    "--proposals",
                ],
            ):
                tmp_path.mkdir(parents=True, exist_ok=True)
                main()
                _, kwargs = mock_fn.call_args
                assert kwargs.get("show_proposals") is True

    def test_proposals_flag_passed_to_image_mode(self, tmp_path):
        fake_image = tmp_path / "img.jpg"
        cv2.imwrite(str(fake_image), _make_image())
        with patch("scripts.run_demo.run_image_demo") as mock_fn:
            with patch(
                "sys.argv",
                [
                    "run_demo.py",
                    "--mode", "image",
                    "--image", str(fake_image),
                    "--output-dir", str(tmp_path),
                    "--proposals",
                ],
            ):
                main()
                _, kwargs = mock_fn.call_args
                assert kwargs.get("show_proposals") is True
