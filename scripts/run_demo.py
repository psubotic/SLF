#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Pavle Subotic
"""
Usage:
    python run_demo.py --mode synthetic       # Generate and process synthetic trap image abd process
    python run_demo.py --mode image --image path/to/trap.jpg  # provide real image and process
    python run_demo.py --mode batch --input-dir images/ --output-dir results/ # provide folder
"""
import argparse
import json
import logging

import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.augmentation import SyntheticConfig, SyntheticTrapGenerator


from src.detection.descriptor_classifier import (
    DescriptorClassifier,
    DescriptorClassifierConfig,
)
from src.detection.feature_filter import FeatureFilter, FeatureFilterConfig
from src.detection.pipeline import SLFDetectionPipeline
from src.detection.preprocessor import PreprocessorConfig, TrapImagePreprocessor
from src.detection.region_proposer import RegionProposer, RegionProposerConfig
from src.utils.visualization import add_summary_overlay, draw_detections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_demo")


def build_pipeline(config_path: str = "configs/default.yaml") -> SLFDetectionPipeline:
    try:
        return SLFDetectionPipeline.from_config(config_path)
    except FileNotFoundError:
        logger.warning("Config not found at %s; using defaults.", config_path)
        return SLFDetectionPipeline(
            preprocessor=TrapImagePreprocessor(PreprocessorConfig()),
            proposer=RegionProposer(RegionProposerConfig()),
            feature_filter=FeatureFilter(FeatureFilterConfig()),
            classifier=DescriptorClassifier(DescriptorClassifierConfig()),
        )


def run_on_image(
    image_bgr: np.ndarray,
    pipeline: SLFDetectionPipeline,
    output_path: Path,
    image_id: str = "demo",
    show_proposals: bool = False,
) -> dict:
    result = pipeline.run(image_bgr, image_id=image_id)

    vis = draw_detections(result.processed_image, result.detections)

    if show_proposals:
        for det in result.detections:
            x, y, w, h = det.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 200), 1)

    vis = add_summary_overlay(
        vis,
        slf_count=result.slf_count,
        elapsed_sec=result.elapsed_sec,
        mode="HOG+LBP+RF",
    )

    # Save annotated image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])

    logger.info("Saved annotated result → %s", output_path)

    return result.to_dict()


def run_synthetic_demo(output_dir: Path) -> None:
    logger.info("=== Synthetic Demo Mode ===")

    # Generate synthetic trap image
    generator = SyntheticTrapGenerator(
        SyntheticConfig(insects_per_image_range=(3, 8), seed=42)
    )
    image, annotation = generator.generate_one(image_id="demo")

    # Save input image and ground truth
    input_path = output_dir / "input_synthetic.jpg"
    cv2.imwrite(str(input_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    with open(output_dir / "ground_truth.json", "w") as f:
        json.dump(annotation, f, indent=2)

    logger.info(
        "Generated synthetic image with %d SLF adults", len(annotation["annotations"])
    )
    logger.info("Saved input image → %s", input_path)

    # Build pipeline and run detection
    pipeline = build_pipeline()
    result_dict = run_on_image(
        image, pipeline, output_dir / "result_synthetic.jpg", "synthetic_demo"
    )

    # Save JSON results
    with open(output_dir / "result_synthetic.json", "w") as f:
        json.dump(result_dict, f, indent=2)

    # summary
    print(f"\n{'='*60}")
    print("DEMO RESULTS")
    print(f"{'='*60}")
    print(f"Ground truth SLF count:  {len(annotation['annotations'])}")
    print(f"Detected SLF count:      {result_dict['slf_count']}")
    print(f"Processing time:         {result_dict['elapsed_sec']:.2f}s")
    print(f"Total detections:        {len(result_dict['detections'])}")
    print()
    print("Files created:")
    print(f"  {input_path}")
    print(f"  {output_dir / 'result_synthetic.jpg'}")
    print(f"  {output_dir / 'result_synthetic.json'}")
    print(f"  {output_dir / 'ground_truth.json'}")


def run_image_demo(image_path: Path, output_dir: Path) -> None:
    logger.info("=== Image Demo Mode ===")

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Load img
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    logger.info("Loaded image: %s (%dx%d)", image_path, image.shape[1], image.shape[0])

    # Run detection
    pipeline = build_pipeline()
    result_dict = run_on_image(
        image, pipeline, output_dir / f"result_{image_path.stem}.jpg", image_path.stem
    )

    # save JSON results
    with open(output_dir / f"result_{image_path.stem}.json", "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n{'='*60}")
    print("DEMO RESULTS")
    print(f"{'='*60}")
    print(f"Input image:             {image_path}")
    print(f"Detected SLF count:      {result_dict['slf_count']}")
    print(f"Processing time:         {result_dict['elapsed_sec']:.2f}s")
    print(f"Total detections:        {len(result_dict['detections'])}")
    print()
    if result_dict["detections"]:
        print("Detection details:")
        for i, det in enumerate(result_dict["detections"]):
            print(
                f"  [{i+1}] bbox={det['bbox_xywh']}, "
                f"heuristic={det['heuristic_score']:.3f}, "
                f"classifier={det['classifier_score']:.3f}, "
                f"label={'SLF' if det['label'] else 'negative'}"
            )


def run_batch_demo(input_dir: Path, output_dir: Path) -> None:
    logger.info("=== Batch Demo Mode ===")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_paths = [
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_exts
    ]

    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")

    logger.info("Found %d images to process", len(image_paths))

    pipeline = build_pipeline()
    results_summary = []

    for i, image_path in enumerate(image_paths, 1):
        logger.info("[%d/%d] Processing %s", i, len(image_paths), image_path.name)

        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Skipping unreadable image: %s", image_path)
            continue

        try:
            result_dict = run_on_image(
                image,
                pipeline,
                output_dir / f"result_{image_path.stem}.jpg",
                image_path.stem,
            )
            results_summary.append(
                {
                    "filename": image_path.name,
                    "slf_count": result_dict["slf_count"],
                    "elapsed_sec": result_dict["elapsed_sec"],
                    "total_detections": len(result_dict["detections"]),
                }
            )
        except Exception as e:
            logger.error("Failed to process %s: %s", image_path.name, e)

    with open(output_dir / "batch_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    total_slf = sum(r["slf_count"] for r in results_summary)
    total_time = sum(r["elapsed_sec"] for r in results_summary)

    print(f"\n{'='*60}")
    print("BATCH RESULTS")
    print(f"{'='*60}")
    print(f"Images processed:        {len(results_summary)}")
    print(f"Total SLF detected:      {total_slf}")
    print(f"Total processing time:   {total_time:.1f}s")
    print(f"Average time per image:  {total_time/len(results_summary):.2f}s")


def main():
    parser = argparse.ArgumentParser(description="SLF Detection Demo")
    parser.add_argument(
        "--mode",
        choices=["synthetic", "image", "batch"],
        required=True,
        help="Demo mode to run",
    )
    parser.add_argument("--image", type=Path, help="Input image path (for image mode)")
    parser.add_argument(
        "--input-dir", type=Path, help="Input directory (for batch mode)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/demo"),
        help="Output directory (default: outputs/demo)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="configs/default.yaml",
        help="Pipeline config file",
    )
    parser.add_argument(
        "--proposals",
        action="store_true",
        help="Show region proposals in output visualization",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.mode == "synthetic":
            run_synthetic_demo(args.output_dir)
        elif args.mode == "image":
            if not args.image:
                parser.error("--image is required for image mode")
            run_image_demo(args.image, args.output_dir)
        elif args.mode == "batch":
            if not args.input_dir:
                parser.error("--input-dir is required for batch mode")
            run_batch_demo(args.input_dir, args.output_dir)

        print(f"\nDemo completed successfully! Check {args.output_dir}")

    except Exception as e:
        logger.error("Demo failed: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
