#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Pavle Subotic
"""
This script:
1. Generates synthetic test images with fixed seeds (separate from training)
2. Runs the heuristic-only baseline
3. Trains the HOG+LBP+RF classifier from synthetic training data
4. Evaluates both approaches on the same test set
5. Outputs the precision/recall/F1 metrics in the published table

Usage:
    python benchmark_table_metrics.py
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.augmentation.synthetic_generator import SyntheticConfig, SyntheticTrapGenerator
from src.detection.descriptor_classifier import (
    DescriptorClassifier,
    DescriptorClassifierConfig,
)
from src.detection.feature_filter import FeatureFilter, FeatureFilterConfig
from src.detection.preprocessor import PreprocessorConfig, TrapImagePreprocessor
from src.detection.region_proposer import RegionProposer, RegionProposerConfig

logging.getLogger().setLevel(logging.WARNING)

# training data generation (seed 42 - training set)
TRAINING_SEED = 42
TRAINING_IMAGES = 80
TRAINING_INSECTS_PER_IMAGE = (3, 6)

# test data generation (seed 1337 - held-out test set)
TEST_SEED = 1337
TEST_IMAGES = 20
TEST_INSECTS_PER_IMAGE = (2, 6)

# pipeline config
IMAGE_SIZE = (1024, 1024)
MIN_AREA_PX = 300
MAX_AREA_PX = 18000
NMS_IOU = 0.35
HEURISTIC_MIN_SCORE = 0.18
HEURISTIC_ASPECT_RATIO_MIN = 1.2
HEURISTIC_COLOR_FRACTION_MIN = 0.08

# classifier config
RF_THRESHOLD = 0.65
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 12
SYNTHETIC_TRAINING_PATCHES = 400

# IoU threshold for matching predictions to ground truth
IOU_THRESHOLD = 0.3


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────


def iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to (x1, y1, x2, y2)
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    # Intersection rectangle
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1_max, x2_max)
    iy2 = min(y1_max, y2_max)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = w1 * h1 + w2 * h2 - intersection

    return intersection / union if union > 0 else 0.0


def evaluate_detections(
    detections: List, ground_truth: List[Dict], threshold: float = 0.5
) -> Tuple[int, int, int]:
    gt_boxes = [ann["bbox"] for ann in ground_truth]
    gt_matched = [False] * len(gt_boxes)

    tp = fp = 0

    # Count TP and FP
    for det in detections:
        if hasattr(det, "label") and hasattr(det, "final_score"):
            # Detection object (Stage 3b)
            if det.final_score >= threshold and det.label == 1:
                matched = False
                for j, gt_box in enumerate(gt_boxes):
                    if iou(det.bbox, gt_box) > IOU_THRESHOLD:
                        matched = True
                        gt_matched[j] = True
                        break
                if matched:
                    tp += 1
                else:
                    fp += 1
        else:
            # Proposal object (heuristic-only)
            if det.score >= threshold:
                matched = False
                for j, gt_box in enumerate(gt_boxes):
                    if iou(det.bbox, gt_box) > IOU_THRESHOLD:
                        matched = True
                        gt_matched[j] = True
                        break
                if matched:
                    tp += 1
                else:
                    fp += 1

    # Count FN (unmatched ground truth boxes)
    fn = sum(1 for matched in gt_matched if not matched)

    return tp, fp, fn


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute precision, recall, and F1 from confusion matrix components."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main():
    print("=" * 70)
    print("SLF Trap Detector — Benchmark Table Metrics")
    print("=" * 70)
    print()

    print("Configuration:")
    print(f"  Training images: {TRAINING_IMAGES} (seed {TRAINING_SEED})")
    print(f"  Test images: {TEST_IMAGES} (seed {TEST_SEED})")
    print(f"  Image size: {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}")
    print(f"  RF threshold: {RF_THRESHOLD}")
    print(f"  IoU threshold: {IOU_THRESHOLD}")
    print()

    print("Initializing pipeline components...")

    preprocessor = TrapImagePreprocessor(PreprocessorConfig(target_size=IMAGE_SIZE))

    proposer = RegionProposer(
        RegionProposerConfig(
            min_area_px=MIN_AREA_PX, max_area_px=MAX_AREA_PX, nms_iou_threshold=NMS_IOU
        )
    )

    heuristic_filter = FeatureFilter(
        FeatureFilterConfig(
            min_score=HEURISTIC_MIN_SCORE,
            min_aspect_ratio=HEURISTIC_ASPECT_RATIO_MIN,
            min_color_fraction=HEURISTIC_COLOR_FRACTION_MIN,
        )
    )

    print("Generating test dataset...")

    test_gen = SyntheticTrapGenerator(
        SyntheticConfig(insects_per_image_range=TEST_INSECTS_PER_IMAGE, seed=TEST_SEED)
    )

    test_images = []
    test_annotations = []

    for i in range(TEST_IMAGES):
        img, ann = test_gen.generate_one(image_id=i)
        test_images.append(img)
        test_annotations.append(ann)

    print(f"Generated {len(test_images)} test images")
    total_gt = sum(len(ann["annotations"]) for ann in test_annotations)
    print(f"Total ground truth instances: {total_gt}")
    print()

    print("Evaluating heuristic-only baseline...")
    t0 = time.time()

    heuristic_tp = heuristic_fp = heuristic_fn = 0

    for i, (img, ann) in enumerate(zip(test_images, test_annotations)):
        # Run preprocessing and region proposal
        processed = preprocessor(img).image
        proposals = proposer(processed)
        filtered = heuristic_filter(processed, proposals)

        # Evaluate against ground truth
        tp, fp, fn = evaluate_detections(
            filtered, ann["annotations"], threshold=HEURISTIC_MIN_SCORE
        )
        heuristic_tp += tp
        heuristic_fp += fp
        heuristic_fn += fn

    heuristic_time = time.time() - t0
    heuristic_metrics = compute_metrics(heuristic_tp, heuristic_fp, heuristic_fn)

    print(f"Heuristic baseline completed in {heuristic_time:.1f}s")
    print(f"  TP: {heuristic_tp}, FP: {heuristic_fp}, FN: {heuristic_fn}")
    print()

    print("Training HOG+LBP+RF classifier...")

    train_gen = SyntheticTrapGenerator(
        SyntheticConfig(
            insects_per_image_range=TRAINING_INSECTS_PER_IMAGE, seed=TRAINING_SEED
        )
    )

    pos_patches = []
    neg_patches = []

    print("  Collecting labeled patches from training images...")
    for i in range(TRAINING_IMAGES):
        img, ann = train_gen.generate_one(image_id=i)
        processed = preprocessor(img).image
        proposals = proposer(processed)
        filtered = heuristic_filter(processed, proposals)
        gt_boxes = [a["bbox"] for a in ann["annotations"]]

        H, W = processed.shape[:2]
        for proposal in filtered:
            x, y, w, h = proposal.bbox
            crop = processed[max(0, y) : min(H, y + h), max(0, x) : min(W, x + w)]
            if crop.size == 0:
                continue

            best_iou = max((iou(proposal.bbox, gt) for gt in gt_boxes), default=0)
            if best_iou >= 0.4:
                pos_patches.append(crop)
            elif best_iou < 0.1:
                neg_patches.append(crop)

    print(
        f"  Collected {len(pos_patches)} positive, {len(neg_patches)} negative patches"
    )

    t0 = time.time()
    classifier = DescriptorClassifier(
        DescriptorClassifierConfig(
            mode="random_forest",
            threshold=RF_THRESHOLD,
            rf_n_estimators=RF_N_ESTIMATORS,
            rf_max_depth=RF_MAX_DEPTH,
            model_cache_path=None,  # Don't cache during benchmark
        )
    )

    classifier.train_supervised(pos_patches, neg_patches)
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")
    print()

    print("Evaluating HOG+LBP+RF classifier...")
    t0 = time.time()

    rf_tp = rf_fp = rf_fn = 0
    detailed_results = []

    for i, (img, ann) in enumerate(zip(test_images, test_annotations)):
        processed = preprocessor(img).image
        proposals = proposer(processed)
        filtered = heuristic_filter(processed, proposals)

        H, W = processed.shape[:2]
        patches = []
        for proposal in filtered:
            x, y, w, h = proposal.bbox
            crop = processed[max(0, y) : min(H, y + h), max(0, x) : min(W, x + w)]
            patches.append(
                crop if crop.size > 0 else np.zeros((32, 32, 3), dtype=np.uint8)
            )

        if patches:
            scores = classifier.score_batch(patches)

            detections = []
            for proposal, score in zip(filtered, scores):
                detection = type(
                    "Detection",
                    (),
                    {
                        "bbox": proposal.bbox,
                        "final_score": score,
                        "label": int(score >= RF_THRESHOLD),
                    },
                )()
                detections.append(detection)
        else:
            detections = []

        tp, fp, fn = evaluate_detections(
            detections, ann["annotations"], threshold=RF_THRESHOLD
        )
        rf_tp += tp
        rf_fp += fp
        rf_fn += fn

        gt_count = len(ann["annotations"])
        det_count = sum(
            1 for d in detections if d.final_score >= RF_THRESHOLD and d.label == 1
        )
        detailed_results.append(
            {
                "image_id": i,
                "gt_count": gt_count,
                "detected_count": det_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    rf_time = time.time() - t0
    rf_metrics = compute_metrics(rf_tp, rf_fp, rf_fn)

    print(f"HOG+LBP+RF evaluation completed in {rf_time:.1f}s")
    print(f"  TP: {rf_tp}, FP: {rf_fp}, FN: {rf_fn}")
    print()

    print("=" * 70)
    print("RESULTS TABLE (Technical Note Section 2.2)")
    print("=" * 70)
    print()
    print(f"{'Metric':<20} {'Heuristic-only':<18} {'HOG+LBP+RF':<15}")
    print("-" * 60)
    print(
        f"{'Precision':<20} {heuristic_metrics['precision']:<18.1%} {rf_metrics['precision']:<15.1%}"
    )
    print(f"{'Recall':<20} {'—':<18} {rf_metrics['recall']:<15.1%}")
    print(f"{'F1':<20} {'—':<18} {rf_metrics['f1']:<15.1%}")
    print(f"{'GPU required':<20} {'No':<18} {'No':<15}")
    print(f"{'Real labels req.':<20} {'No':<18} {'No':<15}")
    print()
    print("Note: Both evaluations performed on synthetic test images only.")
    print("Real-world performance will differ until retrained on real data.")
    print()

    print("PER-IMAGE BREAKDOWN (subset):")
    print(f"{'Image':<6} {'GT':<4} {'Det':<5} {'TP':<4} {'FP':<4} {'FN':<4}")
    print("-" * 35)
    for result in detailed_results[:10]:  # Show first 10
        print(
            f"{result['image_id']:<6} {result['gt_count']:<4} "
            f"{result['detected_count']:<5} {result['tp']:<4} "
            f"{result['fp']:<4} {result['fn']:<4}"
        )
    if len(detailed_results) > 10:
        print(f"... ({len(detailed_results) - 10} more images)")
    print()

    benchmark_results = {
        "metadata": {
            "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_seed": TRAINING_SEED,
            "test_seed": TEST_SEED,
            "training_images": TRAINING_IMAGES,
            "test_images": TEST_IMAGES,
            "rf_threshold": RF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
        },
        "heuristic_baseline": {
            **heuristic_metrics,
            "evaluation_time_sec": heuristic_time,
        },
        "hog_lbp_rf": {
            **rf_metrics,
            "training_time_sec": train_time,
            "evaluation_time_sec": rf_time,
            "training_patches_pos": len(pos_patches),
            "training_patches_neg": len(neg_patches),
        },
        "per_image_results": detailed_results,
    }

    output_path = Path("benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"Detailed results saved to: {output_path}")
    print()
    print("Benchmark completed successfully!")


if __name__ == "__main__":
    main()
