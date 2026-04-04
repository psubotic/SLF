# MIT License
# Copyright (c) 2026 Pavle Subotic

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def draw_detections(
    image_bgr: np.ndarray,
    detections,  # List[Detection] from pipeline
    color_positive: Tuple[int, int, int] = (0, 220, 80),
    color_negative: Tuple[int, int, int] = (60, 60, 220),
    thickness: int = 2,
    font_scale: float = 0.45,
    show_score: bool = True,
    show_negatives: bool = False,
) -> np.ndarray:
    vis = image_bgr.copy()

    for det in detections:
        if det.label == 0 and not show_negatives:
            continue

        color = color_positive if det.label == 1 else (160, 160, 160)
        x, y, w, h = det.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)

        if show_score:
            label_text = (
                f"SLF {det.final_score:.2f}"
                if det.label == 1
                else f"? {det.final_score:.2f}"
            )
            cv2.putText(
                vis,
                label_text,
                (x, max(y - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )

    return vis


def draw_proposals(
    image_bgr: np.ndarray,
    proposals,
    color: Tuple[int, int, int] = (200, 200, 0),
    thickness: int = 1,
) -> np.ndarray:
    vis = image_bgr.copy()
    for p in proposals:
        x, y, w, h = p.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
    return vis


def add_summary_overlay(
    image_bgr: np.ndarray,
    slf_count: int,
    elapsed_sec: float,
    mode: str = "",
) -> np.ndarray:
    vis = image_bgr.copy()
    lines = [
        f"SLF Adults: {slf_count}",
        f"Time: {elapsed_sec:.2f}s",
        f"Mode: {mode}" if mode else "",
    ]
    lines = [ln for ln in lines if ln]

    padding = 8
    line_h = 22
    box_h = padding * 2 + line_h * len(lines)
    box_w = 220

    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.55, vis, 0.45, 0)

    for i, line in enumerate(lines):
        y = padding + (i + 1) * line_h - 4
        cv2.putText(
            vis,
            line,
            (padding, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def make_comparison_grid(
    images: List[np.ndarray],
    labels: Optional[List[str]] = None,
    n_cols: int = 3,
    cell_size: Tuple[int, int] = (320, 320),
) -> np.ndarray:
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    H, W = cell_size

    grid = np.zeros((n_rows * H, n_cols * W, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        row, col = divmod(i, n_cols)
        resized = cv2.resize(img, (W, H))
        grid[row * H : (row + 1) * H, col * W : (col + 1) * W] = resized
        if labels and i < len(labels):
            cv2.putText(
                grid,
                labels[i],
                (col * W + 5, row * H + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return grid
