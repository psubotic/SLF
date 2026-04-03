from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np


def load_image(
    path: str | Path, target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def load_coco_annotations(ann_path: str | Path) -> Dict:
    with open(ann_path) as f:
        return json.load(f)


def iter_images(
    directory: str | Path,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tiff", ".bmp"),
) -> Iterator[Path]:
    directory = Path(directory)
    for ext in extensions:
        yield from sorted(directory.glob(f"*{ext}"))
        yield from sorted(directory.glob(f"*{ext.upper()}"))


def save_result_json(result_dict: Dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)


def batch_iter(items: List, batch_size: int) -> Iterator[List]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
