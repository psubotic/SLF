# MIT License
# Copyright (c) 2026 Pavle Subotic

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyntheticConfig:
    image_size: Tuple[int, int] = (1024, 1024)
    insects_per_image_range: Tuple[int, int] = (1, 8)
    slf_scale_range: Tuple[float, float] = (0.03, 0.12)
    rotation_range: Tuple[float, float] = (-180.0, 180.0)
    brightness_jitter: float = 0.3
    noise_std: float = 5.0
    occlusion_probability: float = 0.25
    debris_probability: float = 0.4
    n_debris_range: Tuple[int, int] = (2, 15)
    seed: Optional[int] = None


class SyntheticTrapGenerator:
    def __init__(
        self,
        config: SyntheticConfig | None = None,
        slf_reference_dir: Optional[Path] = None,
    ):
        self.cfg = config or SyntheticConfig()
        self.slf_references: List[np.ndarray] = []
        if slf_reference_dir:
            self._load_references(slf_reference_dir)

        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

    def generate_batch(self, n: int) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Generate n synthetic trap images.

        Returns
        -------
        images : list of BGR numpy arrays
        annotations : list of COCO-compatible annotation dicts
        """
        images, annotations = [], []
        for i in range(n):
            img, ann = self.generate_one(image_id=i)
            images.append(img)
            annotations.append(ann)
            if (i + 1) % 10 == 0:
                logger.info("Generated %d / %d synthetic images", i + 1, n)
        return images, annotations

    def generate_one(self, image_id: int = 0) -> Tuple[np.ndarray, Dict]:
        """Generate a single synthetic trap image with COCO annotation."""
        H, W = self.cfg.image_size

        # 1. Background: sticky board texture
        background = self._make_background(H, W)

        # 2. Optionally add debris
        if random.random() < self.cfg.debris_probability:
            background = self._add_debris(background)

        # 3. Paste SLF specimens
        n_insects = random.randint(*self.cfg.insects_per_image_range)
        bboxes = []
        for _ in range(n_insects):
            background, bbox = self._paste_insect(background)
            if bbox is not None:
                bboxes.append(bbox)

        # 4. Global noise + lighting
        background = self._add_lighting_gradient(background)
        background = self._add_gaussian_noise(background)

        annotation = {
            "image_id": image_id,
            "width": W,
            "height": H,
            "annotations": [
                {
                    "bbox": list(bbox),  # [x, y, w, h]
                    "category_id": 1,  # 1 = SLF adult
                    "category_name": "slf_adult",
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
                for bbox in bboxes
            ],
        }
        return background, annotation

    def save_batch(
        self,
        images: List[np.ndarray],
        annotations: List[Dict],
        output_dir: str | Path,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        img_dir = output_dir / "images"
        img_dir.mkdir(exist_ok=True)

        coco_output = {
            "info": {"description": "Synthetic SLF trap images"},
            "categories": [{"id": 1, "name": "slf_adult", "supercategory": "insect"}],
            "images": [],
            "annotations": [],
        }
        ann_id = 0
        for i, (img, ann) in enumerate(zip(images, annotations)):
            fname = f"synthetic_{i:05d}.jpg"
            cv2.imwrite(str(img_dir / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            coco_output["images"].append(
                {
                    "id": i,
                    "file_name": fname,
                    "width": ann["width"],
                    "height": ann["height"],
                }
            )
            for bbox_ann in ann["annotations"]:
                coco_output["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": i,
                        "category_id": bbox_ann["category_id"],
                        "bbox": bbox_ann["bbox"],
                        "area": bbox_ann["area"],
                        "iscrowd": bbox_ann["iscrowd"],
                    }
                )
                ann_id += 1

        ann_path = output_dir / "annotations.json"
        with open(ann_path, "w") as f:
            json.dump(coco_output, f, indent=2)

        logger.info(
            "Saved %d synthetic images and annotations to %s", len(images), output_dir
        )

    # ------------------------------------------------------------------
    # Internal: background generation
    # ------------------------------------------------------------------

    def _make_background(self, H: int, W: int) -> np.ndarray:
        """
        Simulate a sticky yellow/brown pheromone trap board.
        SLF traps are typically bright yellow or white boards.
        """
        # Base colour: pale yellow (typical commercial SLF trap colour)
        base_hue = random.randint(20, 35)  # Yellow-orange hue in HSV
        base_sat = random.randint(150, 230)
        base_val = random.randint(180, 240)

        bg = np.full((H, W, 3), [base_hue, base_sat, base_val], dtype=np.uint8)
        bg_bgr = cv2.cvtColor(bg, cv2.COLOR_HSV2BGR)

        # Add subtle texture: simplex-like noise via sum of random blurs
        noise = np.random.randint(0, 20, (H, W, 3), dtype=np.uint8)
        blur_scale = random.choice([31, 63, 127])
        texture = cv2.GaussianBlur(noise, (blur_scale, blur_scale), 0)
        bg_bgr = cv2.add(bg_bgr, texture)

        # Glue sheen: occasional specular highlight blobs
        n_sheen = random.randint(0, 5)
        for _ in range(n_sheen):
            cx, cy = random.randint(0, W), random.randint(0, H)
            r = random.randint(20, 120)
            brightness = random.randint(220, 255)
            cv2.circle(bg_bgr, (cx, cy), r, (brightness, brightness, brightness), -1)
            bg_bgr = cv2.GaussianBlur(bg_bgr, (51, 51), 0)

        return bg_bgr

    # ------------------------------------------------------------------
    # Internal: insect rendering
    # ------------------------------------------------------------------

    def _paste_insect(
        self, background: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        H, W = background.shape[:2]

        if self.slf_references:
            insect = random.choice(self.slf_references).copy()
        else:
            insect = self._draw_procedural_slf()

        # Scale
        scale = random.uniform(*self.cfg.slf_scale_range)
        target_w = max(20, int(W * scale))
        aspect = insect.shape[1] / max(insect.shape[0], 1)
        target_h = max(10, int(target_w / aspect))
        insect = cv2.resize(insect, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Rotate
        angle = random.uniform(*self.cfg.rotation_range)
        insect = self._rotate_with_alpha(insect, angle)

        ih, iw = insect.shape[:2]

        # Position: random, allow partial occlusion at edges
        x = random.randint(-iw // 3, W - iw // 3 * 2)
        y = random.randint(-ih // 3, H - ih // 3 * 2)

        # Partial occlusion by another insect-shaped patch
        if random.random() < self.cfg.occlusion_probability:
            occ_mask = self._random_occlusion_mask(ih, iw)
        else:
            occ_mask = None

        # Composite
        background = self._composite(background, insect, x, y, occ_mask)

        # Compute visible bbox
        x_clip = max(0, x)
        y_clip = max(0, y)
        x2_clip = min(W, x + iw)
        y2_clip = min(H, y + ih)
        if x2_clip <= x_clip or y2_clip <= y_clip:
            return background, None

        bbox = (x_clip, y_clip, x2_clip - x_clip, y2_clip - y_clip)
        return background, bbox

    def _draw_procedural_slf(self) -> np.ndarray:
        W, H = 120, 60  # Approximate pixel
        img = np.zeros((H, W, 4), dtype=np.uint8)  # BGRA

        # --- Forewing (dominant visible surface) ---
        # Base: mottled grey-brown
        fw_colour = np.array(
            [
                random.randint(80, 130),  # B
                random.randint(80, 120),  # G
                random.randint(100, 160),  # R
                255,
            ],
            dtype=np.uint8,
        )
        cv2.ellipse(
            img,
            (W // 2, H // 2),
            (W // 2 - 4, H // 2 - 6),
            0,
            0,
            360,
            fw_colour.tolist(),
            -1,
        )

        # Black spots on forewings (characteristic pattern)
        n_spots = random.randint(6, 14)
        for _ in range(n_spots):
            sx = random.randint(10, W - 10)
            sy = random.randint(8, H - 8)
            sr = random.randint(2, 5)
            cv2.circle(img, (sx, sy), sr, (20, 20, 20, 255), -1)

        # --- Red accent near wing base (visible when wings slightly open) ---
        red_visible = random.random() > 0.4
        if red_visible:
            red_x = random.randint(W // 3, 2 * W // 3)
            cv2.ellipse(
                img,
                (red_x, H // 2),
                (15, 8),
                0,
                0,
                360,
                (30, 30, 200, 200),
                -1,  # Red in BGR
            )

        # --- Body outline (darker centre stripe) ---
        cv2.ellipse(
            img, (W // 2, H // 2), (8, H // 2 - 4), 0, 0, 360, (30, 30, 40, 255), -1
        )

        # --- Antennae ---
        ant_len = random.randint(20, 35)
        cv2.line(img, (W // 2, 4), (W // 2 - ant_len, 2), (20, 20, 20, 255), 1)
        cv2.line(img, (W // 2, 4), (W // 2 + ant_len, 2), (20, 20, 20, 255), 1)

        # Slight blur to soften hard edges
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        bgr = cv2.GaussianBlur(bgr, (3, 3), 0)
        img = np.dstack([bgr, alpha])

        return img

    @staticmethod
    def _rotate_with_alpha(img_bgra: np.ndarray, angle: float) -> np.ndarray:
        """Rotate BGRA image, keeping alpha intact."""
        h, w = img_bgra.shape[:2]
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += new_w / 2 - cx
        M[1, 2] += new_h / 2 - cy
        return cv2.warpAffine(img_bgra, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

    @staticmethod
    def _composite(
        background: np.ndarray,
        insect_bgra: np.ndarray,
        x: int,
        y: int,
        occlusion_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Alpha-composite insect onto background."""
        bg = background.copy()
        H, W = bg.shape[:2]
        ih, iw = insect_bgra.shape[:2]

        sx, sy = max(0, -x), max(0, -y)
        ex, ey = min(iw, W - x), min(ih, H - y)
        dx, dy = max(0, x), max(0, y)

        if ex <= sx or ey <= sy:
            return bg

        patch = insect_bgra[sy:ey, sx:ex]
        alpha = patch[:, :, 3:4].astype(np.float32) / 255.0

        if occlusion_mask is not None:
            occ = occlusion_mask[sy:ey, sx:ex, np.newaxis].astype(np.float32) / 255.0
            alpha = alpha * occ

        region = bg[dy : dy + (ey - sy), dx : dx + (ex - sx)].astype(np.float32)
        blended = alpha * patch[:, :, :3].astype(np.float32) + (1 - alpha) * region
        bg[dy : dy + (ey - sy), dx : dx + (ex - sx)] = blended.astype(np.uint8)
        return bg

    @staticmethod
    def _random_occlusion_mask(h: int, w: int) -> np.ndarray:
        """Random rectangular occlusion (simulates overlap between insects)."""
        mask = np.ones((h, w), dtype=np.uint8) * 255
        occ_x = random.randint(0, w // 2)
        occ_y = random.randint(0, h // 2)
        occ_w = random.randint(w // 4, w // 2)
        occ_h = random.randint(h // 4, h // 2)
        mask[occ_y : occ_y + occ_h, occ_x : occ_x + occ_w] = 0
        return mask

    def _add_debris(self, bg: np.ndarray) -> np.ndarray:
        H, W = bg.shape[:2]
        n = random.randint(*self.cfg.n_debris_range)
        for _ in range(n):
            dtype = random.choice(["line", "blob", "fragment"])
            colour = (
                random.randint(30, 100),
                random.randint(30, 100),
                random.randint(30, 100),
            )
            if dtype == "line":
                x1, y1 = random.randint(0, W), random.randint(0, H)
                length = random.randint(10, 80)
                angle = random.uniform(0, np.pi)
                x2 = int(x1 + length * np.cos(angle))
                y2 = int(y1 + length * np.sin(angle))
                cv2.line(bg, (x1, y1), (x2, y2), colour, random.randint(1, 3))
            elif dtype == "blob":
                cx, cy = random.randint(0, W), random.randint(0, H)
                rx, ry = random.randint(3, 20), random.randint(3, 12)
                cv2.ellipse(
                    bg, (cx, cy), (rx, ry), random.uniform(0, 360), 0, 360, colour, -1
                )
            else:
                pts = np.array(
                    [
                        [random.randint(0, W), random.randint(0, H)]
                        for _ in range(random.randint(3, 7))
                    ],
                    dtype=np.int32,
                )
                cv2.fillPoly(bg, [pts], colour)
        return bg

    def _add_gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.cfg.noise_std, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy

    def _add_lighting_gradient(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        direction = random.choice(["horizontal", "vertical", "radial"])
        strength = random.uniform(0.1, self.cfg.brightness_jitter)

        if direction == "horizontal":
            gradient = np.linspace(1 - strength, 1 + strength, W).reshape(1, W, 1)
            gradient = np.tile(gradient, (H, 1, 3))
        elif direction == "vertical":
            gradient = np.linspace(1 - strength, 1 + strength, H).reshape(H, 1, 1)
            gradient = np.tile(gradient, (1, W, 3))
        else:
            cx, cy = random.randint(W // 4, 3 * W // 4), random.randint(
                H // 4, 3 * H // 4
            )
            Y, X = np.ogrid[:H, :W]
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            max_dist = np.sqrt(H**2 + W**2) / 2
            gradient = (1 + strength) - (strength * dist / max_dist)
            gradient = gradient[:, :, np.newaxis]
            gradient = np.tile(gradient, (1, 1, 3))

        result = np.clip(img.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
        return result

    def _load_references(self, ref_dir: Path) -> None:
        ref_dir = Path(ref_dir)
        for p in ref_dir.glob("*.png"):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] == 4:
                self.slf_references.append(img)
        logger.info(
            "Loaded %d SLF reference cutouts from %s", len(self.slf_references), ref_dir
        )
