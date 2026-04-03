"""
descriptor_classifier.py
────────────────────────
A classical feature-based classifier that operates without any deep learning
dependencies. Works in two modes:

  one_class_svm  : Cold-start — trained on SYNTHETIC positive patches only.
                   Uses a One-Class SVM to score candidates. Requires no real
                   labeled data at all.

  random_forest  : Supervised — trained on synthetic + confirmed real patches.
                   A Random Forest on the same descriptor. Drops in once
                   ~50+ confirmed real positives are available.

Feature vector (per patch, ~600-dim):
  - HOG (Histogram of Oriented Gradients): captures wing venation, body outline
  - LBP (Local Binary Pattern histogram): captures surface texture (spots, glue)
  - HSV histogram: encodes the colour distribution compactly
  - Shape statistics: aspect ratio, solidity, extent — fast and discriminative

The OC-SVM is trained on the fly from synthetic data the first time it's
needed, then cached. No internet, no GPU, no pretrained weights required.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)

PATCH_SIZE = (64, 128)  # (W, H) — standard HOG window, portrait orientation


@dataclass
class DescriptorClassifierConfig:
    mode: str = "one_class_svm"  # one_class_svm | random_forest
    patch_size: Tuple[int, int] = (64, 128)

    # HOG
    hog_orientations: int = 9
    hog_pixels_per_cell: Tuple[int, int] = (8, 8)
    hog_cells_per_block: Tuple[int, int] = (2, 2)

    # LBP
    lbp_radius: int = 3
    lbp_n_points: int = 24  # 8 * radius
    lbp_n_bins: int = 26  # n_points + 2

    # HSV histogram
    hsv_h_bins: int = 18
    hsv_s_bins: int = 8
    hsv_v_bins: int = 8

    # One-Class SVM
    ocsvm_nu: float = 0.15  # Expected fraction of outliers
    ocsvm_kernel: str = "rbf"
    ocsvm_gamma: str = "scale"

    # Random Forest
    rf_n_estimators: int = 200
    rf_max_depth: int = 12
    rf_class_weight: str = "balanced"

    threshold: float = 0.40  # Minimum score to accept as SLF

    # Auto-train from synthetic data if no model is cached
    auto_train_n_synthetic: int = 300
    model_cache_path: Optional[str] = "outputs/models/descriptor_classifier.pkl"


class DescriptorClassifier:
    """
    Classical HOG + LBP + HSV feature classifier for SLF patch scoring.

    Usage
    -----
    >>> clf = DescriptorClassifier()
    >>> clf.train_from_synthetic()          # Cold start — no real data needed
    >>> scores = clf.score_batch(patches)
    """

    def __init__(self, config: DescriptorClassifierConfig | None = None):
        self.cfg = config or DescriptorClassifierConfig()
        self._scaler: Optional[StandardScaler] = None
        self._model = None
        self._trained = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def score_batch(self, patches_bgr: List[np.ndarray]) -> List[float]:
        """
        Score a list of BGR image crops. Returns float scores in [0, 1].
        Automatically trains from synthetic data on first call if untrained.
        """
        if not patches_bgr:
            return []

        if not self._trained:
            self._auto_train()

        features = np.array([self._extract(p) for p in patches_bgr])
        features_scaled = self._scaler.transform(features)
        return self._predict_proba(features_scaled)

    def train_from_synthetic(
        self,
        n: int | None = None,
        extra_negatives: List[np.ndarray] | None = None,
    ) -> None:
        """
        Train the classifier using only synthetic SLF patches.

        In one_class_svm mode:   trains on synthetic positives only.
        In random_forest mode:   trains on synthetic positives + hard negatives.

        Parameters
        ----------
        n : number of synthetic patches to generate (default from config)
        extra_negatives : additional BGR patches confirmed as non-SLF
        """
        n = n or self.cfg.auto_train_n_synthetic
        logger.info("Generating %d synthetic patches for classifier training...", n)

        pos_patches = self._generate_synthetic_patches(n)
        logger.info("Generated %d positive patches.", len(pos_patches))

        pos_features = np.array([self._extract(p) for p in pos_patches])

        self._scaler = StandardScaler()

        if self.cfg.mode == "one_class_svm":
            X = self._scaler.fit_transform(pos_features)
            self._model = OneClassSVM(
                nu=self.cfg.ocsvm_nu,
                kernel=self.cfg.ocsvm_kernel,
                gamma=self.cfg.ocsvm_gamma,
            )
            self._model.fit(X)
            logger.info(
                "One-Class SVM trained on %d positive patches.", len(pos_patches)
            )

        elif self.cfg.mode == "random_forest":
            neg_patches = self._generate_hard_negatives(n)
            if extra_negatives:
                neg_patches += extra_negatives
            neg_features = np.array([self._extract(p) for p in neg_patches])

            X = np.vstack([pos_features, neg_features])
            y = np.array([1] * len(pos_features) + [0] * len(neg_features))
            X_scaled = self._scaler.fit_transform(X)

            self._model = RandomForestClassifier(
                n_estimators=self.cfg.rf_n_estimators,
                max_depth=self.cfg.rf_max_depth,
                class_weight=self.cfg.rf_class_weight,
                n_jobs=-1,
                random_state=42,
            )
            self._model.fit(X_scaled, y)
            logger.info(
                "Random Forest trained: %d pos / %d neg.",
                len(pos_features),
                len(neg_features),
            )
        else:
            raise ValueError(f"Unknown mode: {self.cfg.mode}")

        self._trained = True
        self._maybe_save()

    def train_supervised(
        self,
        positive_patches: List[np.ndarray],
        negative_patches: List[np.ndarray],
    ) -> None:
        """Train a Random Forest on confirmed real positive and negative patches."""
        self.cfg.mode = "random_forest"
        syn_pos = self._generate_synthetic_patches(max(50, len(positive_patches)))
        all_pos = positive_patches + syn_pos
        all_neg = negative_patches + self._generate_hard_negatives(len(all_pos))

        pos_feat = np.array([self._extract(p) for p in all_pos])
        neg_feat = np.array([self._extract(p) for p in all_neg])
        X = np.vstack([pos_feat, neg_feat])
        y = np.array([1] * len(pos_feat) + [0] * len(neg_feat))

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._model = RandomForestClassifier(
            n_estimators=self.cfg.rf_n_estimators,
            max_depth=self.cfg.rf_max_depth,
            class_weight=self.cfg.rf_class_weight,
            n_jobs=-1,
            random_state=42,
        )
        self._model.fit(X_scaled, y)
        self._trained = True
        logger.info(
            "Supervised RF trained: %d pos / %d neg.", len(all_pos), len(all_neg)
        )
        self._maybe_save()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"model": self._model, "scaler": self._scaler, "mode": self.cfg.mode}, f
            )
        logger.info("Classifier saved → %s", path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._model = state["model"]
        self._scaler = state["scaler"]
        self.cfg.mode = state["mode"]
        self._trained = True
        logger.info("Classifier loaded from %s (mode=%s)", path, self.cfg.mode)

    # ──────────────────────────────────────────────────────────────────────────
    # Feature extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _extract(self, patch_bgr: np.ndarray) -> np.ndarray:
        """Extract the full feature vector from a single BGR patch."""
        W, H = self.cfg.patch_size
        patch = cv2.resize(patch_bgr, (W, H), interpolation=cv2.INTER_AREA)

        hog_feat = self._hog_features(patch)
        lbp_feat = self._lbp_features(patch)
        hsv_feat = self._hsv_features(patch)
        shape_feat = self._shape_features(patch)

        return np.concatenate([hog_feat, lbp_feat, hsv_feat, shape_feat])

    def _hog_features(self, patch_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        feat = hog(
            gray,
            orientations=self.cfg.hog_orientations,
            pixels_per_cell=self.cfg.hog_pixels_per_cell,
            cells_per_block=self.cfg.hog_cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return feat.astype(np.float32)

    def _lbp_features(self, patch_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(
            gray,
            P=self.cfg.lbp_n_points,
            R=self.cfg.lbp_radius,
            method="uniform",
        )
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=self.cfg.lbp_n_bins,
            range=(0, self.cfg.lbp_n_bins),
            density=True,
        )
        return hist.astype(np.float32)

    def _hsv_features(self, patch_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist(
            [hsv], [0], None, [self.cfg.hsv_h_bins], [0, 180]
        ).flatten()
        s_hist = cv2.calcHist(
            [hsv], [1], None, [self.cfg.hsv_s_bins], [0, 256]
        ).flatten()
        v_hist = cv2.calcHist(
            [hsv], [2], None, [self.cfg.hsv_v_bins], [0, 256]
        ).flatten()
        feat = np.concatenate([h_hist, s_hist, v_hist])
        total = feat.sum() + 1e-8
        return (feat / total).astype(np.float32)

    def _shape_features(self, patch_bgr: np.ndarray) -> np.ndarray:
        """
        Compact shape descriptor from the largest foreground contour.
        Captures: aspect ratio, solidity, extent, ellipse eccentricity,
        hu moments (7), and contour area fraction.
        """
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return np.zeros(17, dtype=np.float32)

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_area = w * h + 1e-8

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) + 1e-8

        aspect = w / max(h, 1)
        solidity = area / hull_area
        extent = area / bbox_area

        # Ellipse fit — gives eccentricity and orientation
        if len(cnt) >= 5:
            (ex, ey), (ma, Mi), angle = cv2.fitEllipse(cnt)
            eccentricity = np.sqrt(1 - (min(ma, Mi) / (max(ma, Mi) + 1e-8)) ** 2)
        else:
            eccentricity = 0.0
            angle = 0.0

        # Hu moments — rotation/scale invariant shape signature
        moments = cv2.moments(cnt)
        hu = cv2.HuMoments(moments).flatten()
        # Log-transform for numerical stability
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        return np.array(
            [aspect, solidity, extent, eccentricity, area / (64 * 128)]
            + hu_log.tolist(),
            dtype=np.float32,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Scoring
    # ──────────────────────────────────────────────────────────────────────────

    def _predict_proba(self, X_scaled: np.ndarray) -> List[float]:
        if self.cfg.mode == "one_class_svm":
            # decision_function returns signed distance from boundary
            # positive = inlier (SLF-like), negative = outlier
            scores_raw = self._model.decision_function(X_scaled)
            # Sigmoid-normalise to [0,1]
            scores = 1.0 / (1.0 + np.exp(-scores_raw * 2.0))
            return scores.tolist()

        elif self.cfg.mode == "random_forest":
            proba = self._model.predict_proba(X_scaled)[:, 1]
            return proba.tolist()

        return [0.5] * len(X_scaled)

    # ──────────────────────────────────────────────────────────────────────────
    # Synthetic patch generation (internal training data)
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_synthetic_patches(self, n: int) -> List[np.ndarray]:
        """Generate synthetic SLF-like patches for OC-SVM / RF training."""
        from src.augmentation.synthetic_generator import (
            SyntheticTrapGenerator,
            SyntheticConfig,
        )

        rng = np.random.default_rng(0)
        gen = SyntheticTrapGenerator(
            SyntheticConfig(
                image_size=(512, 512),
                insects_per_image_range=(4, 10),
                seed=0,
            )
        )
        patches = []
        images_needed = max(1, n // 6)
        for i in range(images_needed):
            img, ann = gen.generate_one(i)
            for bbox_ann in ann["annotations"]:
                x, y, w, h = [int(v) for v in bbox_ann["bbox"]]
                # Expand bbox slightly for context
                pad = 8
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(512, x + w + pad)
                y2 = min(512, y + h + pad)
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    patches.append(crop)
                if len(patches) >= n:
                    break
            if len(patches) >= n:
                break

        # Augment with flips, brightness jitter
        augmented = []
        for p in patches:
            augmented.append(p)
            augmented.append(cv2.flip(p, 1))
            # Brightness jitter
            factor = rng.uniform(0.7, 1.3)
            bright = np.clip(p.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            augmented.append(bright)

        return augmented[:n]

    def _generate_hard_negatives(self, n: int) -> List[np.ndarray]:
        """
        Generate hard negative patches: debris, non-SLF shapes, plain background.
        These are the most common false positive sources.
        """
        rng = np.random.default_rng(1)
        patches = []
        W, H = self.cfg.patch_size

        while len(patches) < n:
            patch_type = rng.choice(["debris", "blob", "background", "elongated"])
            p = np.zeros((H * 2, W * 2, 3), dtype=np.uint8)

            # Yellow-ish trap board background
            base_col = rng.integers(150, 220)
            p[:] = [
                int(base_col * 0.55),
                int(base_col * 0.85),
                int(base_col),
            ]

            if patch_type == "debris":
                # Random dark fragments — twigs, leaf edges
                for _ in range(rng.integers(2, 8)):
                    x1, y1 = rng.integers(0, W * 2, size=2)
                    angle = rng.uniform(0, np.pi)
                    length = rng.integers(15, 60)
                    x2 = int(x1 + length * np.cos(angle))
                    y2 = int(y1 + length * np.sin(angle))
                    col = tuple(rng.integers(20, 80, size=3).tolist())
                    cv2.line(p, (x1, y1), (x2, y2), col, rng.integers(1, 5))

            elif patch_type == "blob":
                # Random circular blobs — non-SLF insects, seeds
                cx, cy = rng.integers(20, W * 2 - 20, size=2)
                r = rng.integers(8, 30)
                col = tuple(rng.integers(30, 180, size=3).tolist())
                cv2.circle(p, (int(cx), int(cy)), int(r), col, -1)

            elif patch_type == "background":
                # Plain background with noise — no insect at all
                noise = rng.integers(-15, 15, p.shape).astype(np.int16)
                p = np.clip(p.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            elif patch_type == "elongated":
                # Very elongated shapes — twigs, plant stems
                cx, cy = rng.integers(20, W * 2 - 20, size=2)
                rx = rng.integers(30, 55)
                ry = rng.integers(3, 10)
                angle = rng.uniform(0, 180)
                col = tuple(rng.integers(50, 120, size=3).tolist())
                cv2.ellipse(
                    p, (int(cx), int(cy)), (int(rx), int(ry)), angle, 0, 360, col, -1
                )

            # Crop to patch size
            patches.append(cv2.resize(p, (W, H)))

        return patches[:n]

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _auto_train(self) -> None:
        if self.cfg.model_cache_path:
            cache = Path(self.cfg.model_cache_path)
            if cache.exists():
                self.load(cache)
                return
        logger.info("No cached model found. Auto-training from synthetic data...")
        self.train_from_synthetic()

    def _maybe_save(self) -> None:
        if self.cfg.model_cache_path:
            self.save(self.cfg.model_cache_path)
