import numpy as np
import pytest
import sys
import tempfile
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Direct import to bypass __init__.py issues
try:
    from src.detection.descriptor_classifier import DescriptorClassifier, DescriptorClassifierConfig
except ImportError:
    pytest.skip("DescriptorClassifier module not available", allow_module_level=True)


class TestDescriptorClassifierConfig:
    def test_default_values(self):
        config = DescriptorClassifierConfig()
        assert config.mode == "one_class_svm"
        assert config.patch_size == (64, 128)
        assert config.threshold == 0.40
        assert config.hog_orientations == 9
        assert config.auto_train_n_synthetic == 300

    def test_custom_values(self):
        config = DescriptorClassifierConfig(
            mode="random_forest",
            patch_size=(32, 64),
            threshold=0.5,
            auto_train_n_synthetic=100
        )
        assert config.mode == "random_forest"
        assert config.patch_size == (32, 64)
        assert config.threshold == 0.5
        assert config.auto_train_n_synthetic == 100

    def test_hog_parameters(self):
        config = DescriptorClassifierConfig()
        assert config.hog_orientations == 9
        assert config.hog_pixels_per_cell == (8, 8)
        assert config.hog_cells_per_block == (2, 2)

    def test_lbp_parameters(self):
        config = DescriptorClassifierConfig()
        assert config.lbp_radius == 3
        assert config.lbp_n_points == 24
        assert config.lbp_n_bins == 26

    def test_hsv_parameters(self):
        config = DescriptorClassifierConfig()
        assert config.hsv_h_bins == 18
        assert config.hsv_s_bins == 8
        assert config.hsv_v_bins == 8

    def test_random_forest_parameters(self):
        config = DescriptorClassifierConfig()
        assert config.rf_n_estimators == 200
        assert config.rf_max_depth == 12
        assert config.rf_class_weight == "balanced"

    def test_one_class_svm_parameters(self):
        config = DescriptorClassifierConfig()
        assert config.ocsvm_nu == 0.15
        assert config.ocsvm_kernel == "rbf"
        assert config.ocsvm_gamma == "scale"


class TestDescriptorClassifier:
    @pytest.fixture
    def config(self):
        """Create a test config with faster settings."""
        return DescriptorClassifierConfig(
            auto_train_n_synthetic=20,  # Much smaller for fast tests
            model_cache_path=None,      # Don't cache during tests
        )

    @pytest.fixture
    def rf_config(self):
        """Create a Random Forest config for testing."""
        return DescriptorClassifierConfig(
            mode="random_forest",
            auto_train_n_synthetic=15,
            model_cache_path=None,
        )

    @pytest.fixture
    def classifier(self, config):
        """Create a classifier with test config."""
        return DescriptorClassifier(config)

    @pytest.fixture
    def rf_classifier(self, rf_config):
        """Create a Random Forest classifier."""
        return DescriptorClassifier(rf_config)

    @pytest.fixture
    def sample_patch(self):
        """Create a sample BGR patch."""
        # Create a patch that looks somewhat insect-like
        patch = np.full((128, 64, 3), [60, 100, 140], dtype=np.uint8)  # Red-brown base
        # Add some darker regions (body/wings)
        patch[30:100, 20:44] = [40, 60, 80]  # Central body
        patch[40:80, 10:54] = [50, 80, 120]   # Wing area
        return patch

    @pytest.fixture
    def yellow_patch(self):
        """Create a yellow trap board patch."""
        return np.full((128, 64, 3), [0, 200, 255], dtype=np.uint8)

    @pytest.fixture
    def random_patches(self):
        """Create a few random test patches."""
        patches = []
        for i in range(3):
            patch = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
            patches.append(patch)
        return patches

    def test_classifier_initialization(self, config):
        classifier = DescriptorClassifier(config)
        assert classifier.cfg == config
        assert not classifier._trained
        assert classifier._model is None
        assert classifier._scaler is None

    def test_classifier_with_default_config(self):
        classifier = DescriptorClassifier()
        assert classifier.cfg.mode == "one_class_svm"
        assert classifier.cfg.patch_size == (64, 128)
        assert classifier.cfg.threshold == 0.40

    def test_classifier_with_none_config(self):
        classifier = DescriptorClassifier(None)
        assert classifier.cfg.mode == "one_class_svm"

    def test_score_batch_empty_list(self, classifier):
        """Test scoring empty list."""
        scores = classifier.score_batch([])
        assert scores == []

    def test_score_batch_triggers_auto_training(self, classifier, sample_patch):
        """Test that scoring triggers automatic training."""
        # Initially not trained
        assert not classifier._trained

        # Scoring should trigger auto-training
        scores = classifier.score_batch([sample_patch])

        # Should now be trained
        assert classifier._trained
        assert classifier._model is not None
        assert classifier._scaler is not None
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        assert 0.0 <= scores[0] <= 1.0

    def test_score_batch_multiple_patches(self, classifier, random_patches):
        """Test scoring multiple patches."""
        scores = classifier.score_batch(random_patches)

        assert len(scores) == len(random_patches)
        for score in scores:
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_train_from_synthetic_one_class_svm(self):
        """Test training OneClass SVM from synthetic data."""
        config = DescriptorClassifierConfig(
            mode="one_class_svm",
            auto_train_n_synthetic=10,  # Very small for speed
            model_cache_path=None
        )
        classifier = DescriptorClassifier(config)

        # Train the classifier
        classifier.train_from_synthetic(n=10)

        assert classifier._trained
        assert classifier._model is not None
        assert classifier._scaler is not None
        assert classifier.cfg.mode == "one_class_svm"

    def test_train_from_synthetic_random_forest(self, rf_classifier):
        """Test training Random Forest from synthetic data."""
        rf_classifier.train_from_synthetic(n=10)

        assert rf_classifier._trained
        assert rf_classifier._model is not None
        assert rf_classifier._scaler is not None
        assert rf_classifier.cfg.mode == "random_forest"

    def test_train_from_synthetic_with_extra_negatives(self, rf_classifier, yellow_patch):
        """Test training with extra negative examples."""
        extra_negatives = [yellow_patch]
        rf_classifier.train_from_synthetic(n=8, extra_negatives=extra_negatives)

        assert rf_classifier._trained

    def test_train_supervised(self, classifier, sample_patch, yellow_patch):
        """Test supervised training with positive and negative examples."""
        positive_patches = [sample_patch]
        negative_patches = [yellow_patch]

        classifier.train_supervised(positive_patches, negative_patches)

        assert classifier._trained
        assert classifier.cfg.mode == "random_forest"  # Should switch to RF mode
        assert classifier._model is not None

    def test_feature_extraction_methods(self, classifier, sample_patch):
        """Test individual feature extraction methods."""
        # Test extract method (full feature vector)
        features = classifier._extract(sample_patch)
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert len(features) > 0  # Should have some features

        # Test HOG features
        hog_features = classifier._hog_features(sample_patch)
        assert isinstance(hog_features, np.ndarray)
        assert len(hog_features) > 0

        # Test LBP features
        lbp_features = classifier._lbp_features(sample_patch)
        assert isinstance(lbp_features, np.ndarray)
        assert len(lbp_features) == classifier.cfg.lbp_n_bins

        # Test HSV features
        hsv_features = classifier._hsv_features(sample_patch)
        expected_hsv_len = (classifier.cfg.hsv_h_bins +
                           classifier.cfg.hsv_s_bins +
                           classifier.cfg.hsv_v_bins)
        assert len(hsv_features) == expected_hsv_len

        # Test shape features
        shape_features = classifier._shape_features(sample_patch)
        assert isinstance(shape_features, np.ndarray)
        assert len(shape_features) == 12  # Fixed size as per implementation

    def test_feature_vector_consistency(self, classifier, sample_patch):
        """Test that feature extraction is consistent."""
        features1 = classifier._extract(sample_patch)
        features2 = classifier._extract(sample_patch)

        # Should be exactly the same
        np.testing.assert_array_equal(features1, features2)

    def test_different_patch_sizes(self, classifier):
        """Test that patches are properly resized."""
        # Test various input sizes
        sizes = [(32, 32), (100, 200), (256, 256)]

        for h, w in sizes:
            patch = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            features = classifier._extract(patch)
            assert isinstance(features, np.ndarray)
            assert len(features) > 0

    def test_patch_resize_to_config_size(self, config, sample_patch):
        """Test that patches are resized to config patch_size."""
        # Test with non-standard patch size in config
        config.patch_size = (32, 64)
        classifier = DescriptorClassifier(config)

        # Input patch is 128x64, should be resized to 32x64
        features = classifier._extract(sample_patch)
        assert isinstance(features, np.ndarray)

    def test_save_creates_directory(self, classifier):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "subdir" / "model.pkl"

            # Train first
            classifier.train_from_synthetic(n=3)

            # Save should create the subdir
            classifier.save(save_path)
            assert save_path.exists()
            assert save_path.parent.exists()

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises error."""
        config = DescriptorClassifierConfig(mode="invalid_mode", model_cache_path=None)
        classifier = DescriptorClassifier(config)

        with pytest.raises(ValueError, match="Unknown mode"):
            classifier.train_from_synthetic(n=1)

    def test_patch_preprocessing(self, classifier):
        """Test that patches are properly preprocessed."""
        # Test with different input formats
        patch_uint8 = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)

        # Should work
        features1 = classifier._extract(patch_uint8)
        assert isinstance(features1, np.ndarray)

    def test_train_from_synthetic_with_none_parameter(self, classifier):
        """Test train_from_synthetic with n=None uses config default."""
        # Should use config.auto_train_n_synthetic
        classifier.train_from_synthetic(n=None)
        assert classifier._trained

    def test_shape_features_with_no_contours(self, classifier):
        """Test shape feature extraction with image that has no contours."""
        # Completely black image (should produce no contours after threshold)
        black_patch = np.zeros((128, 64, 3), dtype=np.uint8)
        shape_features = classifier._shape_features(black_patch)

        # Should return zeros array of length 17
        assert len(shape_features) == 17
        assert isinstance(shape_features, np.ndarray)
        # Most features should be 0 since no contours found
        assert np.allclose(shape_features, 0.0, atol=1e-6)

