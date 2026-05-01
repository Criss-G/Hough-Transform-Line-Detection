"""Unit tests for ImagePreprocessor class."""

import pytest
import numpy as np
import cv2

from Code.hough_transform.preprocessor import ImagePreprocessor


class TestImagePreprocessor:
    """Test suite for ImagePreprocessor class."""

    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance for testing."""
        return ImagePreprocessor(
            blur_kernel_size=(3, 3),
            canny_low_threshold=100,
            canny_high_threshold=200
        )

    @pytest.fixture
    def sample_bgr_image(self):
        """Create a simple BGR test image with a line."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.line(image, (10, 10), (90, 90), (255, 255, 255), 2)
        return image

    @pytest.fixture
    def blank_bgr_image(self):
        """Create a blank BGR image."""
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_initialization_with_defaults(self):
        """Test default initialization."""
        preprocessor = ImagePreprocessor()

        assert preprocessor.blur_kernel_size == (3, 3)
        assert preprocessor.blur_sigma == 1.0
        assert preprocessor.canny_low_threshold == 100
        assert preprocessor.canny_high_threshold == 200

    def test_initialization_with_custom_values(self):
        """Test custom initialization."""
        preprocessor = ImagePreprocessor(
            blur_kernel_size=(5, 5),
            blur_sigma=2.0,
            canny_low_threshold=50,
            canny_high_threshold=150
        )

        assert preprocessor.blur_kernel_size == (5, 5)
        assert preprocessor.blur_sigma == 2.0
        assert preprocessor.canny_low_threshold == 50
        assert preprocessor.canny_high_threshold == 150

    def test_invalid_kernel_size_raises_error(self):
        """Test that even kernel size raises ValueError."""
        with pytest.raises(ValueError, match="odd numbers"):
            ImagePreprocessor(blur_kernel_size=(4, 4))

    def test_invalid_canny_thresholds_raises_error(self):
        """Test that invalid Canny thresholds raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            ImagePreprocessor(canny_low_threshold=-1)

        with pytest.raises(ValueError, match="less than or equal"):
            ImagePreprocessor(canny_low_threshold=200, canny_high_threshold=100)

    def test_preprocess_returns_binary_image(self, preprocessor, sample_bgr_image):
        """Test that preprocessing returns a binary edge image."""
        result = preprocessor.preprocess(sample_bgr_image)

        assert result is not None
        assert len(result.shape) == 2  # Grayscale
        assert result.dtype == np.uint8

    def test_preprocess_detects_edges(self, preprocessor, sample_bgr_image):
        """Test that edges are detected in the image."""
        result = preprocessor.preprocess(sample_bgr_image)

        # Should have some edge pixels
        assert np.count_nonzero(result) > 0

    def test_preprocess_blank_image_has_no_edges(self, preprocessor, blank_bgr_image):
        """Test that blank image produces no edges."""
        result = preprocessor.preprocess(blank_bgr_image)

        assert np.count_nonzero(result) == 0

    def test_preprocess_none_image_raises_error(self, preprocessor):
        """Test that None input raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            preprocessor.preprocess(None)

    def test_preprocess_empty_image_raises_error(self, preprocessor):
        """Test that empty image raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            preprocessor.preprocess(np.array([]))

    def test_preprocess_grayscale_image_raises_error(self, preprocessor):
        """Test that grayscale input raises ValueError."""
        gray_image = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(ValueError, match="3-channel BGR"):
            preprocessor.preprocess(gray_image)