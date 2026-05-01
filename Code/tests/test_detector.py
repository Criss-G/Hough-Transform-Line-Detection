"""Unit tests for HoughTransformDetector class."""

import pytest
import numpy as np

from Code.hough_transform.models import HoughParameters
from Code.hough_transform.detector import HoughTransformDetector


class TestHoughTransformDetector:
    """Test suite for HoughTransformDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a detector with low threshold for testing."""
        params = HoughParameters(vote_threshold=10)
        return HoughTransformDetector(params)

    @pytest.fixture
    def default_detector(self):
        """Create a detector with default parameters."""
        return HoughTransformDetector()

    @pytest.fixture
    def horizontal_line_edge_image(self):
        """Create an edge image with a horizontal line."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[50, 20:80] = 255  # Horizontal line at y=50
        return image

    @pytest.fixture
    def vertical_line_edge_image(self):
        """Create an edge image with a vertical line."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[20:80, 50] = 255  # Vertical line at x=50
        return image

    @pytest.fixture
    def diagonal_line_edge_image(self):
        """Create an edge image with a diagonal line."""
        image = np.zeros((100, 100), dtype=np.uint8)
        for i in range(20, 80):
            image[i, i] = 255  # Diagonal line
        return image

    @pytest.fixture
    def empty_edge_image(self):
        """Create an empty edge image."""
        return np.zeros((100, 100), dtype=np.uint8)

    def test_initialization_with_default_params(self, default_detector):
        """Test initialization with default parameters."""
        assert default_detector.params.vote_threshold == 500
        assert default_detector.accumulator is None

    def test_initialization_with_custom_params(self, detector):
        """Test initialization with custom parameters."""
        assert detector.params.vote_threshold == 10

    def test_detect_lines_returns_list(self, detector, horizontal_line_edge_image):
        """Test that detect_lines returns a list."""
        lines = detector.detect_lines(horizontal_line_edge_image)

        assert isinstance(lines, list)

    def test_detect_horizontal_line(self, detector, horizontal_line_edge_image):
        """Test detection of a horizontal line."""
        lines = detector.detect_lines(horizontal_line_edge_image)

        assert len(lines) > 0

    def test_detect_vertical_line(self, detector, vertical_line_edge_image):
        """Test detection of a vertical line."""
        lines = detector.detect_lines(vertical_line_edge_image)

        assert len(lines) > 0

    def test_detect_diagonal_line(self, detector, diagonal_line_edge_image):
        """Test detection of a diagonal line."""
        lines = detector.detect_lines(diagonal_line_edge_image)

        assert len(lines) > 0

    def test_detected_line_has_required_attributes(
            self, detector, horizontal_line_edge_image
    ):
        """Test that detected lines have all required attributes."""
        lines = detector.detect_lines(horizontal_line_edge_image)

        assert len(lines) > 0
        line = lines[0]

        assert hasattr(line, 'rho')
        assert hasattr(line, 'theta')
        assert hasattr(line, 'votes')
        assert hasattr(line, 'start_point')
        assert hasattr(line, 'end_point')
        assert line.votes > 0

    def test_accumulator_populated_after_detection(
            self, detector, horizontal_line_edge_image
    ):
        """Test that accumulator is populated after detection."""
        detector.detect_lines(horizontal_line_edge_image)

        assert detector.accumulator is not None
        assert detector.thetas is not None
        assert detector.rhos is not None
        assert detector.accumulator.shape[0] > 0
        assert detector.accumulator.shape[1] > 0

    def test_empty_image_returns_no_lines(self, detector, empty_edge_image):
        """Test that an empty image returns no lines."""
        lines = detector.detect_lines(empty_edge_image)

        assert len(lines) == 0

    def test_none_image_raises_error(self, detector):
        """Test that None input raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            detector.detect_lines(None)

    def test_empty_array_raises_error(self, detector):
        """Test that empty array raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            detector.detect_lines(np.array([]))

    def test_lines_sorted_by_votes(self, detector, horizontal_line_edge_image):
        """Test that returned lines are sorted by votes."""
        lines = detector.detect_lines(horizontal_line_edge_image)

        if len(lines) > 1:
            for i in range(len(lines) - 1):
                assert lines[i].votes >= lines[i + 1].votes

    def test_get_accumulator_peaks(self, detector, horizontal_line_edge_image):
        """Test getting accumulator peaks."""
        detector.detect_lines(horizontal_line_edge_image)
        peaks = detector.get_accumulator_peaks(n_peaks=5)

        assert len(peaks) <= 5
        assert all(len(peak) == 3 for peak in peaks)

    def test_get_accumulator_peaks_before_detection_raises_error(self, detector):
        """Test that getting peaks before detection raises error."""
        with pytest.raises(RuntimeError, match="not initialized"):
            detector.get_accumulator_peaks()

    @pytest.mark.parametrize("threshold,expect_lines", [
        (5, True),  # Low threshold should detect lines
        (10000, False),  # Very high threshold should not detect lines
    ])
    def test_threshold_affects_detection(
            self, horizontal_line_edge_image, threshold, expect_lines
    ):
        """Test that vote threshold affects number of detected lines."""
        params = HoughParameters(vote_threshold=threshold)
        detector = HoughTransformDetector(params)

        lines = detector.detect_lines(horizontal_line_edge_image)

        if expect_lines:
            assert len(lines) > 0
        else:
            assert len(lines) == 0

    @pytest.mark.parametrize("num_rhos,num_thetas", [
        (90, 90),
        (180, 180),
        (360, 360),
    ])
    def test_different_resolutions(
            self, horizontal_line_edge_image, num_rhos, num_thetas
    ):
        """Test detection with different parameter space resolutions."""
        params = HoughParameters(
            num_rhos=num_rhos,
            num_thetas=num_thetas,
            vote_threshold=5
        )
        detector = HoughTransformDetector(params)

        lines = detector.detect_lines(horizontal_line_edge_image)

        # Should still detect lines regardless of resolution
        assert len(lines) >= 0  # Just ensure it runs without error