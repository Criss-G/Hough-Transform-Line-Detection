"""Unit tests for HoughVisualizer class."""

import pytest
import numpy as np
import cv2
from unittest.mock import patch

from Code.hough_transform.visualizer import HoughVisualizer
from Code.hough_transform.models import DetectedLine, HoughParameters
from Code.hough_transform.detector import HoughTransformDetector


class TestHoughVisualizer:
    """Test suite for HoughVisualizer class."""

    @pytest.fixture
    def visualizer(self):
        """Create a visualizer instance."""
        return HoughVisualizer(figsize=(12, 3))

    @pytest.fixture
    def sample_bgr_image(self):
        """Create a sample BGR image."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.line(image, (10, 10), (90, 90), (255, 255, 255), 2)
        return image

    @pytest.fixture
    def sample_edge_image(self):
        """Create a sample edge image."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[50, 20:80] = 255
        return image

    @pytest.fixture
    def sample_lines(self):
        """Create sample detected lines."""
        return [
            DetectedLine(
                rho=50.0,
                theta=90.0,
                votes=100,
                start_point=(20, 50),
                end_point=(80, 50)
            ),
            DetectedLine(
                rho=0.0,
                theta=45.0,
                votes=80,
                start_point=(10, 10),
                end_point=(90, 90)
            )
        ]

    @pytest.fixture
    def mock_detector(self, sample_edge_image):
        """Create a mock detector with accumulator."""
        params = HoughParameters(vote_threshold=10)
        detector = HoughTransformDetector(params)
        detector.detect_lines(sample_edge_image)
        return detector

    def test_initialization_with_defaults(self):
        """Test default initialization."""
        visualizer = HoughVisualizer()

        assert visualizer.figsize == (16, 4)
        assert visualizer.line_color == "red"
        assert visualizer.line_width == 2.0

    def test_initialization_with_custom_values(self):
        """Test custom initialization."""
        visualizer = HoughVisualizer(
            figsize=(20, 5),
            line_color="blue",
            line_width=3.0
        )

        assert visualizer.figsize == (20, 5)
        assert visualizer.line_color == "blue"
        assert visualizer.line_width == 3.0

    @patch('matplotlib.pyplot.show')
    def test_visualize_returns_figure(
            self, mock_show, visualizer, sample_bgr_image,
            sample_edge_image, mock_detector, sample_lines
    ):
        """Test that visualize returns a matplotlib Figure."""
        fig = visualizer.visualize(
            sample_bgr_image,
            sample_edge_image,
            mock_detector,
            sample_lines,
            show_plot=False
        )

        assert fig is not None
        # Figure should have 4 subplots
        assert len(fig.axes) == 4

    @patch('matplotlib.pyplot.show')
    def test_visualize_with_empty_lines(
            self, mock_show, visualizer, sample_bgr_image,
            sample_edge_image, mock_detector
    ):
        """Test visualization with no detected lines."""
        fig = visualizer.visualize(
            sample_bgr_image,
            sample_edge_image,
            mock_detector,
            [],  # Empty lines list
            show_plot=False
        )

        assert fig is not None

    @patch('matplotlib.pyplot.show')
    def test_visualize_saves_to_file(
            self, mock_show, visualizer, sample_bgr_image,
            sample_edge_image, mock_detector, sample_lines, tmp_path
    ):
        """Test that visualization can be saved to file."""
        save_path = tmp_path / "output" / "test_viz.png"

        visualizer.visualize(
            sample_bgr_image,
            sample_edge_image,
            mock_detector,
            sample_lines,
            save_path=save_path,
            show_plot=False
        )

        assert save_path.exists()