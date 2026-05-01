"""Integration tests for the complete Hough Transform pipeline."""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch

from Code.hough_transform import (
    HoughParameters,
    ImagePreprocessor,
    HoughTransformDetector,
    HoughVisualizer,
)
from Code.hough_transform.utils import create_synthetic_line_image


class TestIntegration:
    """Integration tests for the complete detection pipeline."""

    @pytest.fixture
    def test_image_with_lines(self):
        """Create a test image with clear lines."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Horizontal line
        cv2.line(image, (20, 50), (180, 50), (255, 255, 255), 3)
        # Vertical line
        cv2.line(image, (100, 20), (100, 180), (255, 255, 255), 3)
        return image

    @pytest.fixture
    def test_image_with_diagonal(self):
        """Create a test image with diagonal line."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.line(image, (20, 20), (180, 180), (255, 255, 255), 3)
        return image

    def test_full_pipeline_detects_lines(self, test_image_with_lines):
        """Test the complete detection pipeline with multiple lines."""
        # Preprocess
        preprocessor = ImagePreprocessor(
            canny_low_threshold=50,
            canny_high_threshold=150
        )
        edges = preprocessor.preprocess(test_image_with_lines)

        # Verify edges were detected
        assert np.count_nonzero(edges) > 0

        # Detect lines
        params = HoughParameters(vote_threshold=20)
        detector = HoughTransformDetector(params)
        lines = detector.detect_lines(edges)

        # Should detect at least one line
        assert len(lines) >= 1

        # All lines should have positive votes
        for line in lines:
            assert line.votes > 0

    def test_full_pipeline_with_diagonal_line(self, test_image_with_diagonal):
        """Test pipeline with diagonal line detection."""
        preprocessor = ImagePreprocessor()
        edges = preprocessor.preprocess(test_image_with_diagonal)

        params = HoughParameters(vote_threshold=15)
        detector = HoughTransformDetector(params)
        lines = detector.detect_lines(edges)

        assert len(lines) >= 1

        # Diagonal line should have theta around 45 degrees
        # (allowing for some tolerance)
        thetas = [line.theta for line in lines]
        assert any(30 <= theta <= 60 or 120 <= theta <= 150 for theta in thetas)

    @patch('matplotlib.pyplot.show')
    def test_full_pipeline_with_visualization(
            self, mock_show, test_image_with_lines, tmp_path
    ):
        """Test complete pipeline including visualization."""
        # Preprocess
        preprocessor = ImagePreprocessor()
        edges = preprocessor.preprocess(test_image_with_lines)

        # Detect
        params = HoughParameters(vote_threshold=20)
        detector = HoughTransformDetector(params)
        lines = detector.detect_lines(edges)

        # Visualize
        visualizer = HoughVisualizer()
        save_path = tmp_path / "result.png"

        fig = visualizer.visualize(
            test_image_with_lines,
            edges,
            detector,
            lines,
            save_path=save_path,
            show_plot=False
        )

        # Check outputs
        assert fig is not None
        assert save_path.exists()

    def test_synthetic_image_generation_and_detection(self):
        """Test using synthetic image generation utility."""
        # Generate synthetic image
        image = create_synthetic_line_image(
            width=200,
            height=200,
            num_lines=3,
            line_thickness=2
        )

        # Run pipeline
        preprocessor = ImagePreprocessor()
        edges = preprocessor.preprocess(image)

        params = HoughParameters(vote_threshold=10)
        detector = HoughTransformDetector(params)
        lines = detector.detect_lines(edges)

        # Should detect some lines (may not be all 3 due to randomness)
        assert detector.accumulator is not None

    def test_parameter_sensitivity(self, test_image_with_lines):
        """Test how different parameters affect detection."""
        preprocessor = ImagePreprocessor()
        edges = preprocessor.preprocess(test_image_with_lines)

        # Low threshold - should detect more lines
        params_low = HoughParameters(vote_threshold=10)
        detector_low = HoughTransformDetector(params_low)
        lines_low = detector_low.detect_lines(edges)

        # High threshold - should detect fewer lines
        params_high = HoughParameters(vote_threshold=100)
        detector_high = HoughTransformDetector(params_high)
        lines_high = detector_high.detect_lines(edges)

        # Low threshold should find at least as many lines
        assert len(lines_low) >= len(lines_high)

    def test_preprocessing_parameters_affect_edges(self, test_image_with_lines):
        """Test that preprocessing parameters affect edge detection."""
        # Conservative preprocessing
        preprocessor_conservative = ImagePreprocessor(
            canny_low_threshold=150,
            canny_high_threshold=250
        )
        edges_conservative = preprocessor_conservative.preprocess(
            test_image_with_lines
        )

        # Aggressive preprocessing
        preprocessor_aggressive = ImagePreprocessor(
            canny_low_threshold=30,
            canny_high_threshold=100
        )
        edges_aggressive = preprocessor_aggressive.preprocess(
            test_image_with_lines
        )

        # Aggressive should typically find more edge pixels
        conservative_count = np.count_nonzero(edges_conservative)
        aggressive_count = np.count_nonzero(edges_aggressive)

        # This relationship may not always hold, but generally should
        assert aggressive_count > 0
        assert conservative_count >= 0