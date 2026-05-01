"""Unit tests for data models."""

import pytest

from Code.hough_transform.models import HoughParameters, DetectedLine


class TestHoughParameters:
    """Test suite for HoughParameters dataclass."""

    def test_default_parameters(self):
        """Test that default parameters are set correctly."""
        params = HoughParameters()

        assert params.num_rhos == 180
        assert params.num_thetas == 180
        assert params.vote_threshold == 500
        assert params.line_extension == 1000

    def test_custom_parameters(self):
        """Test creating parameters with custom values."""
        params = HoughParameters(
            num_rhos=360,
            num_thetas=360,
            vote_threshold=300,
            line_extension=500
        )

        assert params.num_rhos == 360
        assert params.num_thetas == 360
        assert params.vote_threshold == 300
        assert params.line_extension == 500

    def test_invalid_num_rhos_raises_error(self):
        """Test that invalid num_rhos raises ValueError."""
        with pytest.raises(ValueError, match="Resolution parameters"):
            HoughParameters(num_rhos=0)

        with pytest.raises(ValueError, match="Resolution parameters"):
            HoughParameters(num_rhos=-10)

    def test_invalid_num_thetas_raises_error(self):
        """Test that invalid num_thetas raises ValueError."""
        with pytest.raises(ValueError, match="Resolution parameters"):
            HoughParameters(num_thetas=0)

    def test_negative_threshold_raises_error(self):
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="Vote threshold"):
            HoughParameters(vote_threshold=-100)

    def test_invalid_line_extension_raises_error(self):
        """Test that invalid line extension raises ValueError."""
        with pytest.raises(ValueError, match="Line extension"):
            HoughParameters(line_extension=0)


class TestDetectedLine:
    """Test suite for DetectedLine dataclass."""

    def test_detected_line_creation(self):
        """Test creating a DetectedLine instance."""
        line = DetectedLine(
            rho=100.0,
            theta=45.0,
            votes=250,
            start_point=(0, 50),
            end_point=(100, 150)
        )

        assert line.rho == 100.0
        assert line.theta == 45.0
        assert line.votes == 250
        assert line.start_point == (0, 50)
        assert line.end_point == (100, 150)

    def test_detected_line_repr(self):
        """Test string representation of DetectedLine."""
        line = DetectedLine(
            rho=100.0,
            theta=45.0,
            votes=250,
            start_point=(0, 50),
            end_point=(100, 150)
        )

        repr_str = repr(line)
        assert "100.00" in repr_str
        assert "45.00" in repr_str
        assert "250" in repr_str

    def test_detected_line_length(self):
        """Test length property calculation."""
        line = DetectedLine(
            rho=0,
            theta=0,
            votes=100,
            start_point=(0, 0),
            end_point=(3, 4)  # 3-4-5 triangle
        )

        assert line.length == 5.0