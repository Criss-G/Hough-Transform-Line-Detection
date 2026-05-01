"""
Data models for Hough Transform line detection.

This module contains dataclasses that represent configuration parameters
and detection results used throughout the package.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class HoughParameters:
    """Configuration parameters for Hough Transform line detection.

    Attributes:
        num_rhos: Number of rho bins in the accumulator (resolution).
        num_thetas: Number of theta bins in the accumulator (resolution).
        vote_threshold: Minimum votes required to detect a line.
        line_extension: Length to extend detected lines for visualization.

    Example:
        >>> params = HoughParameters(vote_threshold=300)
        >>> print(params.num_rhos)
        180
    """
    num_rhos: int = 180
    num_thetas: int = 180
    vote_threshold: int = 500
    line_extension: int = 1000

    def __post_init__(self) -> None:
        """Validate parameters after initialization.

        Raises:
            ValueError: If resolution parameters are not positive integers.
            ValueError: If vote threshold is negative.
        """
        if self.num_rhos <= 0 or self.num_thetas <= 0:
            raise ValueError(
                "Resolution parameters (num_rhos, num_thetas) must be positive integers"
            )
        if self.vote_threshold < 0:
            raise ValueError("Vote threshold must be non-negative")
        if self.line_extension <= 0:
            raise ValueError("Line extension must be a positive integer")


@dataclass
class DetectedLine:
    """Represents a detected line in both Hough and Cartesian space.

    Attributes:
        rho: Distance from origin to the line (perpendicular distance).
        theta: Angle of the line in degrees (0-180).
        votes: Number of votes accumulated in the Hough accumulator.
        start_point: (x, y) start point for visualization.
        end_point: (x, y) end point for visualization.

    Example:
        >>> line = DetectedLine(
        ...     rho=100.0,
        ...     theta=45.0,
        ...     votes=250,
        ...     start_point=(0, 50),
        ...     end_point=(100, 150)
        ... )
        >>> print(f"Line at angle {line.theta}° with {line.votes} votes")
        Line at angle 45.0° with 250 votes
    """
    rho: float
    theta: float
    votes: int
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        return (
            f"DetectedLine(rho={self.rho:.2f}, theta={self.theta:.2f}°, "
            f"votes={self.votes})"
        )

    @property
    def length(self) -> float:
        """Calculate the Euclidean length of the line segment.

        Returns:
            Length of the line segment in pixels.
        """
        import math
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]
        return math.sqrt(dx ** 2 + dy ** 2)