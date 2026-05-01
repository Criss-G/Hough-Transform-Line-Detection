"""
Hough Transform line detection module.

This module implements the core Hough Transform algorithm for detecting
straight lines in binary edge images.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from .models import HoughParameters, DetectedLine

logger = logging.getLogger(__name__)


class HoughTransformDetector:
    """Implements the Hough Transform algorithm for line detection.

    The Hough Transform converts points in image space to curves in
    parameter space (ρ, θ), allowing detection of collinear points.

    Mathematical Background:
        For each edge point (x, y), the line equation is:
        ρ = x·cos(θ) + y·sin(θ)

        Where:
        - ρ: perpendicular distance from origin to the line
        - θ: angle of the perpendicular with the x-axis

    Attributes:
        params: Configuration parameters for detection.

    Example:
        >>> from Code.hough_transform import HoughParameters, HoughTransformDetector
        >>> params = HoughParameters(vote_threshold=300)
        >>> detector = HoughTransformDetector(params)
        >>> lines = detector.detect_lines(edge_image)
        >>> print(f"Found {len(lines)} lines")
    """

    def __init__(self, params: Optional[HoughParameters] = None) -> None:
        """Initialize detector with parameters.

        Args:
            params: Hough Transform parameters. Uses defaults if None.
        """
        self.params = params or HoughParameters()
        self._accumulator: Optional[np.ndarray] = None
        self._thetas: Optional[np.ndarray] = None
        self._rhos: Optional[np.ndarray] = None
        self._image_center: Optional[Tuple[float, float]] = None

    def detect_lines(self, edge_image: np.ndarray) -> List[DetectedLine]:
        """Detect lines in a binary edge image using Hough Transform.

        Algorithm Steps:
        1. Calculate image diagonal for ρ range
        2. Create parameter space discretization
        3. Transform edge points to Hough space
        4. Accumulate votes using histogram
        5. Find peaks above threshold

        Args:
            edge_image: Binary image with edge pixels marked as non-zero.

        Returns:
            List of DetectedLine objects representing found lines,
            sorted by vote count in descending order.

        Raises:
            ValueError: If edge_image is None or empty.
        """
        self._validate_edge_image(edge_image)

        logger.info("Starting Hough Transform line detection...")

        # Setup parameter space
        height, width = edge_image.shape[:2]
        self._image_center = (height / 2, width / 2)

        self._setup_parameter_space(height, width)

        # Transform edge points to Hough space
        rho_values = self._transform_edge_points(edge_image)

        # Build accumulator
        self._build_accumulator(rho_values)

        # Find lines above threshold
        lines = self._extract_lines()

        # Sort by vote count (descending)
        lines.sort(key=lambda x: x.votes, reverse=True)

        logger.info(f"Detected {len(lines)} lines above threshold")
        return lines

    def _validate_edge_image(self, edge_image: np.ndarray) -> None:
        """Validate input edge image.

        Args:
            edge_image: Image to validate.

        Raises:
            ValueError: If image is invalid.
        """
        if edge_image is None:
            raise ValueError("Edge image cannot be None")
        if edge_image.size == 0:
            raise ValueError("Edge image cannot be empty")

    def _setup_parameter_space(self, height: int, width: int) -> None:
        """Setup the Hough parameter space discretization.

        Args:
            height: Image height in pixels.
            width: Image width in pixels.
        """
        # Calculate maximum possible distance (image diagonal)
        diagonal = np.sqrt(height ** 2 + width ** 2)

        # Create parameter space discretization
        self._thetas = np.linspace(
            0, 180, self.params.num_thetas, endpoint=False
        )
        self._rhos = np.linspace(
            -diagonal, diagonal, self.params.num_rhos
        )

        logger.debug(
            f"Parameter space: {self.params.num_thetas} thetas, "
            f"{self.params.num_rhos} rhos"
        )

    def _transform_edge_points(self, edge_image: np.ndarray) -> np.ndarray:
        """Transform edge points to Hough space.

        Args:
            edge_image: Binary edge image.

        Returns:
            Array of rho values for each edge point and theta.
        """
        center_y, center_x = self._image_center

        # Precompute trigonometric values for efficiency
        cos_thetas = np.cos(np.deg2rad(self._thetas))
        sin_thetas = np.sin(np.deg2rad(self._thetas))

        # Find edge points and center them
        edge_points = np.argwhere(edge_image != 0)
        edge_points_centered = edge_points - np.array([[center_y, center_x]])

        logger.debug(f"Processing {len(edge_points)} edge points")

        # Transform to Hough space using matrix multiplication
        rho_values = np.matmul(
            edge_points_centered,
            np.array([sin_thetas, cos_thetas])
        )

        return rho_values

    def _build_accumulator(self, rho_values: np.ndarray) -> None:
        """Build the Hough accumulator using histogram.

        Args:
            rho_values: Rho values for each edge point.
        """
        self._accumulator, _, _ = np.histogram2d(
            np.tile(self._thetas, rho_values.shape[0]),
            rho_values.ravel(),
            bins=[self._thetas, self._rhos]
        )
        self._accumulator = self._accumulator.T

        logger.debug(
            f"Accumulator built with shape {self._accumulator.shape}"
        )

    def _extract_lines(self) -> List[DetectedLine]:
        """Extract detected lines from accumulator.

        Returns:
            List of DetectedLine objects.
        """
        # Find cells above threshold
        detected_indices = np.argwhere(
            self._accumulator > self.params.vote_threshold
        )

        lines = []
        center_y, center_x = self._image_center

        for rho_idx, theta_idx in detected_indices:
            rho = self._rhos[rho_idx]
            theta = self._thetas[theta_idx]
            votes = int(self._accumulator[rho_idx, theta_idx])

            # Convert to Cartesian coordinates for visualization
            start_point, end_point = self._calculate_line_endpoints(
                rho, theta, center_x, center_y
            )

            lines.append(DetectedLine(
                rho=rho,
                theta=theta,
                votes=votes,
                start_point=start_point,
                end_point=end_point
            ))

        return lines

    def _calculate_line_endpoints(
            self,
            rho: float,
            theta: float,
            center_x: float,
            center_y: float
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Calculate line endpoints for visualization.

        Args:
            rho: Distance parameter.
            theta: Angle parameter in degrees.
            center_x: X coordinate of image center.
            center_y: Y coordinate of image center.

        Returns:
            Tuple of (start_point, end_point) as (x, y) coordinates.
        """
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))

        x0 = (a * rho) + center_x
        y0 = (b * rho) + center_y

        ext = self.params.line_extension

        x1 = int(x0 + ext * (-b))
        y1 = int(y0 + ext * a)
        x2 = int(x0 - ext * (-b))
        y2 = int(y0 - ext * a)

        return (x1, y1), (x2, y2)

    @property
    def accumulator(self) -> Optional[np.ndarray]:
        """Return the Hough accumulator array.

        Returns:
            2D numpy array representing votes in (ρ, θ) space,
            or None if detect_lines hasn't been called.
        """
        return self._accumulator

    @property
    def thetas(self) -> Optional[np.ndarray]:
        """Return the theta values used in detection.

        Returns:
            1D array of theta values in degrees, or None.
        """
        return self._thetas

    @property
    def rhos(self) -> Optional[np.ndarray]:
        """Return the rho values used in detection.

        Returns:
            1D array of rho values in pixels, or None.
        """
        return self._rhos

    def get_accumulator_peaks(self, n_peaks: int = 10) -> List[Tuple[float, float, int]]:
        """Get the top N peaks from the accumulator.

        Args:
            n_peaks: Number of peaks to return.

        Returns:
            List of (rho, theta, votes) tuples for top peaks.

        Raises:
            RuntimeError: If detect_lines hasn't been called yet.
        """
        if self._accumulator is None:
            raise RuntimeError(
                "Accumulator not initialized. Call detect_lines first."
            )

        # Flatten and get top indices
        flat_indices = np.argsort(self._accumulator.ravel())[::-1][:n_peaks]

        peaks = []
        for flat_idx in flat_indices:
            rho_idx, theta_idx = np.unravel_index(
                flat_idx, self._accumulator.shape
            )
            peaks.append((
                self._rhos[rho_idx],
                self._thetas[theta_idx],
                int(self._accumulator[rho_idx, theta_idx])
            ))

        return peaks