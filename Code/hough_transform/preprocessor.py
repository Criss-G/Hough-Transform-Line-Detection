"""
Image preprocessing module for edge detection.

This module provides the ImagePreprocessor class which handles all
preprocessing steps required before applying the Hough Transform.
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing for edge detection.

    This class encapsulates all preprocessing steps including grayscale
    conversion, Gaussian blur, and Canny edge detection.

    Attributes:
        blur_kernel_size: Size of the Gaussian blur kernel.
        blur_sigma: Standard deviation for Gaussian blur.
        canny_low_threshold: Lower threshold for Canny edge detection.
        canny_high_threshold: Upper threshold for Canny edge detection.

    Example:
        >>> preprocessor = ImagePreprocessor(
        ...     canny_low_threshold=50,
        ...     canny_high_threshold=150
        ... )
        >>> edges = preprocessor.preprocess(image)
    """

    def __init__(
            self,
            blur_kernel_size: Tuple[int, int] = (3, 3),
            blur_sigma: float = 1.0,
            canny_low_threshold: int = 100,
            canny_high_threshold: int = 200
    ) -> None:
        """Initialize preprocessor with configuration.

        Args:
            blur_kernel_size: Gaussian blur kernel size (must be odd numbers).
            blur_sigma: Gaussian blur sigma value.
            canny_low_threshold: Lower threshold for Canny edge detection.
            canny_high_threshold: Upper threshold for Canny edge detection.

        Raises:
            ValueError: If kernel size values are not positive odd integers.
            ValueError: If thresholds are negative.
        """
        self._validate_parameters(
            blur_kernel_size,
            canny_low_threshold,
            canny_high_threshold
        )

        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold

    def _validate_parameters(
            self,
            blur_kernel_size: Tuple[int, int],
            canny_low: int,
            canny_high: int
    ) -> None:
        """Validate initialization parameters.

        Args:
            blur_kernel_size: Kernel size to validate.
            canny_low: Lower Canny threshold.
            canny_high: Upper Canny threshold.

        Raises:
            ValueError: If parameters are invalid.
        """
        if blur_kernel_size[0] % 2 == 0 or blur_kernel_size[1] % 2 == 0:
            raise ValueError("Blur kernel size must be odd numbers")
        if canny_low < 0 or canny_high < 0:
            raise ValueError("Canny thresholds must be non-negative")
        if canny_low > canny_high:
            raise ValueError(
                "Canny low threshold must be less than or equal to high threshold"
            )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline to input image.

        The preprocessing pipeline consists of:
        1. Grayscale conversion
        2. Gaussian blur for noise reduction
        3. Canny edge detection

        Args:
            image: Input BGR image as numpy array.

        Returns:
            Binary edge image where edge pixels are 255 and others are 0.

        Raises:
            ValueError: If input image is None or empty.
            ValueError: If image doesn't have 3 channels (BGR).
        """
        self._validate_image(image)

        logger.info("Starting image preprocessing pipeline...")

        # Step 1: Convert to grayscale
        gray = self._convert_to_grayscale(image)

        # Step 2: Apply Gaussian blur
        blurred = self._apply_blur(gray)

        # Step 3: Detect edges
        edges = self._detect_edges(blurred)

        edge_count = np.count_nonzero(edges)
        logger.info(f"Preprocessing complete. Found {edge_count} edge pixels")

        return edges

    def _validate_image(self, image: np.ndarray) -> None:
        """Validate input image.

        Args:
            image: Image to validate.

        Raises:
            ValueError: If image is invalid.
        """
        if image is None:
            raise ValueError("Input image cannot be None")
        if image.size == 0:
            raise ValueError("Input image cannot be empty")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel BGR image")

    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale.

        Args:
            image: BGR image.

        Returns:
            Grayscale image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.debug("Converted image to grayscale")
        return gray

    def _apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to reduce noise.

        Args:
            image: Grayscale image.

        Returns:
            Blurred image.
        """
        blurred = cv2.GaussianBlur(
            image,
            self.blur_kernel_size,
            self.blur_sigma
        )
        logger.debug(
            f"Applied Gaussian blur with kernel {self.blur_kernel_size}"
        )
        return blurred

    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection.

        Args:
            image: Blurred grayscale image.

        Returns:
            Binary edge image.
        """
        edges = cv2.Canny(
            image,
            self.canny_low_threshold,
            self.canny_high_threshold
        )
        logger.debug(
            f"Applied Canny edge detection "
            f"(thresholds: {self.canny_low_threshold}, {self.canny_high_threshold})"
        )
        return edges