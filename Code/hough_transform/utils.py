"""
Utility functions for Hough Transform line detection.

This module provides helper functions for common operations
like image loading and saving.
"""

import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk with error handling.

    Args:
        path: Path to the image file.

    Returns:
        Loaded image as BGR numpy array.

    Raises:
        FileNotFoundError: If image file doesn't exist.
        ValueError: If image cannot be read (corrupted or unsupported format).

    Example:
        >>> image = load_image("path/to/image.jpg")
        >>> print(f"Loaded image with shape {image.shape}")
    """
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read image: {path}. "
            "File may be corrupted or in an unsupported format."
        )

    logger.info(
        f"Loaded image: {path} "
        f"(size: {image.shape[1]}x{image.shape[0]}, "
        f"channels: {image.shape[2]})"
    )
    return image


def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """Save an image to disk.

    Args:
        image: Image to save (BGR format for color images).
        path: Path where the image will be saved.

    Raises:
        ValueError: If image is invalid.
        IOError: If image cannot be saved.

    Example:
        >>> save_image(processed_image, "output/result.png")
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot save None or empty image")

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(save_path), image)

    if not success:
        raise IOError(f"Failed to save image to: {path}")

    logger.info(f"Saved image to: {path}")


def create_synthetic_line_image(
        width: int = 200,
        height: int = 200,
        num_lines: int = 2,
        line_thickness: int = 2
) -> np.ndarray:
    """Create a synthetic image with random lines for testing.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        num_lines: Number of lines to draw.
        line_thickness: Thickness of lines in pixels.

    Returns:
        BGR image with white lines on black background.

    Example:
        >>> test_image = create_synthetic_line_image(num_lines=3)
        >>> edges = preprocessor.preprocess(test_image)
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)

    for _ in range(num_lines):
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        x2 = np.random.randint(0, width)
        y2 = np.random.randint(0, height)

        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), line_thickness)

    return image


def setup_logging(
        level: int = logging.INFO,
        format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """Configure logging for the package.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        format_string: Format string for log messages.

    Example:
        >>> setup_logging(level=logging.DEBUG)
    """
    logging.basicConfig(level=level, format=format_string)
    logger.info(f"Logging configured at level {logging.getLevelName(level)}")