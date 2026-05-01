#!/usr/bin/env python3
"""
Geometric Shapes Example

Demonstrates line detection on various geometric shapes including:
- Rectangles
- Triangles
- Polygons
- Grids

This example shows how the Hough Transform performs on structured images.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt

from Code.hough_transform import (
    HoughParameters,
    ImagePreprocessor,
    HoughTransformDetector,
    HoughVisualizer,
)
from Code.hough_transform.utils import setup_logging


def create_rectangle_image(size: int = 300) -> np.ndarray:
    """Create an image with a rectangle."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (250, 200), (255, 255, 255), 2)
    return image


def create_triangle_image(size: int = 300) -> np.ndarray:
    """Create an image with a triangle."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    pts = np.array([[150, 30], [50, 250], [250, 250]], np.int32)
    cv2.polylines(image, [pts], True, (255, 255, 255), 2)
    return image


def create_pentagon_image(size: int = 300) -> np.ndarray:
    """Create an image with a pentagon."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = 100

    pts = []
    for i in range(5):
        angle = np.deg2rad(90 + i * 72)  # Start from top
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] - radius * np.sin(angle))
        pts.append([x, y])

    pts = np.array(pts, np.int32)
    cv2.polylines(image, [pts], True, (255, 255, 255), 2)
    return image


def create_grid_image(size: int = 300, grid_size: int = 5) -> np.ndarray:
    """Create an image with a grid pattern."""
    image = np.zeros((size, size, 3), dtype=np.uint8)

    step = size // (grid_size + 1)

    # Vertical lines
    for i in range(1, grid_size + 1):
        x = i * step
        cv2.line(image, (x, step), (x, size - step), (255, 255, 255), 1)

    # Horizontal lines
    for i in range(1, grid_size + 1):
        y = i * step
        cv2.line(image, (step, y), (size - step, y), (255, 255, 255), 1)

    return image


def create_star_image(size: int = 300) -> np.ndarray:
    """Create an image with a star shape."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    outer_radius = 120
    inner_radius = 50

    pts = []
    for i in range(10):
        angle = np.deg2rad(90 + i * 36)
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] - radius * np.sin(angle))
        pts.append([x, y])

    pts = np.array(pts, np.int32)
    cv2.polylines(image, [pts], True, (255, 255, 255), 2)
    return image


def create_crosshatch_image(size: int = 300) -> np.ndarray:
    """Create an image with crosshatch pattern (diagonal lines)."""
    image = np.zeros((size, size, 3), dtype=np.uint8)

    # Diagonal lines (top-left to bottom-right)
    for i in range(-size, size * 2, 40):
        cv2.line(image, (i, 0), (i + size, size), (255, 255, 255), 1)

    # Diagonal lines (top-right to bottom-left)
    for i in range(-size, size * 2, 40):
        cv2.line(image, (size - i, 0), (-i, size), (255, 255, 255), 1)

    return image


def detect_and_visualize(
        image: np.ndarray,
        title: str,
        vote_threshold: int = 30
) -> tuple:
    """Detect lines in an image and return results."""
    preprocessor = ImagePreprocessor(
        canny_low_threshold=50,
        canny_high_threshold=150
    )
    edges = preprocessor.preprocess(image)

    params = HoughParameters(vote_threshold=vote_threshold)
    detector = HoughTransformDetector(params)
    lines = detector.detect_lines(edges)

    return edges, detector, lines


def main():
    """Run geometric shapes example."""
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    Path("output").mkdir(exist_ok=True)

    logger.info("=== Geometric Shapes Example ===")

    # Define shapes to test
    shapes = [
        ("Rectangle", create_rectangle_image(), 30),
        ("Triangle", create_triangle_image(), 30),
        ("Pentagon", create_pentagon_image(), 20),
        ("Grid", create_grid_image(), 15),
        ("Star", create_star_image(), 15),
        ("Crosshatch", create_crosshatch_image(), 20),
    ]

    fig, axes = plt.subplots(3, len(shapes), figsize=(20, 10))
    fig.suptitle("Line Detection on Geometric Shapes", fontsize=16)

    for idx, (name, image, threshold) in enumerate(shapes):
        logger.info(f"Processing {name}...")

        edges, detector, lines = detect_and_visualize(image, name, threshold)

        # Row 1: Original image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0, idx].imshow(image_rgb)
        axes[0, idx].set_title(f"{name}\n(Original)")
        axes[0, idx].axis('off')

        # Row 2: Edge detection
        axes[1, idx].imshow(edges, cmap='gray')
        axes[1, idx].set_title(f"Edges\n({np.count_nonzero(edges)} pixels)")
        axes[1, idx].axis('off')

        # Row 3: Detected lines
        axes[2, idx].imshow(image_rgb)
        for line in lines[:20]:  # Limit lines shown
            x1, y1 = line.start_point
            x2, y2 = line.end_point
            axes[2, idx].plot([x1, x2], [y1, y2], 'r-', linewidth=1.5)
        axes[2, idx].set_title(f"Detected Lines\n({len(lines)} lines)")
        axes[2, idx].axis('off')

        logger.info(f"  {name}: Found {len(lines)} lines")

    plt.tight_layout()
    plt.savefig("output/geometric_shapes.png", dpi=150, bbox_inches='tight')
    plt.show()

    logger.info("Done! Results saved to output/geometric_shapes.png")


if __name__ == "__main__":
    main()