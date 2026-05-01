#!/usr/bin/env python3
"""
Real-World Simulation Example

Simulates real-world scenarios where line detection is commonly used:
- Road lane detection
- Document scanning (paper edges)
- Building/architecture analysis
- Barcode detection

These synthetic examples demonstrate practical applications.
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
)
from Code.hough_transform.utils import setup_logging


def create_road_scene(width: int = 400, height: int = 300) -> np.ndarray:
    """
    Create a simulated road scene with lane markings.

    This simulates a bird's-eye or dash-cam view of a road.
    """
    # Create gray road background
    image = np.ones((height, width, 3), dtype=np.uint8) * 80

    # Add road texture (slight variation)
    noise = np.random.randint(-10, 10, (height, width), dtype=np.int16)
    for c in range(3):
        image[:, :, c] = np.clip(image[:, :, c].astype(np.int16) + noise, 0, 255)

    # Left lane marking (dashed)
    for y in range(0, height, 40):
        cv2.line(image, (120, y), (150, y + 20), (255, 255, 255), 3)

    # Right lane marking (dashed)
    for y in range(0, height, 40):
        cv2.line(image, (250, y), (280, y + 20), (255, 255, 255), 3)

    # Road edges (solid lines)
    cv2.line(image, (50, 0), (80, height), (200, 200, 200), 2)
    cv2.line(image, (320, 0), (350, height), (200, 200, 200), 2)

    return image.astype(np.uint8)


def create_document_scene(width: int = 400, height: int = 300) -> np.ndarray:
    """
    Create a simulated document on a desk (for document scanning).

    Demonstrates detection of document edges for perspective correction.
    """
    # Create wooden desk texture
    image = np.ones((height, width, 3), dtype=np.uint8) * np.array([60, 80, 120])

    # Add wood grain effect
    for i in range(0, width, 30):
        shade = np.random.randint(-20, 20)
        cv2.line(image, (i, 0), (i + 10, height),
                 (60 + shade, 80 + shade, 120 + shade), 2)

    # Draw white paper (slightly rotated rectangle)
    paper_pts = np.array([
        [80, 50],
        [320, 30],
        [340, 250],
        [100, 270]
    ], np.int32)

    cv2.fillPoly(image, [paper_pts], (250, 250, 250))
    cv2.polylines(image, [paper_pts], True, (200, 200, 200), 2)

    # Add some text lines on paper (gray)
    for y in range(80, 230, 25):
        cv2.line(image, (110, y), (300, y - 5), (150, 150, 150), 1)

    return image.astype(np.uint8)


def create_building_scene(width: int = 400, height: int = 300) -> np.ndarray:
    """
    Create a simulated building facade with windows.

    Demonstrates architectural line detection.
    """
    # Sky gradient
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        blue = int(200 - y * 0.3)
        image[y, :] = [blue, blue + 30, 255]

    # Building facade
    cv2.rectangle(image, (50, 80), (350, height), (180, 180, 190), -1)

    # Building edges
    cv2.line(image, (50, 80), (50, height), (100, 100, 110), 3)
    cv2.line(image, (350, 80), (350, height), (100, 100, 110), 3)
    cv2.line(image, (50, 80), (350, 80), (100, 100, 110), 3)

    # Windows (grid pattern)
    for row in range(3):
        for col in range(4):
            x = 80 + col * 70
            y = 100 + row * 60
            cv2.rectangle(image, (x, y), (x + 40, y + 40), (100, 150, 200), -1)
            cv2.rectangle(image, (x, y), (x + 40, y + 40), (60, 60, 70), 2)
            # Window dividers
            cv2.line(image, (x + 20, y), (x + 20, y + 40), (60, 60, 70), 1)
            cv2.line(image, (x, y + 20), (x + 40, y + 20), (60, 60, 70), 1)

    return image.astype(np.uint8)


def create_barcode_scene(width: int = 400, height: int = 200) -> np.ndarray:
    """
    Create a simulated barcode.

    Demonstrates detection of parallel vertical lines.
    """
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Generate random barcode pattern
    np.random.seed(42)  # For reproducibility
    x = 50

    while x < width - 50:
        bar_width = np.random.choice([2, 3, 4, 5])
        space_width = np.random.choice([2, 3, 4, 5])

        # Draw bar
        cv2.rectangle(image, (x, 50), (x + bar_width, height - 50), (0, 0, 0), -1)
        x += bar_width + space_width

    return image


def create_parking_lot_scene(width: int = 400, height: int = 300) -> np.ndarray:
    """
    Create a simulated parking lot with parking lines.

    Demonstrates detection of angled parallel lines.
    """
    # Asphalt background
    image = np.ones((height, width, 3), dtype=np.uint8) * 60
    noise = np.random.randint(-10, 10, (height, width), dtype=np.int16)
    for c in range(3):
        image[:, :, c] = np.clip(image[:, :, c].astype(np.int16) + noise, 0, 255)

    # Parking space lines (angled)
    for i in range(8):
        x_start = 30 + i * 50
        cv2.line(image, (x_start, 50), (x_start + 30, height - 50),
                 (255, 255, 255), 2)

    # Horizontal boundary lines
    cv2.line(image, (20, 50), (width - 20, 50), (255, 255, 0), 3)
    cv2.line(image, (20, height - 50), (width - 20, height - 50),
             (255, 255, 0), 3)

    return image.astype(np.uint8)


def process_scene(image: np.ndarray, name: str, threshold: int) -> dict:
    """Process a scene and return detection results."""
    preprocessor = ImagePreprocessor(
        canny_low_threshold=50,
        canny_high_threshold=150
    )
    edges = preprocessor.preprocess(image)

    params = HoughParameters(vote_threshold=threshold)
    detector = HoughTransformDetector(params)
    lines = detector.detect_lines(edges)

    return {
        'name': name,
        'image': image,
        'edges': edges,
        'lines': lines,
        'detector': detector
    }


def main():
    """Run real-world simulation examples."""
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    Path("output").mkdir(exist_ok=True)

    logger.info("=== Real-World Simulation Example ===")

    # Define scenes
    scenes = [
        ("Road Lanes", create_road_scene(), 25),
        ("Document", create_document_scene(), 40),
        ("Building", create_building_scene(), 30),
        ("Barcode", create_barcode_scene(), 15),
        ("Parking Lot", create_parking_lot_scene(), 20),
    ]

    # Process all scenes
    results = []
    for name, image, threshold in scenes:
        logger.info(f"Processing: {name}")
        result = process_scene(image, name, threshold)
        results.append(result)
        logger.info(f"  Found {len(result['lines'])} lines")

    # Create visualization
    fig, axes = plt.subplots(3, len(results), figsize=(20, 12))
    fig.suptitle("Real-World Line Detection Applications", fontsize=16)

    for idx, result in enumerate(results):
        image_rgb = cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB)

        # Row 1: Original scene
        axes[0, idx].imshow(image_rgb)
        axes[0, idx].set_title(f"{result['name']}\n(Original)")
        axes[0, idx].axis('off')

        # Row 2: Edge detection
        axes[1, idx].imshow(result['edges'], cmap='gray')
        axes[1, idx].set_title("Edge Detection")
        axes[1, idx].axis('off')

        # Row 3: Detected lines overlay
        axes[2, idx].imshow(image_rgb)
        for line in result['lines'][:15]:
            x1, y1 = line.start_point
            x2, y2 = line.end_point
            axes[2, idx].plot([x1, x2], [y1, y2], 'lime', linewidth=2)
        axes[2, idx].set_title(f"Detected: {len(result['lines'])} lines")
        axes[2, idx].axis('off')

    plt.tight_layout()
    plt.savefig("output/real_world_simulation.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary
    logger.info("\n=== Detection Summary ===")
    for result in results:
        logger.info(f"{result['name']}: {len(result['lines'])} lines detected")

    logger.info("\nDone! Results saved to output/real_world_simulation.png")


if __name__ == "__main__":
    main()