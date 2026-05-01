#!/usr/bin/env python3
"""Parameter Tuning Example"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt

from hough_transform import (
    HoughParameters,
    ImagePreprocessor,
    HoughTransformDetector,
)
from hough_transform.utils import setup_logging

# Output directory for this example
OUTPUT_DIR = Path(__file__).parent / "output" / "02_parameter_tuning"


def create_test_image_with_noise():
    """Create a test image with lines and some noise."""
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.line(image, (50, 50), (250, 50), (255, 255, 255), 2)
    cv2.line(image, (50, 50), (50, 250), (255, 255, 255), 2)
    cv2.line(image, (50, 250), (250, 50), (255, 255, 255), 2)
    noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    return image


def compare_vote_thresholds(image, edges):
    """Compare detection with different vote thresholds."""
    thresholds = [10, 30, 50, 100, 200]

    fig, axes = plt.subplots(1, len(thresholds), figsize=(20, 4))
    fig.suptitle("Effect of Vote Threshold on Line Detection", fontsize=14)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for ax, threshold in zip(axes, thresholds):
        params = HoughParameters(vote_threshold=threshold)
        detector = HoughTransformDetector(params)
        lines = detector.detect_lines(edges)

        ax.imshow(image_rgb)
        for line in lines:
            x1, y1 = line.start_point
            x2, y2 = line.end_point
            ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
        ax.set_title(f"Threshold: {threshold}\n({len(lines)} lines)")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_comparison.png", dpi=150)
    plt.show()


def compare_resolutions(image, edges):
    """Compare detection with different parameter space resolutions."""
    resolutions = [(45, 45), (90, 90), (180, 180), (360, 360)]

    fig, axes = plt.subplots(2, len(resolutions), figsize=(16, 8))
    fig.suptitle("Effect of Parameter Space Resolution", fontsize=14)

    for idx, (num_rhos, num_thetas) in enumerate(resolutions):
        params = HoughParameters(
            num_rhos=num_rhos, num_thetas=num_thetas, vote_threshold=20
        )
        detector = HoughTransformDetector(params)
        lines = detector.detect_lines(edges)

        # Accumulator
        axes[0, idx].imshow(detector.accumulator, cmap='hot', aspect='auto')
        axes[0, idx].set_title(f"Resolution: {num_rhos}x{num_thetas}")

        # Lines
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[1, idx].imshow(image_rgb)
        for line in lines[:10]:
            x1, y1 = line.start_point
            x2, y2 = line.end_point
            axes[1, idx].plot([x1, x2], [y1, y2], 'r-', linewidth=2)
        axes[1, idx].set_title(f"{len(lines)} lines")
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resolution_comparison.png", dpi=150)
    plt.show()


def compare_canny_thresholds(image):
    """Compare edge detection with different Canny thresholds."""
    threshold_pairs = [(30, 60), (50, 100), (100, 200), (150, 300)]

    fig, axes = plt.subplots(2, len(threshold_pairs), figsize=(16, 8))
    fig.suptitle("Effect of Canny Edge Detection Thresholds", fontsize=14)

    for idx, (low, high) in enumerate(threshold_pairs):
        preprocessor = ImagePreprocessor(
            canny_low_threshold=low, canny_high_threshold=high
        )
        edges = preprocessor.preprocess(image)

        axes[0, idx].imshow(edges, cmap='gray')
        axes[0, idx].set_title(f"Canny: ({low}, {high})")
        axes[0, idx].axis('off')

        params = HoughParameters(vote_threshold=30)
        detector = HoughTransformDetector(params)
        lines = detector.detect_lines(edges)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[1, idx].imshow(image_rgb)
        for line in lines[:10]:
            x1, y1 = line.start_point
            x2, y2 = line.end_point
            axes[1, idx].plot([x1, x2], [y1, y2], 'r-', linewidth=2)
        axes[1, idx].set_title(f"{len(lines)} lines")
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "canny_comparison.png", dpi=150)
    plt.show()


def main():
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Parameter Tuning Example ===")

    image = create_test_image_with_noise()
    preprocessor = ImagePreprocessor(
        canny_low_threshold=50, canny_high_threshold=150
    )
    edges = preprocessor.preprocess(image)

    logger.info("Comparing vote thresholds...")
    compare_vote_thresholds(image, edges)

    logger.info("Comparing resolutions...")
    compare_resolutions(image, edges)

    logger.info("Comparing Canny thresholds...")
    compare_canny_thresholds(image)

    logger.info(f"All results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()