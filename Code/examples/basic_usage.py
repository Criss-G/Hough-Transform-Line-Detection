#!/usr/bin/env python3
"""Basic Usage Example"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from hough_transform import (
    HoughParameters,
    ImagePreprocessor,
    HoughTransformDetector,
    HoughVisualizer,
)
from hough_transform.utils import setup_logging, create_synthetic_line_image

# Output directory for this example
OUTPUT_DIR = Path(__file__).parent / "output" / "basic_usage_results"


def main():
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Basic Hough Transform Example ===")

    # Create test image
    image = create_synthetic_line_image(
        width=300, height=300, num_lines=4, line_thickness=2
    )

    # Process
    preprocessor = ImagePreprocessor(
        canny_low_threshold=50, canny_high_threshold=150
    )
    edges = preprocessor.preprocess(image)

    params = HoughParameters(vote_threshold=30)
    detector = HoughTransformDetector(params)
    lines = detector.detect_lines(edges)

    logger.info(f"Detected {len(lines)} lines")

    # Visualize and save
    visualizer = HoughVisualizer()
    visualizer.visualize(
        original_image=image,
        edge_image=edges,
        detector=detector,
        lines=lines,
        save_path=OUTPUT_DIR / "detection_result.png",
        show_plot=True
    )

    logger.info(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()