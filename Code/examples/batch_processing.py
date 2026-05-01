#!/usr/bin/env python3
"""Batch Processing Example"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import json
import csv
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass

import numpy as np
import cv2

from hough_transform import (
    HoughParameters,
    ImagePreprocessor,
    HoughTransformDetector,
)
from hough_transform.utils import setup_logging, create_synthetic_line_image

# Output directories
OUTPUT_DIR = Path(__file__).parent / "output" / "07_batch"
SAMPLE_DIR = OUTPUT_DIR / "sample_images"
RESULTS_DIR = OUTPUT_DIR / "results"


@dataclass
class BatchResult:
    """Store batch processing results."""
    name: str
    image_size: int
    num_edges: int
    num_lines: int
    preprocess_time: float
    detect_time: float
    total_time: float


class BatchProcessor:
    """Process multiple images with consistent settings."""

    def __init__(
            self,
            preprocessor: ImagePreprocessor,
            detector_params: HoughParameters,
    ):
        self.preprocessor = preprocessor
        self.detector_params = detector_params
        self.results: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def process_image(self, image: np.ndarray, name: str) -> Dict[str, Any]:
        """Process a single image."""
        start = datetime.now()

        edges = self.preprocessor.preprocess(image)
        edge_count = np.count_nonzero(edges)

        detector = HoughTransformDetector(self.detector_params)
        lines = detector.detect_lines(edges)

        processing_time = (datetime.now() - start).total_seconds()

        result = {
            'name': name,
            'image_size': f"{image.shape[1]}x{image.shape[0]}",
            'edge_pixels': edge_count,
            'lines_detected': len(lines),
            'processing_time_sec': round(processing_time, 4),
            'top_lines': [
                {'rho': round(l.rho, 2), 'theta': round(l.theta, 2), 'votes': l.votes}
                for l in lines[:5]
            ]
        }

        self.results.append(result)
        return result

    def process_directory(self, input_dir: Path) -> List[Dict[str, Any]]:
        """Process all images in a directory."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

        self.logger.info(f"Found {len(image_files)} images")

        for idx, img_file in enumerate(image_files, 1):
            self.logger.info(f"Processing [{idx}/{len(image_files)}]: {img_file.name}")
            image = cv2.imread(str(img_file))
            if image is not None:
                self.process_image(image, img_file.name)

        return self.results

    def save_json(self, filepath: Path):
        """Save results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"JSON saved to {filepath}")

    def save_csv(self, filepath: Path):
        """Save summary to CSV."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Size', 'Edges', 'Lines', 'Time (s)'])
            for r in self.results:
                writer.writerow([
                    r['name'], r['image_size'], r['edge_pixels'],
                    r['lines_detected'], r['processing_time_sec']
                ])
        self.logger.info(f"CSV saved to {filepath}")

    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {}

        lines = [r['lines_detected'] for r in self.results]
        times = [r['processing_time_sec'] for r in self.results]

        return {
            'total_images': len(self.results),
            'total_lines': sum(lines),
            'avg_lines': round(np.mean(lines), 2),
            'total_time': round(sum(times), 2),
            'avg_time': round(np.mean(times), 4),
        }


def create_sample_dataset(output_dir: Path, num_images: int = 10):
    """Create sample images for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        image = create_synthetic_line_image(
            width=300 + np.random.randint(-50, 50),
            height=300 + np.random.randint(-50, 50),
            num_lines=np.random.randint(2, 8),
            line_thickness=np.random.randint(1, 4)
        )
        cv2.imwrite(str(output_dir / f"sample_{i + 1:03d}.png"), image)


def main():
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create directories
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Batch Processing Example ===")

    # Create sample images
    logger.info("Creating sample dataset...")
    create_sample_dataset(SAMPLE_DIR, num_images=10)

    # Configure processor
    preprocessor = ImagePreprocessor(
        canny_low_threshold=50, canny_high_threshold=150
    )
    params = HoughParameters(vote_threshold=25)

    # Process images
    processor = BatchProcessor(preprocessor, params)
    processor.process_directory(SAMPLE_DIR)

    # Save results
    processor.save_json(RESULTS_DIR / "batch_results.json")
    processor.save_csv(RESULTS_DIR / "batch_results.csv")

    # Print summary
    summary = processor.get_summary()
    logger.info("\n=== Summary ===")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    logger.info(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()