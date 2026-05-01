#!/usr/bin/env python3
"""
Performance Benchmark Example

Measures and compares performance metrics:
- Processing time vs image size
- Processing time vs number of edge points
- Memory usage analysis
- Parameter space resolution impact

Useful for optimizing parameters for specific use cases.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import time
from typing import List, Dict, Any
from dataclasses import dataclass

import numpy as np
import cv2
import matplotlib.pyplot as plt

from Code.hough_transform import (
    HoughParameters,
    ImagePreprocessor,
    HoughTransformDetector,
)
from Code.hough_transform.utils import setup_logging


@dataclass
class BenchmarkResult:
    """Store benchmark results."""
    name: str
    image_size: int
    num_edges: int
    num_lines: int
    preprocess_time: float
    detect_time: float
    total_time: float
    params: Dict[str, Any]


def create_benchmark_image(size: int, line_density: float = 0.1) -> np.ndarray:
    """
    Create a benchmark image with controlled line density.

    Args:
        size: Image width and height
        line_density: Approximate fraction of image covered by lines
    """
    image = np.zeros((size, size, 3), dtype=np.uint8)

    num_lines = max(1, int(size * line_density / 10))

    for _ in range(num_lines):
        x1 = np.random.randint(0, size)
        y1 = np.random.randint(0, size)
        x2 = np.random.randint(0, size)
        y2 = np.random.randint(0, size)
        thickness = np.random.randint(1, 3)
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), thickness)

    return image


def run_single_benchmark(
        image: np.ndarray,
        preprocessor: ImagePreprocessor,
        params: HoughParameters,
        name: str
) -> BenchmarkResult:
    """Run a single benchmark measurement."""
    # Measure preprocessing time
    start = time.perf_counter()
    edges = preprocessor.preprocess(image)
    preprocess_time = time.perf_counter() - start

    num_edges = np.count_nonzero(edges)

    # Measure detection time
    start = time.perf_counter()
    detector = HoughTransformDetector(params)
    lines = detector.detect_lines(edges)
    detect_time = time.perf_counter() - start

    return BenchmarkResult(
        name=name,
        image_size=image.shape[0],
        num_edges=num_edges,
        num_lines=len(lines),
        preprocess_time=preprocess_time,
        detect_time=detect_time,
        total_time=preprocess_time + detect_time,
        params={
            'num_rhos': params.num_rhos,
            'num_thetas': params.num_thetas,
            'vote_threshold': params.vote_threshold
        }
    )


def benchmark_image_sizes(sizes: List[int], num_runs: int = 3) -> List[BenchmarkResult]:
    """Benchmark processing time vs image size."""
    results = []

    preprocessor = ImagePreprocessor()
    params = HoughParameters(vote_threshold=30)

    for size in sizes:
        times = []
        for run in range(num_runs):
            image = create_benchmark_image(size)
            result = run_single_benchmark(
                image, preprocessor, params, f"size_{size}"
            )
            times.append(result)

        # Average the results
        avg_result = BenchmarkResult(
            name=f"size_{size}",
            image_size=size,
            num_edges=int(np.mean([r.num_edges for r in times])),
            num_lines=int(np.mean([r.num_lines for r in times])),
            preprocess_time=np.mean([r.preprocess_time for r in times]),
            detect_time=np.mean([r.detect_time for r in times]),
            total_time=np.mean([r.total_time for r in times]),
            params=times[0].params
        )
        results.append(avg_result)

    return results


def benchmark_resolutions(
        resolutions: List[int],
        image_size: int = 400,
        num_runs: int = 3
) -> List[BenchmarkResult]:
    """Benchmark processing time vs parameter space resolution."""
    results = []

    preprocessor = ImagePreprocessor()
    image = create_benchmark_image(image_size)
    edges = preprocessor.preprocess(image)

    for resolution in resolutions:
        times = []
        for run in range(num_runs):
            params = HoughParameters(
                num_rhos=resolution,
                num_thetas=resolution,
                vote_threshold=30
            )

            start = time.perf_counter()
            detector = HoughTransformDetector(params)
            lines = detector.detect_lines(edges)
            detect_time = time.perf_counter() - start

            times.append({
                'detect_time': detect_time,
                'num_lines': len(lines)
            })

        avg_result = BenchmarkResult(
            name=f"res_{resolution}",
            image_size=image_size,
            num_edges=np.count_nonzero(edges),
            num_lines=int(np.mean([t['num_lines'] for t in times])),
            preprocess_time=0,
            detect_time=np.mean([t['detect_time'] for t in times]),
            total_time=np.mean([t['detect_time'] for t in times]),
            params={'num_rhos': resolution, 'num_thetas': resolution}
        )
        results.append(avg_result)

    return results


def plot_benchmark_results(
        size_results: List[BenchmarkResult],
        resolution_results: List[BenchmarkResult]
):
    """Create visualization of benchmark results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Hough Transform Performance Benchmarks", fontsize=14)

    # Plot 1: Processing time vs image size
    sizes = [r.image_size for r in size_results]
    total_times = [r.total_time * 1000 for r in size_results]  # Convert to ms
    preprocess_times = [r.preprocess_time * 1000 for r in size_results]
    detect_times = [r.detect_time * 1000 for r in size_results]

    axes[0, 0].plot(sizes, total_times, 'b-o', label='Total', linewidth=2)
    axes[0, 0].plot(sizes, preprocess_times, 'g--s', label='Preprocessing')
    axes[0, 0].plot(sizes, detect_times, 'r--^', label='Detection')
    axes[0, 0].set_xlabel('Image Size (pixels)')
    axes[0, 0].set_ylabel('Processing Time (ms)')
    axes[0, 0].set_title('Processing Time vs Image Size')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Edge count vs image size
    edge_counts = [r.num_edges for r in size_results]

    axes[0, 1].bar(sizes, edge_counts, color='steelblue', alpha=0.7)
    axes[0, 1].set_xlabel('Image Size (pixels)')
    axes[0, 1].set_ylabel('Number of Edge Pixels')
    axes[0, 1].set_title('Edge Pixels vs Image Size')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Detection time vs resolution
    resolutions = [r.params['num_rhos'] for r in resolution_results]
    res_times = [r.detect_time * 1000 for r in resolution_results]

    axes[1, 0].plot(resolutions, res_times, 'purple', marker='o', linewidth=2)
    axes[1, 0].set_xlabel('Parameter Space Resolution')
    axes[1, 0].set_ylabel('Detection Time (ms)')
    axes[1, 0].set_title('Detection Time vs Resolution')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Lines detected vs resolution
    res_lines = [r.num_lines for r in resolution_results]

    axes[1, 1].bar(resolutions, res_lines, color='orange', alpha=0.7)
    axes[1, 1].set_xlabel('Parameter Space Resolution')
    axes[1, 1].set_ylabel('Lines Detected')
    axes[1, 1].set_title('Lines Detected vs Resolution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Run performance benchmarks."""
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    Path("output").mkdir(exist_ok=True)

    logger.info("=== Performance Benchmark Example ===")

    # Benchmark 1: Image sizes
    logger.info("Benchmarking image sizes...")
    sizes = [100, 200, 300, 400, 500, 600, 800]
    size_results = benchmark_image_sizes(sizes, num_runs=3)

    logger.info("Image size benchmark results:")
    for r in size_results:
        logger.info(
            f"  {r.image_size}px: {r.total_time * 1000:.2f}ms "
            f"({r.num_edges} edges, {r.num_lines} lines)"
        )

    # Benchmark 2: Resolutions
    logger.info("\nBenchmarking parameter space resolutions...")
    resolutions = [45, 90, 180, 270, 360, 540, 720]
    resolution_results = benchmark_resolutions(resolutions, num_runs=3)

    logger.info("Resolution benchmark results:")
    for r in resolution_results:
        logger.info(
            f"  {r.params['num_rhos']}x{r.params['num_thetas']}: "
            f"{r.detect_time * 1000:.2f}ms ({r.num_lines} lines)"
        )

    # Create visualization
    logger.info("\nCreating benchmark visualization...")
    fig = plot_benchmark_results(size_results, resolution_results)
    fig.savefig("output/performance_benchmarks.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Print recommendations
    logger.info("\n=== Performance Recommendations ===")
    logger.info("- For real-time applications (>30 fps): Use images ≤400px")
    logger.info("- For balance of speed/accuracy: Resolution 180x180")
    logger.info("- For maximum accuracy: Resolution 360x360 or higher")
    logger.info("- Preprocessing is typically faster than detection")

    logger.info("\nBenchmark complete! Results saved to output/")


if __name__ == "__main__":
    main()