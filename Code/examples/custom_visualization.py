#!/usr/bin/env python3
"""Custom Visualization Example"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from typing import List, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from hough_transform import (
    HoughParameters,
    ImagePreprocessor,
    HoughTransformDetector,
    DetectedLine,
)
from hough_transform.utils import setup_logging

# Output directory for this example
OUTPUT_DIR = Path(__file__).parent / "output" / "04_visualization"


def create_test_image():
    """Create test image with multiple line orientations."""
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.line(image, (50, 100), (350, 100), (255, 255, 255), 2)
    cv2.line(image, (200, 50), (200, 350), (255, 255, 255), 2)
    cv2.line(image, (50, 50), (350, 350), (255, 255, 255), 2)
    cv2.line(image, (350, 50), (50, 350), (255, 255, 255), 2)
    cv2.line(image, (100, 50), (300, 150), (255, 255, 255), 2)
    cv2.line(image, (50, 200), (150, 350), (255, 255, 255), 2)
    return image


def color_lines_by_angle(lines: List[DetectedLine]) -> List[Tuple[DetectedLine, str]]:
    """Assign colors based on angle."""
    cmap = plt.cm.hsv
    return [
        (line, mcolors.rgb2hex(cmap(line.theta / 180.0)[:3]))
        for line in lines
    ]


def color_lines_by_votes(lines: List[DetectedLine]) -> List[Tuple[DetectedLine, str]]:
    """Assign colors based on vote count."""
    if not lines:
        return []

    max_votes = max(line.votes for line in lines)
    min_votes = min(line.votes for line in lines)
    vote_range = max_votes - min_votes if max_votes != min_votes else 1

    cmap = plt.cm.coolwarm
    return [
        (line, mcolors.rgb2hex(cmap((line.votes - min_votes) / vote_range)[:3]))
        for line in lines
    ]


def visualize_by_angle(image, edges, lines):
    """Visualize lines colored by angle."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title("Edge Detection")
    axes[1].axis('off')

    axes[2].imshow(image_rgb)
    for line, color in color_lines_by_angle(lines):
        x1, y1 = line.start_point
        x2, y2 = line.end_point
        axes[2].plot([x1, x2], [y1, y2], color=color, linewidth=2)
    axes[2].set_title(f"Lines by Angle ({len(lines)} lines)")
    axes[2].axis('off')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=plt.Normalize(0, 180))
    plt.colorbar(sm, ax=axes[2], orientation='horizontal', pad=0.05, label='Angle (°)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lines_by_angle.png", dpi=150, bbox_inches='tight')
    plt.show()


def visualize_by_votes(image, edges, lines):
    """Visualize lines colored by votes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title("Edge Detection")
    axes[1].axis('off')

    axes[2].imshow(image_rgb)
    for line, color in color_lines_by_votes(lines):
        x1, y1 = line.start_point
        x2, y2 = line.end_point
        axes[2].plot([x1, x2], [y1, y2], color=color, linewidth=2)
    axes[2].set_title(f"Lines by Confidence ({len(lines)} lines)")
    axes[2].axis('off')

    if lines:
        min_v, max_v = min(l.votes for l in lines), max(l.votes for l in lines)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(min_v, max_v))
        plt.colorbar(sm, ax=axes[2], orientation='horizontal', pad=0.05, label='Votes')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lines_by_votes.png", dpi=150, bbox_inches='tight')
    plt.show()


def visualize_hough_curves(edges, detector, lines, num_curves=50):
    """Visualize Hough space curves."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    height, width = edges.shape
    center_y, center_x = height / 2, width / 2

    edge_points = np.argwhere(edges != 0)
    if len(edge_points) > num_curves:
        indices = np.random.choice(len(edge_points), num_curves, replace=False)
        sampled_points = edge_points[indices]
    else:
        sampled_points = edge_points

    # Sampled points
    axes[0].imshow(edges, cmap='gray')
    axes[0].scatter(sampled_points[:, 1], sampled_points[:, 0], c='red', s=10, alpha=0.7)
    axes[0].set_title(f"Sampled Edge Points ({len(sampled_points)})")
    axes[0].axis('off')

    # Hough curves
    axes[1].set_facecolor('black')
    thetas = np.linspace(0, 180, 180)
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(sampled_points)))

    for point, color in zip(sampled_points, colors):
        y, x = point
        rhos = (y - center_y) * sin_thetas + (x - center_x) * cos_thetas
        axes[1].plot(thetas, rhos, color=color, alpha=0.3, linewidth=0.5)

    for line in lines[:5]:
        axes[1].plot(line.theta, line.rho, 'yo', markersize=10)

    axes[1].set_xlabel('θ (degrees)')
    axes[1].set_ylabel('ρ (pixels)')
    axes[1].set_title("Hough Space Curves")

    # Accumulator heatmap
    if detector.accumulator is not None:
        im = axes[2].imshow(
            detector.accumulator, cmap='hot', aspect='auto',
            extent=[0, 180, detector.rhos[-1], detector.rhos[0]]
        )
        plt.colorbar(im, ax=axes[2], label='Votes')
        for line in lines[:5]:
            axes[2].plot(line.theta, line.rho, 'co', markersize=8)
    axes[2].set_xlabel('θ (degrees)')
    axes[2].set_ylabel('ρ (pixels)')
    axes[2].set_title("Accumulator Heatmap")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hough_curves.png", dpi=150, bbox_inches='tight')
    plt.show()


def create_publication_figure(image, edges, detector, lines):
    """Create publication-quality figure."""
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # (a) Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_rgb)
    ax1.set_title("(a) Input Image", fontsize=12, fontweight='bold')
    ax1.axis('off')

    # (b) Edges
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(edges, cmap='gray')
    ax2.set_title("(b) Canny Edge Detection", fontsize=12, fontweight='bold')
    ax2.axis('off')

    # (c) Hough Space
    ax3 = fig.add_subplot(gs[1, 0])
    if detector.accumulator is not None:
        im = ax3.imshow(
            detector.accumulator, cmap='jet', aspect='auto',
            extent=[0, 180, detector.rhos[-1], detector.rhos[0]]
        )
        plt.colorbar(im, ax=ax3, label='Votes')
    ax3.set_xlabel('θ (degrees)')
    ax3.set_ylabel('ρ (pixels)')
    ax3.set_title("(c) Hough Parameter Space", fontsize=12, fontweight='bold')

    # (d) Results
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(image_rgb)
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(lines), 10)))
    for i, (line, color) in enumerate(zip(lines[:10], colors)):
        x1, y1 = line.start_point
        x2, y2 = line.end_point
        ax4.plot([x1, x2], [y1, y2], color=color, linewidth=2.5,
                 label=f'θ={line.theta:.0f}°' if i < 5 else None)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.set_title(f"(d) Detected Lines (n={len(lines)})", fontsize=12, fontweight='bold')
    ax4.axis('off')

    fig.text(0.5, 0.02,
             "Figure: Hough Transform pipeline - input, edge detection, parameter space, results.",
             ha='center', fontsize=10, style='italic')

    plt.savefig(OUTPUT_DIR / "publication_figure.png", dpi=300, bbox_inches='tight')
    plt.show()


def filter_horizontal_lines(image, lines):
    """Visualize only near-horizontal lines."""
    horizontal = [l for l in lines if l.theta <= 30 or l.theta >= 150]

    fig, ax = plt.subplots(figsize=(8, 8))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image_rgb)

    for line in horizontal:
        x1, y1 = line.start_point
        x2, y2 = line.end_point
        ax.plot([x1, x2], [y1, y2], 'lime', linewidth=2)

    ax.set_title(f"Near-Horizontal Lines (θ: 0-30° or 150-180°)\n{len(horizontal)} lines")
    ax.axis('off')

    plt.savefig(OUTPUT_DIR / "horizontal_lines.png", dpi=150, bbox_inches='tight')
    plt.show()


def main():
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Custom Visualization Example ===")

    image = create_test_image()

    preprocessor = ImagePreprocessor(
        canny_low_threshold=50, canny_high_threshold=150
    )
    edges = preprocessor.preprocess(image)

    params = HoughParameters(vote_threshold=30)
    detector = HoughTransformDetector(params)
    lines = detector.detect_lines(edges)

    logger.info(f"Detected {len(lines)} lines")

    logger.info("Creating angle-based visualization...")
    visualize_by_angle(image, edges, lines)

    logger.info("Creating vote-based visualization...")
    visualize_by_votes(image, edges, lines)

    logger.info("Creating Hough curves visualization...")
    visualize_hough_curves(edges, detector, lines)

    logger.info("Creating publication figure...")
    create_publication_figure(image, edges, detector, lines)

    logger.info("Creating horizontal lines filter...")
    filter_horizontal_lines(image, lines)

    logger.info(f"All visualizations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()