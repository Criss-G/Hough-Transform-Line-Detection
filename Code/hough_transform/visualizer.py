"""
Visualization module for Hough Transform results.

This module provides the HoughVisualizer class for creating
comprehensive visualizations of line detection results.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.figure import Figure

from .models import DetectedLine
from .detector import HoughTransformDetector

logger = logging.getLogger(__name__)


class HoughVisualizer:
    """Handles visualization of Hough Transform results.

    Creates multi-panel visualizations showing the original image,
    edge detection results, Hough space, and detected lines.

    Attributes:
        figsize: Figure size for matplotlib plots.
        line_color: Color for detected lines.
        line_width: Width for detected lines.

    Example:
        >>> visualizer = HoughVisualizer(figsize=(16, 4))
        >>> fig = visualizer.visualize(
        ...     original_image, edge_image, detector, lines
        ... )
        >>> plt.show()
    """

    def __init__(
            self,
            figsize: Tuple[int, int] = (16, 4),
            line_color: str = "red",
            line_width: float = 2.0
    ) -> None:
        """Initialize visualizer with display settings.

        Args:
            figsize: Matplotlib figure size (width, height).
            line_color: Color for drawing detected lines.
            line_width: Line width for detected lines.
        """
        self.figsize = figsize
        self.line_color = line_color
        self.line_width = line_width

    def visualize(
            self,
            original_image: np.ndarray,
            edge_image: np.ndarray,
            detector: HoughTransformDetector,
            lines: List[DetectedLine],
            save_path: Optional[Path] = None,
            show_plot: bool = True
    ) -> Figure:
        """Create comprehensive visualization of detection results.

        Creates a 4-panel figure showing:
        1. Original image
        2. Edge detection result
        3. Hough accumulator space (heatmap)
        4. Detected lines overlaid on original image

        Args:
            original_image: Original BGR image.
            edge_image: Binary edge image.
            detector: HoughTransformDetector with computed accumulator.
            lines: List of detected lines.
            save_path: Optional path to save the figure.
            show_plot: Whether to display the plot immediately.

        Returns:
            Matplotlib Figure object.
        """
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 4, figsize=self.figsize)

        # Panel 1: Original Image
        self._plot_original_image(axes[0], image_rgb)

        # Panel 2: Edge Image
        self._plot_edge_image(axes[1], edge_image)

        # Panel 3: Hough Space
        self._plot_hough_space(axes[2], detector)

        # Panel 4: Detected Lines
        self._plot_detected_lines(axes[3], image_rgb, lines)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        if show_plot:
            plt.show()

        return fig

    def _plot_original_image(self, ax: plt.Axes, image: np.ndarray) -> None:
        """Plot the original image.

        Args:
            ax: Matplotlib axes to plot on.
            image: RGB image to display.
        """
        ax.imshow(image)
        ax.set_title("Original Image", fontsize=12, fontweight='bold')
        ax.axis('off')

    def _plot_edge_image(self, ax: plt.Axes, edge_image: np.ndarray) -> None:
        """Plot the edge detection result.

        Args:
            ax: Matplotlib axes to plot on.
            edge_image: Binary edge image.
        """
        ax.imshow(edge_image, cmap='gray')
        ax.set_title("Edge Detection", fontsize=12, fontweight='bold')
        ax.axis('off')

    def _plot_hough_space(
            self,
            ax: plt.Axes,
            detector: HoughTransformDetector
    ) -> None:
        """Plot Hough accumulator space as a heatmap.

        Args:
            ax: Matplotlib axes to plot on.
            detector: Detector with computed accumulator.
        """
        if detector.accumulator is not None:
            im = ax.imshow(
                detector.accumulator,
                cmap='hot',
                aspect='auto',
                extent=[0, 180, detector.rhos[-1], detector.rhos[0]]
            )
            ax.set_xlabel('θ (degrees)', fontsize=10)
            ax.set_ylabel('ρ (pixels)', fontsize=10)
            plt.colorbar(im, ax=ax, label='Votes')
        ax.set_title("Hough Space", fontsize=12, fontweight='bold')

    def _plot_detected_lines(
            self,
            ax: plt.Axes,
            image: np.ndarray,
            lines: List[DetectedLine]
    ) -> None:
        """Plot detected lines overlaid on the image.

        Args:
            ax: Matplotlib axes to plot on.
            image: RGB image to overlay lines on.
            lines: List of detected lines.
        """
        ax.imshow(image)

        for i, line in enumerate(lines):
            x1, y1 = line.start_point
            x2, y2 = line.end_point
            ax.add_line(mlines.Line2D(
                [x1, x2], [y1, y2],
                color=self.line_color,
                linewidth=self.line_width,
                label=f'Line {i + 1}' if i < 5 else None
            ))

        ax.set_title(
            f"Detected Lines ({len(lines)})",
            fontsize=12,
            fontweight='bold'
        )
        ax.axis('off')

        if lines and len(lines) <= 5:
            ax.legend(loc='upper right', fontsize=8)

    def _save_figure(self, fig: Figure, save_path: Path) -> None:
        """Save figure to disk.

        Args:
            fig: Figure to save.
            save_path: Path to save to.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")

    def visualize_hough_curves(
            self,
            edge_image: np.ndarray,
            detector: HoughTransformDetector,
            max_curves: int = 100,
            alpha: float = 0.05
    ) -> Figure:
        """Visualize Hough curves for edge points.

        Shows how edge points map to sinusoidal curves in Hough space.

        Args:
            edge_image: Binary edge image.
            detector: Detector with computed parameters.
            max_curves: Maximum number of curves to draw.
            alpha: Transparency of curves.

        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        height, width = edge_image.shape[:2]
        center_y, center_x = height / 2, width / 2

        # Get edge points
        edge_points = np.argwhere(edge_image != 0)
        edge_points_centered = edge_points - np.array([[center_y, center_x]])

        # Limit number of curves for performance
        if len(edge_points_centered) > max_curves:
            indices = np.random.choice(
                len(edge_points_centered), max_curves, replace=False
            )
            edge_points_sample = edge_points_centered[indices]
        else:
            edge_points_sample = edge_points_centered

        # Plot curves
        thetas = np.linspace(0, 180, 180)
        cos_thetas = np.cos(np.deg2rad(thetas))
        sin_thetas = np.sin(np.deg2rad(thetas))

        for point in edge_points_sample:
            rhos = point[0] * sin_thetas + point[1] * cos_thetas
            ax.plot(thetas, rhos, color='white', alpha=alpha)

        ax.set_facecolor('black')
        ax.set_xlabel('θ (degrees)', fontsize=12)
        ax.set_ylabel('ρ (pixels)', fontsize=12)
        ax.set_title('Hough Space Curves', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        plt.tight_layout()
        return fig