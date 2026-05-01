#!/usr/bin/env python3
"""
Interactive Demo Example

Creates an interactive demonstration with matplotlib widgets:
- Adjustable threshold slider
- Real-time parameter updates
- Live visualization updates

Note: Requires running in an interactive environment (not inline).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from Code.hough_transform import (
    HoughParameters,
    ImagePreprocessor,
    HoughTransformDetector,
)
from Code.hough_transform.utils import setup_logging


class InteractiveHoughDemo:
    """Interactive demonstration of Hough Transform parameters."""

    def __init__(self, image: np.ndarray):
        """Initialize the interactive demo."""
        self.original_image = image
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Current parameters
        self.vote_threshold = 50
        self.canny_low = 50
        self.canny_high = 150
        self.blur_size = 3

        # Process initial image
        self._process_image()

        # Setup figure
        self._setup_figure()

    def _process_image(self):
        """Process image with current parameters."""
        # Preprocessing
        self.preprocessor = ImagePreprocessor(
            blur_kernel_size=(self.blur_size, self.blur_size),
            canny_low_threshold=self.canny_low,
            canny_high_threshold=self.canny_high
        )
        self.edges = self.preprocessor.preprocess(self.original_image)

        # Detection
        params = HoughParameters(vote_threshold=self.vote_threshold)
        self.detector = HoughTransformDetector(params)
        self.lines = self.detector.detect_lines(self.edges)

    def _setup_figure(self):
        """Setup the matplotlib figure with widgets."""
        self.fig = plt.figure(figsize=(14, 8))

        # Main plot areas
        self.ax_original = self.fig.add_axes([0.05, 0.35, 0.28, 0.55])
        self.ax_edges = self.fig.add_axes([0.36, 0.35, 0.28, 0.55])
        self.ax_result = self.fig.add_axes([0.67, 0.35, 0.28, 0.55])

        # Sliders
        ax_threshold = self.fig.add_axes([0.15, 0.22, 0.7, 0.03])
        ax_canny_low = self.fig.add_axes([0.15, 0.17, 0.7, 0.03])
        ax_canny_high = self.fig.add_axes([0.15, 0.12, 0.7, 0.03])
        ax_blur = self.fig.add_axes([0.15, 0.07, 0.7, 0.03])

        self.slider_threshold = Slider(
            ax_threshold, 'Vote Threshold', 10, 300,
            valinit=self.vote_threshold, valstep=5
        )
        self.slider_canny_low = Slider(
            ax_canny_low, 'Canny Low', 10, 200,
            valinit=self.canny_low, valstep=10
        )
        self.slider_canny_high = Slider(
            ax_canny_high, 'Canny High', 50, 300,
            valinit=self.canny_high, valstep=10
        )
        self.slider_blur = Slider(
            ax_blur, 'Blur Size', 1, 9,
            valinit=self.blur_size, valstep=2
        )

        # Connect sliders to update function
        self.slider_threshold.on_changed(self._on_threshold_change)
        self.slider_canny_low.on_changed(self._on_canny_low_change)
        self.slider_canny_high.on_changed(self._on_canny_high_change)
        self.slider_blur.on_changed(self._on_blur_change)

        # Reset button
        ax_reset = self.fig.add_axes([0.8, 0.01, 0.1, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset)

        # Initial draw
        self._update_plots()

    def _on_threshold_change(self, val):
        """Handle threshold slider change."""
        self.vote_threshold = int(val)
        params = HoughParameters(vote_threshold=self.vote_threshold)
        self.detector = HoughTransformDetector(params)
        self.lines = self.detector.detect_lines(self.edges)
        self._update_plots()

    def _on_canny_low_change(self, val):
        """Handle Canny low threshold change."""
        self.canny_low = int(val)
        if self.canny_low >= self.canny_high:
            self.canny_low = self.canny_high - 10
            self.slider_canny_low.set_val(self.canny_low)
        self._process_image()
        self._update_plots()

    def _on_canny_high_change(self, val):
        """Handle Canny high threshold change."""
        self.canny_high = int(val)
        if self.canny_high <= self.canny_low:
            self.canny_high = self.canny_low + 10
            self.slider_canny_high.set_val(self.canny_high)
        self._process_image()
        self._update_plots()

    def _on_blur_change(self, val):
        """Handle blur size change."""
        self.blur_size = int(val)
        if self.blur_size % 2 == 0:
            self.blur_size += 1
        self._process_image()
        self._update_plots()

    def _on_reset(self, event):
        """Reset all parameters to defaults."""
        self.slider_threshold.reset()
        self.slider_canny_low.reset()
        self.slider_canny_high.reset()
        self.slider_blur.reset()

    def _update_plots(self):
        """Update all plot displays."""
        # Clear axes
        self.ax_original.clear()
        self.ax_edges.clear()
        self.ax_result.clear()

        # Original image
        self.ax_original.imshow(self.image_rgb)
        self.ax_original.set_title("Original Image")
        self.ax_original.axis('off')

        # Edge image
        self.ax_edges.imshow(self.edges, cmap='gray')
        edge_count = np.count_nonzero(self.edges)
        self.ax_edges.set_title(f"Edges ({edge_count} pixels)")
        self.ax_edges.axis('off')

        # Result with lines
        self.ax_result.imshow(self.image_rgb)

        for line in self.lines[:20]:  # Limit displayed lines
            x1, y1 = line.start_point
            x2, y2 = line.end_point
            self.ax_result.plot([x1, x2], [y1, y2], 'r-', linewidth=2)

        self.ax_result.set_title(f"Detected Lines ({len(self.lines)})")
        self.ax_result.axis('off')

        self.fig.canvas.draw_idle()

    def show(self):
        """Display the interactive demo."""
        plt.show()


def create_demo_image() -> np.ndarray:
    """Create an interesting demo image."""
    image = np.zeros((400, 400, 3), dtype=np.uint8)

    # Background gradient
    for y in range(400):
        image[y, :] = [20 + y // 10, 20 + y // 10, 30 + y // 10]

    # Various lines
    cv2.line(image, (50, 100), (350, 100), (255, 255, 255), 2)
    cv2.line(image, (50, 200), (350, 200), (255, 255, 255), 2)
    cv2.line(image, (50, 300), (350, 300), (255, 255, 255), 2)
    cv2.line(image, (100, 50), (100, 350), (255, 255, 255), 2)
    cv2.line(image, (200, 50), (200, 350), (255, 255, 255), 2)
    cv2.line(image, (300, 50), (300, 350), (255, 255, 255), 2)
    cv2.line(image, (50, 50), (350, 350), (255, 255, 255), 2)
    cv2.line(image, (350, 50), (50, 350), (255, 255, 255), 2)

    return image


def main():
    """Run interactive demo."""
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=== Interactive Hough Transform Demo ===")
    logger.info("Adjust sliders to see real-time parameter effects")

    # Create demo image
    image = create_demo_image()

    # Create and show interactive demo
    demo = InteractiveHoughDemo(image)
    demo.show()


if __name__ == "__main__":
    main()