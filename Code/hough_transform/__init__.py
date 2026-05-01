"""
Hough Transform Line Detection Package.

A modular implementation of the Hough Transform algorithm for detecting
lines in images, designed with software engineering best practices.
"""

from .models import HoughParameters, DetectedLine
from .preprocessor import ImagePreprocessor
from .detector import HoughTransformDetector
from .visualizer import HoughVisualizer
from .utils import load_image, save_image, setup_logging

__all__ = [
    "HoughParameters",
    "DetectedLine",
    "ImagePreprocessor",
    "HoughTransformDetector",
    "HoughVisualizer",
    "load_image",
    "save_image",
]

__version__ = "1.0.0"
__author__ = "Cristina Gombar"