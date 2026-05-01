"""Setup configuration for Hough Transform package."""

from setuptools import setup, find_packages

setup(
    name="hough-transform",
    version="1.0.0",
    description="Line detection using Hough Transform algorithm",
    author="Cristina Gombar",
    author_email="cristina.gombar2@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "hough-detect=hough_transform.detector:main",
        ],
    },
)