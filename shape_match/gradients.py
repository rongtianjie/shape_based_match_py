"""Gradient field computation, orientation quantization, and edge extraction."""

from __future__ import annotations

import cv2
import numpy as np

from shape_match.config import PAT_DEFAULTS
from shape_match.types import (
    AUTO_CANNY_HIGH_RATIO,
    AUTO_CANNY_LOW_RATIO,
    MIN_FEATURES,
    NUM_ORIENTATIONS,
    FloatImage,
    PatternConfig,
    UInt8Image,
)


def gradient_fields(gray: UInt8Image) -> tuple[FloatImage, FloatImage, FloatImage, FloatImage]:
    """Compute horizontal/vertical Sobel gradients, magnitude, and non-zero safe magnitude."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    safe = np.maximum(magnitude, np.float32(1e-6))
    return gx, gy, magnitude, safe


def quantize_orientations(gx: FloatImage, gy: FloatImage) -> UInt8Image:
    """Quantize undirected gradient orientations into [0, NUM_ORIENTATIONS - 1] bins.

    Theta and theta + pi share the same label (undirected / polarity invariant).
    """
    angles = np.mod(np.arctan2(gy, gx), np.pi)
    labels = np.floor((angles + np.pi / (2 * NUM_ORIENTATIONS)) * (NUM_ORIENTATIONS / np.pi))
    return np.mod(labels.astype(np.int16), NUM_ORIENTATIONS).astype(np.uint8)


def auto_canny_thresholds(magnitude: FloatImage) -> tuple[int, int]:
    """Estimate Canny low/high thresholds dynamically from the gradient magnitude distribution."""
    nonzero = magnitude[magnitude > 1.0]
    if nonzero.size < 16:
        return int(PAT_DEFAULTS["contrast_low"]), int(PAT_DEFAULTS["contrast_high"])
    median = float(np.median(nonzero))
    low = int(np.clip(round(AUTO_CANNY_LOW_RATIO * median), 1, 250))
    high = int(np.clip(round(AUTO_CANNY_HIGH_RATIO * median), low + 1, 255))
    return low, high


def consistent_edges(
    gray: UInt8Image, config: PatternConfig
) -> tuple[UInt8Image, FloatImage, FloatImage, FloatImage]:
    """Extract Canny edges filtered by 3x3 neighbourhood orientation consistency voting.

    Following LINE-MOD principles, keeps only edge pixels whose orientation label is
    supported by at least one neighbour with the same orientation.
    """
    gx, gy, magnitude, _ = gradient_fields(gray)
    if config.auto_contrast:
        low, high = auto_canny_thresholds(magnitude)
    else:
        low, high = config.contrast_low, config.contrast_high

    edges = cv2.Canny(gray, low, high, apertureSize=3, L2gradient=True) > 0
    labels = quantize_orientations(gx, gy)

    consistent = np.zeros_like(edges)
    for label in range(NUM_ORIENTATIONS):
        members = (edges & (labels == label)).astype(np.uint8)
        votes = cv2.boxFilter(members, cv2.CV_16U, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        consistent |= (members > 0) & (votes >= 2)

    if int(np.count_nonzero(consistent)) < MIN_FEATURES:
        consistent = edges

    return consistent.astype(np.uint8), gx, gy, magnitude
