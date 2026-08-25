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
    # ``cv2.phase`` uses an optimized vectorized atan2 implementation.  The
    # phase is in [0, 2*pi), so folding theta and theta + pi is equivalent to
    # taking the bin index modulo NUM_ORIENTATIONS; this avoids an additional
    # full-size floating-point modulo array on megapixel images.
    angles = cv2.phase(gx, gy, angleInDegrees=False)
    labels = np.floor(
        (angles + np.pi / (2 * NUM_ORIENTATIONS))
        * (NUM_ORIENTATIONS / np.pi)
    )
    return np.mod(labels.astype(np.int16), NUM_ORIENTATIONS).astype(np.uint8).reshape(gx.shape)


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

    # Each edge pixel already contributes one vote to its own 3x3 window, so
    # ``votes >= 2`` is exactly equivalent to finding one 8-connected edge
    # neighbour with the same orientation.  Comparing shifted views avoids
    # eight full-image box filters and does not wrap at image borders.
    consistent = np.zeros_like(edges)
    height, width = edges.shape
    for offset_y in (-1, 0, 1):
        for offset_x in (-1, 0, 1):
            if offset_x == 0 and offset_y == 0:
                continue
            source_y = slice(max(0, offset_y), min(height, height + offset_y))
            target_y = slice(max(0, -offset_y), min(height, height - offset_y))
            source_x = slice(max(0, offset_x), min(width, width + offset_x))
            target_x = slice(max(0, -offset_x), min(width, width - offset_x))
            consistent[target_y, target_x] |= (
                edges[target_y, target_x]
                & edges[source_y, source_x]
                & (labels[target_y, target_x] == labels[source_y, source_x])
            )

    if int(np.count_nonzero(consistent)) < MIN_FEATURES:
        consistent = edges

    return consistent.astype(np.uint8), gx, gy, magnitude
