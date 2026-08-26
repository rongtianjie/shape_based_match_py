"""Gradient field computation, orientation quantization, and edge extraction."""

from __future__ import annotations

import cv2
import numpy as np

from shape_match.config import PAT_DEFAULTS
from shape_match.image import to_gray, validate_image
from shape_match.types import (
    AUTO_CANNY_HIGH_RATIO,
    AUTO_CANNY_LOW_RATIO,
    DIRECTED_NUM_ORIENTATIONS,
    MIN_FEATURES,
    NUM_ORIENTATIONS,
    FloatImage,
    PatternConfig,
    UInt8Image,
)


_CONTRAST_BLUR_KERNEL: tuple[int, int] = (5, 5)
_CONTRAST_STRONG_PERCENTILE: float = 85.0
_CONTRAST_LOW_RATIO_MIN: float = 0.40
_CONTRAST_LOW_RATIO_MAX: float = 0.75
_CONTRAST_LOW_RATIO_RAMP_START: float = 20.0
_CONTRAST_LOW_RATIO_RAMP_END: float = 45.0
_SOBEL_CONTRAST_SCALE: float = 25.5


def gradient_fields(gray: UInt8Image) -> tuple[FloatImage, FloatImage, FloatImage, FloatImage]:
    """Compute horizontal/vertical Sobel gradients, magnitude, and non-zero safe magnitude."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    safe = np.maximum(magnitude, np.float32(1e-6))
    return gx, gy, magnitude, safe


def quantize_orientations(
    gx: FloatImage, gy: FloatImage, use_polarity: bool = False
) -> UInt8Image:
    """Quantize gradient orientations, optionally preserving gradient polarity.

    With ``use_polarity=False``, theta and theta + pi share one of eight
    labels.  With ``use_polarity=True`` they occupy distinct labels in a
    sixteen-bin full-circle representation.
    """
    # ``cv2.phase`` uses an optimized vectorized atan2 implementation.  The
    # phase is in [0, 2*pi), so folding theta and theta + pi is equivalent to
    # taking the bin index modulo NUM_ORIENTATIONS; this avoids an additional
    # full-size floating-point modulo array on megapixel images.
    angles = cv2.phase(gx, gy, angleInDegrees=False)
    orientation_count = DIRECTED_NUM_ORIENTATIONS if use_polarity else NUM_ORIENTATIONS
    period = 2.0 * np.pi if use_polarity else np.pi
    labels = np.floor((angles + period / (2 * orientation_count)) * (orientation_count / period))
    return np.mod(labels.astype(np.int16), orientation_count).astype(np.uint8).reshape(gx.shape)


def _remove_short_edge_components(edges: np.ndarray, min_cont_len: int) -> np.ndarray:
    """Remove 8-connected components whose longest contour is too short."""
    if min_cont_len <= 1 or not np.any(edges):
        return edges
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        edges.astype(np.uint8), connectivity=8
    )
    keep = np.zeros(count, dtype=bool)
    for label in range(1, count):
        # Skip obviously short components before allocating their masks.
        if stats[label, cv2.CC_STAT_AREA] < 2:
            continue
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        contour_length = max(
            (cv2.arcLength(contour, closed=False) for contour in contours),
            default=0.0,
        )
        keep[label] = contour_length >= float(min_cont_len)
    return keep[labels]


def auto_canny_thresholds(magnitude: FloatImage) -> tuple[int, int]:
    """Estimate Canny low/high thresholds dynamically from the gradient magnitude distribution."""
    nonzero = magnitude[magnitude > 1.0]
    if nonzero.size < 16:
        return int(PAT_DEFAULTS["contrast_low"]), int(PAT_DEFAULTS["contrast_high"])
    median = float(np.median(nonzero))
    low = int(np.clip(round(AUTO_CANNY_LOW_RATIO * median), 1, 250))
    high = int(np.clip(round(AUTO_CANNY_HIGH_RATIO * median), low + 1, 255))
    return low, high


def estimate_contrast_thresholds(template: np.ndarray) -> tuple[int, int]:
    """Estimate explicit Canny low/high thresholds from a template image.

    A small Gaussian blur keeps JPEG noise and thin high-contrast overlays from
    dominating the gradient histogram.  Otsu's threshold is additionally
    capped at the upper quantile of non-zero gradients, so a sparse watermark
    cannot suppress a larger, lower-contrast object contour.  The hysteresis
    ratio rises from 40% for weak templates to 75% for high-contrast templates:
    weak contours retain connectivity, while stronger templates reject more
    texture noise.  Flat or nearly flat templates fall back to the package
    defaults; the returned pair is always valid for :class:`PatternConfig`.
    """
    validated = validate_image(template, "template")
    gray = to_gray(validated)
    smoothed = cv2.GaussianBlur(gray, _CONTRAST_BLUR_KERNEL, 0)
    _, _, magnitude, _ = gradient_fields(smoothed)
    clipped = np.clip(magnitude, 0.0, 255.0).astype(np.uint8)
    nonzero = clipped[clipped > 1]
    if nonzero.size < 16:
        return int(PAT_DEFAULTS["contrast_low"]), int(PAT_DEFAULTS["contrast_high"])

    otsu_threshold, _ = cv2.threshold(
        clipped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    percentile_threshold = float(np.percentile(nonzero, _CONTRAST_STRONG_PERCENTILE))
    if otsu_threshold < 2:
        high = int(round(percentile_threshold))
    else:
        high = int(round(min(float(otsu_threshold), percentile_threshold)))
    high = int(np.clip(high, 2, 255))
    contrast_position = float(
        np.clip(
            (high - _CONTRAST_LOW_RATIO_RAMP_START)
            / (_CONTRAST_LOW_RATIO_RAMP_END - _CONTRAST_LOW_RATIO_RAMP_START),
            0.0,
            1.0,
        )
    )
    low_ratio = _CONTRAST_LOW_RATIO_MIN + contrast_position * (
        _CONTRAST_LOW_RATIO_MAX - _CONTRAST_LOW_RATIO_MIN
    )
    low = int(np.clip(round(high * low_ratio), 1, high - 1))
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
    # The legacy native matcher exposes min_contrast on an approximately
    # 0..10 scale.  A full 8-bit vertical step produces a 3x3 Sobel response
    # of 4*255.  Scaling by one tenth of the input intensity range, then
    # clipping, gives the native parameter its observed practical 0..10 range.
    normalized_contrast = np.minimum(magnitude / _SOBEL_CONTRAST_SCALE, 10.0)
    edges &= normalized_contrast >= float(config.min_contrast)
    edges = _remove_short_edge_components(edges, config.min_cont_len)
    labels = quantize_orientations(gx, gy, bool(config.use_polarity))

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
