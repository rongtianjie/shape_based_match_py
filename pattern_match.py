"""Gradient-orientation based 2-D shape template matching.

The implementation follows the main ideas behind HALCON shape models and
LINE-MOD: a template is represented by a small, spatially distributed set of
edge points and their (polarity independent) gradient orientations. Matching
is performed from coarse to fine over position, rotation, and scale.

The public API also exposes :func:`estimate_contrast_thresholds` for deriving
explicit Canny thresholds from a template image.
"""

from __future__ import annotations

from functools import wraps
import logging
from time import perf_counter
from typing import Any, Callable, Mapping, TypeVar

import numpy as np

from shape_match.engine import extract_model_shape, match_template
from shape_match.gradients import estimate_contrast_thresholds

__all__ = [
    "get_model_shape",
    "get_matched_result",
    "get_aligned_areas",
    "estimate_contrast_thresholds",
]

LOGGER = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


def count_time(func: F) -> F:
    """Log public-call duration without configuring application logging."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            LOGGER.info(
                "%s completed in %.3f ms",
                func.__name__,
                (perf_counter() - started) * 1000.0,
            )

    return wrapper  # type: ignore[return-value]


@count_time
def get_model_shape(
    model: np.ndarray, pat_config: Mapping[str, Any] | None = None
) -> np.ndarray | None:
    """Extract and visualize shape features using the Python implementation."""
    return extract_model_shape(model, pat_config or {})


@count_time
def get_matched_result(
    model: np.ndarray,
    src: np.ndarray,
    pat_config: Mapping[str, Any] | None = None,
    match_config: Mapping[str, Any] | None = None,
) -> tuple[list[list[float]], np.ndarray | None]:
    """Return ``[cx, cy, score, angle, scale]`` rows and a result view."""
    return match_template(model, src, pat_config or {}, match_config or {})

def get_aligned_areas(
    src: np.ndarray, mark: np.ndarray, results: list[list[float]]
) -> list[np.ndarray]:
    """Extract axis-aligned bounding boxes around match locations."""
    if not results:
        return []
    aligned_areas = []
    mark_w, mark_h = mark.shape[1], mark.shape[0]
    for r in results:
        cx, cy = r[0], r[1]
        scale = r[4]
        w, h = int(mark_w * scale), int(mark_h * scale)
        x1, y1 = max(0, int(cx - w // 2)), max(0, int(cy - h // 2))
        x2, y2 = min(src.shape[1], x1 + w), min(src.shape[0], y1 + h)
        match_pattern = src[y1:y2, x1:x2].copy()
        aligned_areas.append(match_pattern)
    return aligned_areas
