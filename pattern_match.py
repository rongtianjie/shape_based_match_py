"""Gradient-orientation based 2-D shape template matching.

The implementation follows the main ideas behind HALCON shape models and
LINE-MOD: a template is represented by a small, spatially distributed set of
edge points and their (polarity independent) gradient orientations. Matching
is performed from coarse to fine over position, rotation, and scale.

Only :func:`get_model_shape` and :func:`get_matched_result` are public API.
"""

from __future__ import annotations

from functools import wraps
import logging
from time import perf_counter
from typing import Any, Callable, Mapping, TypeVar
import numpy as np

from shape_match.config import (
    MATCH_DEFAULTS as _MATCH_DEFAULTS,
    PAT_DEFAULTS as _PAT_DEFAULTS,
    parse_match_config as _parse_match_config,
    parse_pattern_config as _parse_pattern_config,
)
from shape_match.engine import (
    ShapeMatcher,
    TemplateModel,
    extract_model_shape,
    match_template,
)
from shape_match.features import extract_features as _extract_features
from shape_match.image import (
    to_bgr as _to_bgr,
    to_gray as _to_gray,
    to_uint8 as _to_uint8,
    validate_image as _validate_image,
)
from shape_match.matcher import (
    coarse_search as _coarse_search,
    nms as _nms,
    refine_candidate as _refine_candidate,
)
from shape_match.response_maps import orientation_response_maps as _orientation_response_maps
from shape_match.types import (
    Candidate as _Candidate,
    MatchConfig as _MatchConfig,
    ModelFeatures as _ModelFeatures,
    PatternConfig as _PatternConfig,
    PoseKernel as _PoseKernel,
)
from shape_match.visualization import (
    draw_matches as _draw_matches,
    draw_model_features as _draw_model_features,
)

__all__ = ["get_model_shape", "get_matched_result"]

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
            LOGGER.info("%s completed in %.3f ms", func.__name__, (perf_counter() - started) * 1000.0)

    return wrapper  # type: ignore[return-value]


@count_time
def get_model_shape(model: np.ndarray, pat_config: Mapping[str, Any] = {}) -> np.ndarray | None:
    """Return a BGR visualization of the shape features extracted from *model*.

    A flat or otherwise unusable template returns ``None``. Invalid arguments
    raise ``TypeError`` or ``ValueError``.
    """
    return extract_model_shape(model, pat_config)


@count_time
def get_matched_result(
    model: np.ndarray,
    src: np.ndarray,
    pat_config: Mapping[str, Any] = {},
    match_config: Mapping[str, Any] = {},
) -> tuple[list[list[float]], np.ndarray | None]:
    """Find transformed occurrences of *model* in *src*.

    Results are sorted by descending score and have the form
    ``[cx, cy, score, angle, scale]``. Positive angles rotate the model
    counter-clockwise in image coordinates.
    """
    return match_template(model, src, pat_config, match_config)
