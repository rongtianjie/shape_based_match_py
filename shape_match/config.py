"""Configuration handling, parameter defaults, and strict validation."""

from __future__ import annotations

import math
from typing import Any, Mapping
import numpy as np

from shape_match.types import MatchConfig, PatternConfig

PAT_DEFAULTS: dict[str, int | float] = {
    "contrast_low": 3,
    "contrast_high": 5,
    "angle_start": -5.0,
    "angle_extent": 10.0,
    "num_levels": 1,
}

MATCH_DEFAULTS: dict[str, int | float] = {
    "numMatches": 5,
    "minScore": 0.15,
    "scale_min": 1.0,
    "scale_max": 1.0,
}


def as_mapping(value: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
    """Ensure value is a Mapping; return empty dict if None."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def validate_number(value: Any, name: str) -> float:
    """Validate that value is a real, finite number."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_integer(value: Any, name: str) -> int:
    """Validate that value is an integer or exact integer float."""
    number = validate_number(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def parse_pattern_config(config: Mapping[str, Any] | None = None) -> PatternConfig:
    """Parse and validate pattern configuration mapping against specifications."""
    supplied = as_mapping(config, "pat_config")
    unknown = set(supplied) - set(PAT_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown pat_config keys: {', '.join(sorted(unknown))}")

    auto_contrast = "contrast_low" not in supplied and "contrast_high" not in supplied
    values = {**PAT_DEFAULTS, **supplied}

    contrast_low = validate_integer(values["contrast_low"], "contrast_low")
    contrast_high = validate_integer(values["contrast_high"], "contrast_high")
    angle_start = validate_number(values["angle_start"], "angle_start")
    angle_extent = validate_number(values["angle_extent"], "angle_extent")
    num_levels = validate_integer(values["num_levels"], "num_levels")

    if contrast_low <= 0:
        raise ValueError("contrast_low must be greater than 0")
    if contrast_low >= contrast_high:
        raise ValueError("contrast_low must be less than contrast_high")
    if not -360.0 <= angle_start <= 360.0:
        raise ValueError("angle_start must be between -360 and 360")
    if not 0.0 <= angle_extent <= 360.0:
        raise ValueError("angle_extent must be between 0 and 360")
    if angle_start + angle_extent > 360.0 + 1e-9:
        raise ValueError("angle_start + angle_extent must not exceed 360")
    if num_levels not in (0, 1):
        raise ValueError("num_levels must be 0 or 1")

    return PatternConfig(
        contrast_low=contrast_low,
        contrast_high=contrast_high,
        angle_start=angle_start,
        angle_extent=angle_extent,
        num_levels=num_levels,
        auto_contrast=auto_contrast,
    )


def parse_match_config(config: Mapping[str, Any] | None = None) -> MatchConfig:
    """Parse and validate matching configuration mapping against specifications."""
    supplied = as_mapping(config, "match_config")
    unknown = set(supplied) - set(MATCH_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown match_config keys: {', '.join(sorted(unknown))}")

    values = {**MATCH_DEFAULTS, **supplied}

    num_matches = validate_integer(values["numMatches"], "numMatches")
    min_score = validate_number(values["minScore"], "minScore")
    scale_min = validate_number(values["scale_min"], "scale_min")
    scale_max = validate_number(values["scale_max"], "scale_max")

    if num_matches < 1:
        raise ValueError("numMatches must be at least 1")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("minScore must be between 0 and 1")
    if scale_min <= 0.0 or scale_max <= 0.0:
        raise ValueError("scale_min and scale_max must be greater than 0")
    if scale_min > scale_max:
        raise ValueError("scale_min must be less than or equal to scale_max")

    return MatchConfig(
        num_matches=num_matches,
        min_score=min_score,
        scale_min=scale_min,
        scale_max=scale_max,
    )
