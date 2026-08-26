"""Configuration handling, parameter defaults, and strict validation."""

from __future__ import annotations

import math
from typing import Any, Mapping
import numpy as np

from shape_match.types import MatchConfig, PatternConfig

PAT_DEFAULTS: dict[str, int | float] = {
    "contrast_low": 3,
    "contrast_high": 5,
    "min_contrast": 1,
    "min_cont_len": 1,
    "num_levels": 1,
    "use_polarity": 0,
    "angle_start": 0.0,
    "angle_extent": 0.0,
    "angle_step": 0.0,
}

MATCH_DEFAULTS: dict[str, int | float] = {
    "subpixel": 1,
    "scale_min": 0.8,
    "scale_max": 1.2,
    "minScore": 0.15,
    "maxOverLap": 0.5,
    "greedness": 0.75,
    "numMatches": 1,
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

    auto_contrast = False
    values = {**PAT_DEFAULTS, **supplied}

    contrast_low = validate_integer(values["contrast_low"], "contrast_low")
    contrast_high = validate_integer(values["contrast_high"], "contrast_high")
    min_contrast = validate_integer(values["min_contrast"], "min_contrast")
    min_cont_len = validate_integer(values["min_cont_len"], "min_cont_len")
    use_polarity = validate_integer(values["use_polarity"], "use_polarity")
    angle_start = validate_number(values["angle_start"], "angle_start")
    angle_extent = validate_number(values["angle_extent"], "angle_extent")
    angle_step = validate_number(values["angle_step"], "angle_step")
    num_levels = validate_integer(values["num_levels"], "num_levels")

    if contrast_low <= 0:
        raise ValueError("contrast_low must be greater than 0")
    if contrast_low >= contrast_high:
        raise ValueError("contrast_low must be less than contrast_high")
    if not 0 <= min_contrast <= 255:
        raise ValueError("min_contrast must be between 0 and 255")
    if min_cont_len < 1:
        raise ValueError("min_cont_len must be at least 1")
    if use_polarity not in (0, 1):
        raise ValueError("use_polarity must be 0 or 1")
    if not -360.0 <= angle_start <= 360.0:
        raise ValueError("angle_start must be between -360 and 360")
    if not 0.0 <= angle_extent <= 360.0:
        raise ValueError("angle_extent must be between 0 and 360")
    if angle_start + angle_extent > 360.0 + 1e-9:
        raise ValueError("angle_start + angle_extent must not exceed 360")
    if not 0.0 <= angle_step <= 360.0:
        raise ValueError("angle_step must be between 0 and 360")
    if num_levels not in (0, 1):
        raise ValueError("num_levels must be 0 or 1")

    return PatternConfig(
        contrast_low=contrast_low,
        contrast_high=contrast_high,
        min_contrast=min_contrast,
        min_cont_len=min_cont_len,
        use_polarity=use_polarity,
        angle_start=angle_start,
        angle_extent=angle_extent,
        angle_step=angle_step,
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
    subpixel = validate_integer(values["subpixel"], "subpixel")
    max_overlap = validate_number(values["maxOverLap"], "maxOverLap")
    greediness = validate_number(values["greedness"], "greedness")

    if num_matches < 1:
        raise ValueError("numMatches must be at least 1")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("minScore must be between 0 and 1")
    if scale_min <= 0.0 or scale_max <= 0.0:
        raise ValueError("scale_min and scale_max must be greater than 0")
    if scale_min > scale_max:
        raise ValueError("scale_min must be less than or equal to scale_max")
    if subpixel not in (0, 1, 2):
        raise ValueError("subpixel must be 0, 1, or 2")
    if not 0.0 <= max_overlap <= 1.0:
        raise ValueError("maxOverLap must be between 0 and 1")
    if not 0.0 <= greediness <= 1.0:
        raise ValueError("greedness must be between 0 and 1")

    return MatchConfig(
        num_matches=num_matches,
        min_score=min_score,
        scale_min=scale_min,
        scale_max=scale_max,
        subpixel=subpixel,
        max_overlap=max_overlap,
        greediness=greediness,
    )
