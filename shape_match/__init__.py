"""Shape-based 2D template matching package based on gradient orientation features."""

from __future__ import annotations

from shape_match.config import parse_match_config, parse_pattern_config
from shape_match.engine import (
    ShapeMatcher,
    TemplateModel,
    extract_model_shape,
    match_template,
)
from shape_match.gradients import estimate_contrast_thresholds
from shape_match.types import (
    Candidate,
    MatchConfig,
    MatchResult,
    ModelFeatures,
    PatternConfig,
    PoseKernel,
)

__all__ = [
    "TemplateModel",
    "ShapeMatcher",
    "extract_model_shape",
    "match_template",
    "estimate_contrast_thresholds",
    "PatternConfig",
    "MatchConfig",
    "ModelFeatures",
    "PoseKernel",
    "Candidate",
    "MatchResult",
    "parse_pattern_config",
    "parse_match_config",
]
