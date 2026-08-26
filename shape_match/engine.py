"""High-level template model encapsulation and shape matching pipeline engine."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Mapping
import cv2
import numpy as np

from shape_match.config import parse_match_config, parse_pattern_config
from shape_match.features import extract_features
from shape_match.image import to_bgr, to_gray, validate_image
from shape_match.matcher import (
    appearance_score,
    coarse_search,
    nms,
    refine_candidate,
    refine_subpixel_candidate,
    resample_candidate_patch,
)
from shape_match.response_maps import orientation_response_maps
from shape_match.types import (
    APPEARANCE_MIN_SCORE,
    MIN_FEATURES,
    Candidate,
    MatchConfig,
    ModelFeatures,
    PatternConfig,
    UInt8Image,
)
from shape_match.visualization import draw_matches, draw_model_features

LOGGER = logging.getLogger(__name__)


class TemplateModel:
    """Encapsulates a template image, its extracted shape features, and extraction configuration.

    This class enables secondary developers to extract a template once and reuse it across
    multiple search images (e.g. video streams or batch image processing) without redundant
    feature extraction overhead.
    """

    def __init__(self, features: ModelFeatures, config: PatternConfig) -> None:
        self.features = features
        self.config = config

    @classmethod
    def from_image(
        cls,
        image: np.ndarray,
        config: Mapping[str, Any] | PatternConfig | None = None,
    ) -> TemplateModel | None:
        """Create a TemplateModel from an input image and optional configuration.

        Returns ``None`` if the template is flat or has fewer than 8 usable edge features.
        """
        pattern_config = config if isinstance(config, PatternConfig) else parse_pattern_config(config)
        validated = validate_image(image, "model")
        gray = to_gray(validated)
        bgr = to_bgr(validated)
        features = extract_features(gray, pattern_config, bgr)
        if features is None:
            LOGGER.warning("model feature extraction failed: fewer than %d usable edge points", MIN_FEATURES)
            return None
        return cls(features, pattern_config)

    @property
    def feature_count(self) -> int:
        """Number of extracted gradient-orientation feature points."""
        return len(self.features.points)

    @property
    def width(self) -> int:
        """Template width in pixels."""
        return self.features.width

    @property
    def height(self) -> int:
        """Template height in pixels."""
        return self.features.height

    def draw(self, image: np.ndarray | None = None) -> UInt8Image:
        """Draw extracted features overlaid on the template or a specified background."""
        base_img = image if image is not None else self.features.template_gray
        return draw_model_features(base_img, self.features)


class ShapeMatcher:
    """Shape-based template matcher executing coarse-to-fine multi-pose search."""

    def __init__(
        self,
        pat_config: Mapping[str, Any] | PatternConfig | None = None,
        match_config: Mapping[str, Any] | MatchConfig | None = None,
    ) -> None:
        self.pattern_config = pat_config if isinstance(pat_config, PatternConfig) else parse_pattern_config(pat_config)
        self.match_config = match_config if isinstance(match_config, MatchConfig) else parse_match_config(match_config)

    def match(
        self,
        template: TemplateModel | ModelFeatures,
        source: np.ndarray,
    ) -> tuple[list[Candidate], UInt8Image | None]:
        """Match template against source image and return candidate list and visualization."""
        features = template.features if isinstance(template, TemplateModel) else template
        source_array = validate_image(source, "src")
        source_gray = to_gray(source_array)

        if self.pattern_config.num_levels == 1:
            coarse_gray = cv2.pyrDown(source_gray)
            image_factor = 0.5
        else:
            coarse_gray = source_gray
            image_factor = 1.0

        coarse_config = self.pattern_config
        if image_factor != 1.0 and self.pattern_config.min_cont_len > 1:
            coarse_config = replace(
                self.pattern_config,
                min_cont_len=max(
                    1, int(round(self.pattern_config.min_cont_len * image_factor))
                ),
            )
        coarse_responses = orientation_response_maps(coarse_gray, coarse_config)
        coarse = coarse_search(features, coarse_responses, self.pattern_config, self.match_config, image_factor)
        if not coarse:
            return [], None

        full_responses = orientation_response_maps(source_gray, self.pattern_config)
        
        try:
            import torch
            from shape_match.torch_refine import refine_candidates_pytorch, HAS_TORCH
            if HAS_TORCH and torch.cuda.is_available():
                refined = refine_candidates_pytorch(coarse, features, full_responses, self.pattern_config, self.match_config)
            else:
                raise ImportError
        except ImportError:
            refined = []
            for candidate in coarse:
                result = refine_candidate(candidate, features, full_responses, self.pattern_config, self.match_config)
                if result is not None and result.score >= self.match_config.min_score:
                    refined.append(result)

        refined = [
            refine_subpixel_candidate(
                candidate, features, full_responses, self.match_config.subpixel
            )
            for candidate in refined
        ]

        # Appearance is only a verification signal.  A template and its target
        # can legitimately have very different foreground/background colours
        # (for example, the white/green mark versus the orange/yellow-green
        # mark in ``test_data_failed``), while their edge geometry remains an
        # excellent match.  Keep the legacy appearance gate for ordinary
        # candidates, but allow a clearly dominant shape score through it.  The
        # margin is deliberately small so a weak periodic-texture response is
        # still rejected when a similarly strong candidate has better
        # appearance consistency.
        scored_candidates: list[tuple[Candidate, float]] = []
        for result in refined:
            patch, patch_valid = resample_candidate_patch(source_gray, features, result)
            app_score = appearance_score(
                features.template_gray, features.appearance_mask, patch, patch_valid
            )
            scored_candidates.append((result, app_score))

        if not scored_candidates:
            return [], None

        best_shape_score = max(result.score for result, _ in scored_candidates)
        shape_bypass_score = max(0.65, self.match_config.min_score + 0.15)
        shape_bypass_margin = 0.05
        final_candidates = [
            result
            for result, app_score in scored_candidates
            if app_score >= APPEARANCE_MIN_SCORE
            or (
                result.score >= shape_bypass_score
                and result.score >= best_shape_score - shape_bypass_margin
            )
        ]

        if not final_candidates:
            return [], None

        matches = nms(
            final_candidates,
            features,
            self.match_config.num_matches,
            self.match_config.max_overlap,
        )
        if not matches:
            return [], None

        visualization = draw_matches(source_array, matches, features)
        return matches, visualization


def extract_model_shape(
    model: np.ndarray, pat_config: Mapping[str, Any] | PatternConfig = {}
) -> np.ndarray | None:
    """Core function to extract and visualize template shape features."""
    config = pat_config if isinstance(pat_config, PatternConfig) else parse_pattern_config(pat_config)
    template = TemplateModel.from_image(model, config)
    if template is None:
        return None
    return template.draw(model)


def match_template(
    model: np.ndarray | TemplateModel,
    src: np.ndarray,
    pat_config: Mapping[str, Any] = {},
    match_config: Mapping[str, Any] = {},
) -> tuple[list[list[float]], np.ndarray | None]:
    """Core function to find occurrences of model in src image."""
    pattern_config = parse_pattern_config(pat_config)
    matching_config = parse_match_config(match_config)

    if isinstance(model, TemplateModel):
        template = model
    else:
        template = TemplateModel.from_image(model, pattern_config)
        if template is None:
            LOGGER.warning("matching skipped: model contains fewer than %d usable edge points", MIN_FEATURES)
            return [], None

    matcher = ShapeMatcher(pattern_config, matching_config)
    matches, show = matcher.match(template, src)
    if not matches:
        return [], None

    result_rows = [match.to_list() for match in matches]
    return result_rows, show
