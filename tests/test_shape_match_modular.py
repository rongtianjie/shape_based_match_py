"""Unit tests for the newly modularized shape_match package and secondary development APIs."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from shape_match import (
    Candidate,
    MatchConfig,
    ModelFeatures,
    PatternConfig,
    ShapeMatcher,
    TemplateModel,
    extract_model_shape,
    match_template,
    parse_match_config,
    parse_pattern_config,
)
from shape_match.config import validate_integer, validate_number
from shape_match.gradients import gradient_fields, quantize_orientations
from shape_match.image import to_bgr, to_gray, to_uint8, validate_image
from shape_match.transforms import candidate_polygon, polygon_iou, rotation_matrix


def make_test_model() -> np.ndarray:
    image = np.zeros((72, 88), dtype=np.uint8)
    cv2.rectangle(image, (9, 8), (72, 56), 220, 3)
    cv2.line(image, (16, 48), (62, 16), 255, 4)
    cv2.circle(image, (66, 48), 8, 180, 3)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_template_model_creation_and_reuse() -> None:
    model_img = make_test_model()
    pat_config = {"contrast_low": 10, "contrast_high": 25}

    template = TemplateModel.from_image(model_img, pat_config)
    assert template is not None
    assert template.feature_count >= 8
    assert template.width == 88
    assert template.height == 72

    # Draw template features
    show = template.draw()
    assert show.shape == (72, 88, 3)

    # Secondary development: reuse the same template instance across multiple source images
    source1 = np.zeros((200, 200, 3), dtype=np.uint8)
    source1[50 : 50 + 72, 50 : 50 + 88] = model_img

    source2 = np.zeros((220, 220, 3), dtype=np.uint8)
    source2[80 : 80 + 72, 80 : 80 + 88] = model_img

    matcher = ShapeMatcher(pat_config, {"numMatches": 1, "minScore": 0.5})
    matches1, view1 = matcher.match(template, source1)
    matches2, view2 = matcher.match(template, source2)

    assert len(matches1) == 1
    assert len(matches2) == 1
    assert pytest.approx(matches1[0].cx, abs=1.0) == 50 + (88 - 1) / 2
    assert pytest.approx(matches1[0].cy, abs=1.0) == 50 + (72 - 1) / 2
    assert pytest.approx(matches2[0].cx, abs=1.0) == 80 + (88 - 1) / 2
    assert pytest.approx(matches2[0].cy, abs=1.0) == 80 + (72 - 1) / 2


def test_config_parsing_and_dataclasses() -> None:
    pat = parse_pattern_config({"contrast_low": 8, "contrast_high": 20, "angle_start": -30.0, "angle_extent": 60.0})
    assert isinstance(pat, PatternConfig)
    assert pat.contrast_low == 8
    assert pat.contrast_high == 20
    assert pat.angle_start == -30.0
    assert pat.angle_extent == 60.0
    assert not pat.auto_contrast

    mat = parse_match_config({"numMatches": 3, "minScore": 0.7, "scale_min": 0.9, "scale_max": 1.1})
    assert isinstance(mat, MatchConfig)
    assert mat.num_matches == 3
    assert mat.min_score == 0.7
    assert mat.scale_min == 0.9
    assert mat.scale_max == 1.1


def test_image_helpers() -> None:
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    assert validate_image(arr, "test") is arr
    assert to_gray(arr).shape == (10, 10)
    assert to_bgr(np.zeros((10, 10), dtype=np.uint8)).shape == (10, 10, 3)
    assert to_uint8(np.array([[300, -10]], dtype=np.float32)).tolist() == [[255, 0]]


def test_gradient_and_transforms() -> None:
    gray = np.zeros((30, 30), dtype=np.uint8)
    cv2.rectangle(gray, (5, 5), (25, 25), 255, 2)
    gx, gy, mag, safe = gradient_fields(gray)
    assert gx.shape == (30, 30)
    labels = quantize_orientations(gx, gy)
    assert labels.shape == (30, 30)
    assert labels.dtype == np.uint8

    rot = rotation_matrix(90.0, 1.0)
    assert rot.shape == (2, 2)

    poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    poly2 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    assert pytest.approx(polygon_iou(poly1, poly2), abs=1e-4) == 1.0
