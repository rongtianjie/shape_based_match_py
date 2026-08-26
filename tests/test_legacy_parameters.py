from __future__ import annotations

import cv2
import numpy as np
import pytest

from pattern_match import get_matched_result, get_model_shape
from shape_match import Candidate, ModelFeatures, parse_match_config, parse_pattern_config
from shape_match.gradients import (
    _remove_short_edge_components,
    consistent_edges,
    quantize_orientations,
)
from shape_match.matcher import coarse_angles, coarse_search_limits, nms, refine_subpixel_candidate
from shape_match.response_maps import orientation_response_maps


def make_model() -> np.ndarray:
    image = np.zeros((64, 72), dtype=np.uint8)
    cv2.rectangle(image, (8, 7), (61, 54), 220, 3)
    cv2.line(image, (15, 47), (54, 14), 255, 4)
    cv2.circle(image, (54, 45), 7, 170, 3)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


from shape_match.config import MATCH_DEFAULTS, PAT_DEFAULTS


def test_legacy_defaults_and_all_legacy_keys_are_accepted() -> None:
    pat = parse_pattern_config()
    assert (pat.contrast_low, pat.contrast_high) == (PAT_DEFAULTS["contrast_low"], PAT_DEFAULTS["contrast_high"])
    assert (pat.min_contrast, pat.min_cont_len) == (PAT_DEFAULTS["min_contrast"], PAT_DEFAULTS["min_cont_len"])
    assert (pat.num_levels, pat.use_polarity) == (PAT_DEFAULTS["num_levels"], PAT_DEFAULTS["use_polarity"])
    assert (pat.angle_start, pat.angle_extent, pat.angle_step) == (
        PAT_DEFAULTS["angle_start"],
        PAT_DEFAULTS["angle_extent"],
        PAT_DEFAULTS["angle_step"],
    )

    match = parse_match_config()
    assert (match.scale_min, match.scale_max) == (MATCH_DEFAULTS["scale_min"], MATCH_DEFAULTS["scale_max"])
    assert (match.min_score, match.num_matches) == (MATCH_DEFAULTS["minScore"], MATCH_DEFAULTS["numMatches"])
    assert (match.subpixel, match.max_overlap, match.greediness) == (
        MATCH_DEFAULTS["subpixel"],
        MATCH_DEFAULTS["maxOverLap"],
        MATCH_DEFAULTS["greedness"],
    )

    model = make_model()
    assert get_model_shape(
        model,
        {
            "contrast_low": 10,
            "contrast_high": 25,
            "min_contrast": 4,
            "min_cont_len": 2,
            "num_levels": 0,
            "use_polarity": 0,
            "angle_start": 0.0,
            "angle_extent": 0.0,
            "angle_step": 0.0,
        },
    ) is not None


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"use_polarity": 2}, "use_polarity"),
        ({"min_cont_len": 0}, "min_cont_len"),
        ({"min_contrast": -1}, "min_contrast"),
        ({"angle_step": -1}, "angle_step"),
    ],
)
def test_legacy_pattern_parameter_validation(config: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_pattern_config(config)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"subpixel": 3}, "subpixel"),
        ({"maxOverLap": 1.1}, "maxOverLap"),
        ({"greedness": -0.1}, "greedness"),
    ],
)
def test_legacy_match_parameter_validation(config: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_match_config(config)


def test_use_polarity_changes_orientation_domain() -> None:
    gx = np.asarray((1.0, -1.0), dtype=np.float32)
    gy = np.zeros_like(gx)
    invariant = quantize_orientations(gx, gy, use_polarity=False)
    directed = quantize_orientations(gx, gy, use_polarity=True)
    assert invariant[0] == invariant[1]
    assert directed[0] != directed[1]

    gray = cv2.cvtColor(make_model(), cv2.COLOR_BGR2GRAY)
    assert orientation_response_maps(gray, parse_pattern_config({"use_polarity": 0})).shape[0] == 8
    assert orientation_response_maps(gray, parse_pattern_config({"use_polarity": 1})).shape[0] == 16
    directed_view = get_model_shape(make_model(), {"use_polarity": 1})
    assert directed_view is not None and directed_view.shape == make_model().shape


def test_use_polarity_rejects_contrast_inversion() -> None:
    model = make_model()
    source = np.full((180, 200, 3), 255, dtype=np.uint8)
    source[50:114, 70:142] = 255 - model
    common_pattern = {
        "contrast_low": 10,
        "contrast_high": 25,
        "angle_start": 0.0,
        "angle_extent": 0.0,
        "num_levels": 0,
    }
    match_config = {
        "numMatches": 1,
        "minScore": 0.8,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "subpixel": 0,
    }
    invariant, _ = get_matched_result(
        model, source, {**common_pattern, "use_polarity": 0}, match_config
    )
    directed, _ = get_matched_result(
        model, source, {**common_pattern, "use_polarity": 1}, match_config
    )
    assert invariant
    assert directed == []


def test_min_contrast_and_min_cont_len_filter_edges() -> None:
    gray = np.zeros((100, 120), dtype=np.uint8)
    cv2.rectangle(gray, (10, 10), (100, 80), 35, 2)
    cv2.line(gray, (110, 90), (112, 90), 255, 1)

    baseline = parse_pattern_config(
        {"contrast_low": 2, "contrast_high": 5, "min_contrast": 0, "min_cont_len": 1}
    )
    strong_only = parse_pattern_config(
        {"contrast_low": 2, "contrast_high": 5, "min_contrast": 6, "min_cont_len": 1}
    )
    long_only = parse_pattern_config(
        {"contrast_low": 2, "contrast_high": 5, "min_contrast": 0, "min_cont_len": 20}
    )
    baseline_edges = consistent_edges(gray, baseline)[0]
    strong_edges = consistent_edges(gray, strong_only)[0]
    long_edges = consistent_edges(gray, long_only)[0]
    assert np.count_nonzero(strong_edges) < np.count_nonzero(baseline_edges)
    assert np.count_nonzero(long_edges) >= 8

    raw_edges = cv2.Canny(gray, 2, 5, apertureSize=3, L2gradient=True) > 0
    filtered_edges = _remove_short_edge_components(raw_edges, 20)
    assert np.count_nonzero(filtered_edges) < np.count_nonzero(raw_edges)


def test_angle_step_and_greediness_control_search() -> None:
    pattern = parse_pattern_config(
        {"angle_start": -20.0, "angle_extent": 40.0, "angle_step": 15.0}
    )
    assert coarse_angles(pattern) == [-20.0, -5.0, 10.0, 20.0]

    exhaustive = coarse_search_limits(parse_match_config({"greedness": 0.0, "numMatches": 2}))
    greedy = coarse_search_limits(parse_match_config({"greedness": 1.0, "numMatches": 2}))
    assert exhaustive[0] < greedy[0]
    assert exhaustive[1] > greedy[1]
    assert exhaustive[2] > greedy[2]


def make_single_feature_model() -> ModelFeatures:
    return ModelFeatures(
        offsets=np.asarray(((0.0, 0.0),), dtype=np.float32),
        unit_gradients=np.asarray(((1.0, 0.0),), dtype=np.float32),
        points=np.asarray(((0, 0),), dtype=np.int32),
        labels=np.asarray((0,), dtype=np.uint8),
        width=9,
        height=9,
        template_gray=np.zeros((9, 9), dtype=np.uint8),
        appearance_mask=None,
    )


def test_subpixel_modes_and_max_overlap() -> None:
    features = make_single_feature_model()
    yy, xx = np.mgrid[:15, :15]
    response = np.exp(-((xx - 7.3) ** 2 + (yy - 6.6) ** 2) / 3.0).astype(np.float32)
    responses = np.zeros((8, 15, 15), dtype=np.float32)
    responses[0] = response
    candidate = Candidate(7.0, 7.0, float(response[7, 7]), 0.0, 1.0)

    pixel = refine_subpixel_candidate(candidate, features, responses, 0)
    interpolated = refine_subpixel_candidate(candidate, features, responses, 1)
    least_squares = refine_subpixel_candidate(candidate, features, responses, 2)
    assert (pixel.cx, pixel.cy) == (7.0, 7.0)
    assert pixel.score == interpolated.score == least_squares.score == candidate.score
    assert interpolated.cx > 7.0 and interpolated.cy < 7.0
    assert least_squares.cx > 7.0 and least_squares.cy < 7.0

    overlaps = [
        Candidate(20.0, 20.0, 0.9, 0.0, 1.0),
        Candidate(21.0, 20.0, 0.8, 0.0, 1.0),
    ]
    assert len(nms(overlaps, features, 2, max_overlap=0.0)) == 1
    assert len(nms(overlaps, features, 2, max_overlap=1.0)) == 2

    flat_responses = np.ones_like(responses)
    flat = refine_subpixel_candidate(candidate, features, flat_responses, 2)
    assert (flat.cx, flat.cy) == (7.0, 7.0)


def test_subpixel_modes_recover_known_quadratic_peak() -> None:
    features = make_single_feature_model()
    yy, xx = np.mgrid[:12, :12]
    response = 1.0 - 0.02 * (xx - 6.3) ** 2 - 0.03 * (yy - 5.8) ** 2
    responses = np.zeros((8, 12, 12), dtype=np.float32)
    responses[0] = response.astype(np.float32)
    candidate = Candidate(6.0, 6.0, float(response[6, 6]), 0.0, 1.0)

    interpolated = refine_subpixel_candidate(candidate, features, responses, 1)
    least_squares = refine_subpixel_candidate(candidate, features, responses, 2)
    assert (interpolated.cx, interpolated.cy) == pytest.approx((6.3, 5.8), abs=1e-5)
    assert (least_squares.cx, least_squares.cy) == pytest.approx((6.3, 5.8), abs=1e-5)


def test_public_match_accepts_all_legacy_match_keys() -> None:
    model = make_model()
    source = np.zeros((180, 200, 3), dtype=np.uint8)
    source[50:114, 70:142] = model
    results, _ = get_matched_result(
        model,
        source,
        {
            "contrast_low": 10,
            "contrast_high": 25,
            "angle_start": 0.0,
            "angle_extent": 0.0,
            "angle_step": 5.0,
            "use_polarity": 0,
        },
        {
            "subpixel": 2,
            "scale_min": 1.0,
            "scale_max": 1.0,
            "minScore": 0.4,
            "maxOverLap": 0.4,
            "greedness": 0.8,
            "numMatches": 1,
        },
    )
    assert len(results) == 1
