from __future__ import annotations

import cv2
import numpy as np
import pytest

from pattern_match import get_matched_result


def make_model() -> np.ndarray:
    """Create synthetic test model with rectangle, line, and circle shapes."""
    image = np.zeros((72, 88), dtype=np.uint8)
    cv2.rectangle(image, (9, 8), (72, 56), 220, 3)
    cv2.line(image, (16, 48), (62, 16), 255, 4)
    cv2.circle(image, (66, 48), 8, 180, 3)
    cv2.rectangle(image, (9, 8), (30, 21), 0, -1)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def place_model(
    source: np.ndarray,
    model: np.ndarray,
    centre: tuple[float, float],
    angle: float = 0.0,
    scale: float = 1.0,
) -> None:
    """Warp model to specified pose and superimpose onto source image."""
    height, width = model.shape[:2]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle, scale)
    matrix[0, 2] += centre[0] - (width - 1) / 2.0
    matrix[1, 2] += centre[1] - (height - 1) / 2.0
    transformed = cv2.warpAffine(
        model, matrix, (source.shape[1], source.shape[0]), flags=cv2.INTER_LINEAR
    )
    np.maximum(source, transformed, out=source)


@pytest.mark.parametrize(
    ("name", "centre"),
    [
        ("left_boundary", (15.0, 137.0)),
        ("right_boundary", (305.0, 137.0)),
        ("top_boundary", (160.0, 15.0)),
        ("bottom_boundary", (160.0, 245.0)),
        ("top_left_corner", (25.0, 22.0)),
        ("bottom_right_corner", (295.0, 238.0)),
    ],
)
def test_partial_fov_boundary_matching(name: str, centre: tuple[float, float]) -> None:
    """Verify that template marks partially outside the FOV on all 4 borders and corners match accurately."""
    model = make_model()
    source = np.zeros((260, 320, 3), dtype=np.uint8)
    place_model(source, model, centre)

    results, show = get_matched_result(
        model,
        source,
        {
            "contrast_low": 10,
            "contrast_high": 25,
            "angle_start": 0.0,
            "angle_extent": 0.0,
            "num_levels": 1,
        },
        {"numMatches": 1, "minScore": 0.5},
    )

    assert len(results) == 1, f"Failed matching for {name}"
    cx, cy, score, angle, scale = results[0]
    assert cx == pytest.approx(centre[0], abs=2.5)
    assert cy == pytest.approx(centre[1], abs=2.5)
    assert score >= 0.80
    assert angle == pytest.approx(0.0, abs=2.0)
    assert scale == pytest.approx(1.0, abs=0.06)
    assert show is not None and show.shape == source.shape


def test_partial_fov_with_rotation_and_scale() -> None:
    """Verify partial FOV matching when the mark has non-zero rotation and scale."""
    model = make_model()
    source = np.zeros((260, 320, 3), dtype=np.uint8)
    centre = (20.0, 140.0)
    angle = 15.0
    scale = 1.08
    place_model(source, model, centre, angle=angle, scale=scale)

    results, show = get_matched_result(
        model,
        source,
        {
            "contrast_low": 10,
            "contrast_high": 25,
            "angle_start": -20.0,
            "angle_extent": 40.0,
            "num_levels": 1,
        },
        {"numMatches": 1, "minScore": 0.4, "scale_min": 0.8, "scale_max": 1.2},
    )

    assert len(results) == 1
    cx, cy, score, matched_angle, matched_scale = results[0]
    assert cx == pytest.approx(centre[0], abs=2.5)
    assert cy == pytest.approx(centre[1], abs=2.5)
    assert matched_angle == pytest.approx(angle, abs=2.0)
    assert matched_scale == pytest.approx(scale, abs=0.06)
    assert score >= 0.80
    assert show is not None


def test_partial_fov_multiple_occurrences() -> None:
    """Verify matching multiple occurrences where one is inside FOV and another is partially outside."""
    model = make_model()
    source = np.zeros((300, 400, 3), dtype=np.uint8)
    centre1 = (200.0, 150.0)
    centre2 = (20.0, 150.0)
    place_model(source, model, centre1)
    place_model(source, model, centre2)

    results, show = get_matched_result(
        model,
        source,
        {
            "contrast_low": 10,
            "contrast_high": 25,
            "angle_start": 0.0,
            "angle_extent": 0.0,
            "num_levels": 1,
        },
        {"numMatches": 2, "minScore": 0.5},
    )

    assert len(results) == 2
    returned_centres = [(r[0], r[1]) for r in results]
    for expected in (centre1, centre2):
        assert min(float(np.hypot(expected[0] - r[0], expected[1] - r[1])) for r in returned_centres) <= 2.5
    assert show is not None


@pytest.mark.parametrize(
    ("name", "centre"),
    [
        ("extreme_out_of_fov", (-40.0, 150.0)),
        ("left_under_50_percent", (-10.0, 137.0)),
        ("top_under_50_percent", (160.0, -10.0)),
        ("corner_under_50_percent", (12.0, 12.0)),
    ],
)
def test_partial_fov_rejects_insufficient_visibility(name: str, centre: tuple[float, float]) -> None:
    """Verify that marks whose remaining area in FOV is <= 50% are NOT matched."""
    model = make_model()
    source = np.zeros((260, 320, 3), dtype=np.uint8)
    place_model(source, model, centre)

    results, show = get_matched_result(
        model,
        source,
        {
            "contrast_low": 10,
            "contrast_high": 25,
            "angle_start": 0.0,
            "angle_extent": 0.0,
            "num_levels": 1,
        },
        {"numMatches": 1, "minScore": 0.5, "scale_min": 1.0, "scale_max": 1.0},
    )

    assert results == [], f"Should not match mark with <= 50% remaining area for {name}"

