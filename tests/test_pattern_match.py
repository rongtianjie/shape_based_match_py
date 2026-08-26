from __future__ import annotations

import cv2
import json
import numpy as np
from pathlib import Path
import pytest

from pattern_match import get_matched_result, get_model_shape


def make_model() -> np.ndarray:
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
    height, width = model.shape[:2]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle, scale)
    matrix[0, 2] += centre[0] - (width - 1) / 2.0
    matrix[1, 2] += centre[1] - (height - 1) / 2.0
    transformed = cv2.warpAffine(model, matrix, (source.shape[1], source.shape[0]), flags=cv2.INTER_LINEAR)
    np.maximum(source, transformed, out=source)


STRICT_PATTERN = {
    "contrast_low": 10,
    "contrast_high": 25,
    "angle_start": 0.0,
    "angle_extent": 0.0,
    "num_levels": 1,
}


def assert_pose(result: list[float], centre: tuple[float, float], angle: float, scale: float) -> None:
    assert result[0] == pytest.approx(centre[0], abs=2.0)
    assert result[1] == pytest.approx(centre[1], abs=2.0)
    assert result[3] == pytest.approx(angle, abs=2.0)
    assert result[4] == pytest.approx(scale, abs=0.04)


def test_model_shape_is_bgr_and_does_not_mutate_inputs() -> None:
    model = make_model()
    original = model.copy()
    config = {"contrast_low": 10, "contrast_high": 25}
    original_config = config.copy()

    show = get_model_shape(model, config)

    assert show is not None
    assert show.shape == model.shape
    assert show.dtype == np.uint8
    assert np.array_equal(model, original)
    assert config == original_config


def test_flat_model_is_a_processing_failure() -> None:
    flat = np.zeros((50, 50), dtype=np.uint8)
    assert get_model_shape(flat) is None
    assert get_matched_result(flat, flat) == ([], None)


def test_invalid_config_and_image_inputs() -> None:
    model = make_model()
    with pytest.raises(ValueError, match="contrast_low must be less than contrast_high"):
        get_model_shape(model, {"contrast_low": 10, "contrast_high": 10})
    with pytest.raises(ValueError, match="scale_min must be less than or equal"):
        get_matched_result(model, model, match_config={"scale_min": 1.2, "scale_max": 0.8})
    with pytest.raises(ValueError, match="unknown pat_config keys"):
        get_model_shape(model, {"typo": 1})
    with pytest.raises(TypeError, match="numpy.ndarray"):
        get_model_shape([[0, 1], [2, 3]])


def test_translation_and_result_contract() -> None:
    model = make_model()
    source = np.zeros((260, 320, 3), dtype=np.uint8)
    centre = (189.0, 137.0)
    place_model(source, model, centre)

    results, show = get_matched_result(
        model,
        source,
        STRICT_PATTERN,
        {"numMatches": 1, "minScore": 0.5},
    )

    assert len(results) == 1
    assert len(results[0]) == 5
    assert 0.0 <= results[0][2] <= 1.0
    assert_pose(results[0], centre, 0.0, 1.0)
    assert show is not None and show.shape == source.shape


@pytest.mark.parametrize(
    ("angle", "scale", "num_levels"),
    [(13.0, 1.12, 1), (-12.0, 0.86, 0)],
)
def test_positive_and_negative_rotation_scale_noise_and_occlusion(
    angle: float, scale: float, num_levels: int
) -> None:
    rng = np.random.default_rng(17)
    model = make_model()
    source = rng.integers(0, 18, size=(320, 380, 3), dtype=np.uint8)
    centre = (214.0, 163.0)
    place_model(source, model, centre, angle=angle, scale=scale)
    cv2.rectangle(source, (208, 145), (229, 171), (0, 0, 0), -1)

    pattern = {
        "contrast_low": 10,
        "contrast_high": 25,
        "angle_start": -20.0,
        "angle_extent": 40.0,
        "num_levels": num_levels,
    }
    results, _ = get_matched_result(
        model,
        source,
        pattern,
        {"numMatches": 1, "minScore": 0.35, "scale_min": 0.8, "scale_max": 1.2},
    )

    assert results
    assert_pose(results[0], centre, angle, scale)


def test_grayscale_and_bgra_inputs_with_partial_defaults() -> None:
    model = cv2.cvtColor(make_model(), cv2.COLOR_BGR2GRAY)
    source = np.zeros((220, 280), dtype=np.uint8)
    centre = (151.0, 112.0)
    color_source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    place_model(color_source, cv2.cvtColor(model, cv2.COLOR_GRAY2BGR), centre)
    source_bgra = cv2.cvtColor(color_source, cv2.COLOR_BGR2BGRA)

    results, show = get_matched_result(
        model,
        source_bgra,
        {"contrast_low": 10, "contrast_high": 25, "angle_start": 0.0, "angle_extent": 0.0},
        {"numMatches": 1, "minScore": 0.5},
    )

    assert results
    assert_pose(results[0], centre, 0.0, 1.0)
    assert show is not None and show.shape == (220, 280, 3)


def test_multiple_matches_are_sorted_and_deduplicated() -> None:
    model = make_model()
    source = np.zeros((340, 440, 3), dtype=np.uint8)
    expected = [(106.0, 102.0), (330.0, 239.0)]
    place_model(source, model, expected[0])
    place_model(source, model, expected[1])

    results, _ = get_matched_result(
        model,
        source,
        STRICT_PATTERN,
        {"numMatches": 2, "minScore": 0.5},
    )

    assert len(results) == 2
    assert results[0][2] >= results[1][2]
    returned = [(item[0], item[1]) for item in results]
    for centre in expected:
        assert min(math_distance(centre, item) for item in returned) <= 2.0


def math_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return float(np.hypot(first[0] - second[0], first[1] - second[1]))


def test_contrast_inversion_and_no_match() -> None:
    model = make_model()
    inverted = 255 - model
    source = np.zeros((240, 300, 3), dtype=np.uint8)
    centre = (145.0, 119.0)
    place_model(source, inverted, centre)

    results, _ = get_matched_result(
        model,
        source,
        STRICT_PATTERN,
        {"numMatches": 1, "minScore": 0.45},
    )
    assert results
    assert_pose(results[0], centre, 0.0, 1.0)

    blank = np.zeros_like(source)
    assert get_matched_result(model, blank, STRICT_PATTERN, {"minScore": 0.2}) == ([], None)


def test_public_noise_and_contrast_sample() -> None:
    data_dir = Path(__file__).parent / "data" / "meiqua_case2"
    model = cv2.imread(str(data_dir / "model.png"), cv2.IMREAD_COLOR)
    source = cv2.imread(str(data_dir / "source.png"), cv2.IMREAD_COLOR)
    ground_truth = json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8"))

    results, show = get_matched_result(
        model,
        source,
        {
            "contrast_low": 10,
            "contrast_high": 30,
            "angle_start": 0.0,
            "angle_extent": 0.0,
            "num_levels": 1,
        },
        {"numMatches": 8, "minScore": 0.15, "scale_min": 0.8, "scale_max": 1.2},
    )

    assert show is not None
    returned = [(item[0], item[1]) for item in results]
    for centre in ground_truth["centres"]:
        assert min(math_distance(tuple(centre), item) for item in returned) <= ground_truth["tolerance_px"]
