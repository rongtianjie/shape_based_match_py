"""Gradient-orientation based 2-D shape template matching.

The implementation follows the main ideas behind HALCON shape models and
LINE-MOD: a template is represented by a small, spatially distributed set of
edge points and their (polarity independent) gradient orientations.  Matching
is performed from coarse to fine over position, rotation, and scale.

Only :func:`get_model_shape` and :func:`get_matched_result` are public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import logging
import math
from time import perf_counter
from typing import Any, Callable, Mapping, TypeVar

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = ["get_model_shape", "get_matched_result"]

LOGGER = logging.getLogger(__name__)

_PAT_DEFAULTS: dict[str, int | float] = {
    "contrast_low": 3,
    "contrast_high": 5,
    "angle_start": -5.0,
    "angle_extent": 10.0,
    "num_levels": 1,
}
_MATCH_DEFAULTS: dict[str, int | float] = {
    "numMatches": 5,
    "minScore": 0.15,
    "scale_min": 1.0,
    "scale_max": 1.0,
}

_NUM_ORIENTATIONS = 8
_MAX_FEATURES = 256
_MIN_FEATURES = 8
_COARSE_ANGLE_STEP = 5.0
_FINE_ANGLE_STEP = 1.0
_COARSE_SCALE_STEP = 0.05
_FINE_SCALE_STEP = 0.01
_NMS_IOU = 0.5

_AUTO_CANNY_LOW_RATIO = 0.65
_AUTO_CANNY_HIGH_RATIO = 1.30

_FG_BORDER_FRACTION = 0.07
_FG_BORDER_MIN = 3
_FG_BORDER_MAX = 20
_FG_MAX_SIGMA = 20.0
_FG_MIN_SIGMA = 6.0
_FG_Z_THRESHOLD = 4.0
_FG_MIN_COMPONENT_AREA_FRAC = 0.005
_FG_MAX_COMPONENT_AREA_FRAC = 0.9
_FG_MAX_UNION_AREA_FRAC = 0.85
_FG_CENTRAL_FRAC = 0.4
_FG_BAND_RADIUS_FRACTION = 0.05
_FG_BAND_RADIUS_MIN = 3
_FG_BAND_RADIUS_MAX = 15
_FG_CORE_KERNEL = np.ones((5, 5), np.uint8)

_APPEARANCE_MIN_SCORE = 0.3
_APPEARANCE_MASK_MIN_PIXELS = 16

FloatImage = NDArray[np.float32]
UInt8Image = NDArray[np.uint8]
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


@dataclass(frozen=True)
class _PatternConfig:
    contrast_low: int
    contrast_high: int
    angle_start: float
    angle_extent: float
    num_levels: int
    auto_contrast: bool = False


@dataclass(frozen=True)
class _MatchConfig:
    num_matches: int
    min_score: float
    scale_min: float
    scale_max: float


@dataclass(frozen=True)
class _ModelFeatures:
    offsets: FloatImage  # (N, 2), x/y relative to geometric image centre
    unit_gradients: FloatImage  # (N, 2), gx/gy
    points: NDArray[np.int32]  # (N, 2), x/y in the model image
    labels: UInt8Image
    width: int
    height: int
    template_gray: UInt8Image
    appearance_mask: UInt8Image | None


@dataclass(frozen=True)
class _PoseKernel:
    kernels: tuple[FloatImage | None, ...]
    feature_count: int
    anchor_x: int
    anchor_y: int
    width: int
    height: int


@dataclass(frozen=True)
class _Candidate:
    cx: float
    cy: float
    score: float
    angle: float
    scale: float


def _as_mapping(value: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    number = _number(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _parse_pattern_config(config: Mapping[str, Any] | None) -> _PatternConfig:
    supplied = _as_mapping(config, "pat_config")
    unknown = set(supplied) - set(_PAT_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown pat_config keys: {', '.join(sorted(unknown))}")
    auto_contrast = "contrast_low" not in supplied and "contrast_high" not in supplied
    values = {**_PAT_DEFAULTS, **supplied}

    contrast_low = _integer(values["contrast_low"], "contrast_low")
    contrast_high = _integer(values["contrast_high"], "contrast_high")
    angle_start = _number(values["angle_start"], "angle_start")
    angle_extent = _number(values["angle_extent"], "angle_extent")
    num_levels = _integer(values["num_levels"], "num_levels")

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

    return _PatternConfig(contrast_low, contrast_high, angle_start, angle_extent, num_levels, auto_contrast)


def _parse_match_config(config: Mapping[str, Any] | None) -> _MatchConfig:
    supplied = _as_mapping(config, "match_config")
    unknown = set(supplied) - set(_MATCH_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown match_config keys: {', '.join(sorted(unknown))}")
    values = {**_MATCH_DEFAULTS, **supplied}

    num_matches = _integer(values["numMatches"], "numMatches")
    min_score = _number(values["minScore"], "minScore")
    scale_min = _number(values["scale_min"], "scale_min")
    scale_max = _number(values["scale_max"], "scale_max")

    if num_matches < 1:
        raise ValueError("numMatches must be at least 1")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("minScore must be between 0 and 1")
    if scale_min <= 0.0 or scale_max <= 0.0:
        raise ValueError("scale_min and scale_max must be greater than 0")
    if scale_min > scale_max:
        raise ValueError("scale_min must be less than or equal to scale_max")

    return _MatchConfig(num_matches, min_score, scale_min, scale_max)


def _validate_image(image: Any, name: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if image.size == 0:
        raise ValueError(f"{name} must not be empty")
    if image.ndim == 2:
        pass
    elif image.ndim == 3 and image.shape[2] in (1, 3, 4):
        pass
    else:
        raise ValueError(f"{name} must be a grayscale, BGR, or BGRA image")
    if not np.issubdtype(image.dtype, np.number) or np.issubdtype(image.dtype, np.complexfloating):
        raise TypeError(f"{name} must have a real numeric dtype")
    if np.issubdtype(image.dtype, np.floating) and not np.isfinite(image).all():
        raise ValueError(f"{name} must contain only finite values")
    return image


def _to_uint8(image: np.ndarray) -> UInt8Image:
    if image.dtype == np.uint8:
        return image.copy()
    return np.clip(image, 0, 255).astype(np.uint8)


def _to_gray(image: np.ndarray) -> UInt8Image:
    converted = _to_uint8(image)
    if converted.ndim == 2:
        return converted
    if converted.shape[2] == 1:
        return converted[:, :, 0]
    if converted.shape[2] == 3:
        return cv2.cvtColor(converted, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(converted, cv2.COLOR_BGRA2GRAY)


def _to_bgr(image: np.ndarray) -> UInt8Image:
    converted = _to_uint8(image)
    if converted.ndim == 2:
        return cv2.cvtColor(converted, cv2.COLOR_GRAY2BGR)
    if converted.shape[2] == 1:
        return cv2.cvtColor(converted[:, :, 0], cv2.COLOR_GRAY2BGR)
    if converted.shape[2] == 4:
        return cv2.cvtColor(converted, cv2.COLOR_BGRA2BGR)
    return converted


def _gradient_fields(gray: UInt8Image) -> tuple[FloatImage, FloatImage, FloatImage, FloatImage]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    safe = np.maximum(magnitude, np.float32(1e-6))
    return gx, gy, magnitude, safe


def _quantize_orientations(gx: FloatImage, gy: FloatImage) -> UInt8Image:
    # Gradient direction is undirected: theta and theta + pi share a label.
    angles = np.mod(np.arctan2(gy, gx), np.pi)
    labels = np.floor((angles + np.pi / (2 * _NUM_ORIENTATIONS)) * (_NUM_ORIENTATIONS / np.pi))
    return np.mod(labels.astype(np.int16), _NUM_ORIENTATIONS).astype(np.uint8)


def _auto_canny_thresholds(magnitude: FloatImage) -> tuple[int, int]:
    nonzero = magnitude[magnitude > 1.0]
    if nonzero.size < 16:
        return int(_PAT_DEFAULTS["contrast_low"]), int(_PAT_DEFAULTS["contrast_high"])
    median = float(np.median(nonzero))
    low = int(np.clip(round(_AUTO_CANNY_LOW_RATIO * median), 1, 250))
    high = int(np.clip(round(_AUTO_CANNY_HIGH_RATIO * median), low + 1, 255))
    return low, high


def _consistent_edges(gray: UInt8Image, config: _PatternConfig) -> tuple[UInt8Image, FloatImage, FloatImage, FloatImage]:
    gx, gy, magnitude, _ = _gradient_fields(gray)
    if config.auto_contrast:
        low, high = _auto_canny_thresholds(magnitude)
    else:
        low, high = config.contrast_low, config.contrast_high
    edges = cv2.Canny(gray, low, high, apertureSize=3, L2gradient=True) > 0
    labels = _quantize_orientations(gx, gy)

    # LINE-MOD keeps orientations supported by their neighbourhood.  Two votes
    # are sufficient for one-pixel-wide Canny contours while removing speckles.
    consistent = np.zeros_like(edges)
    for label in range(_NUM_ORIENTATIONS):
        members = (edges & (labels == label)).astype(np.uint8)
        votes = cv2.boxFilter(members, cv2.CV_16U, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        consistent |= (members > 0) & (votes >= 2)
    if int(np.count_nonzero(consistent)) < _MIN_FEATURES:
        consistent = edges
    return consistent.astype(np.uint8), gx, gy, magnitude


def _select_scattered(points_xy: NDArray[np.int32], strengths: FloatImage, limit: int) -> NDArray[np.int32]:
    if len(points_xy) <= limit:
        return np.arange(len(points_xy), dtype=np.int32)

    order = np.argsort(strengths)[::-1]
    area = max(1.0, float(np.ptp(points_xy[:, 0]) + 1) * float(np.ptp(points_xy[:, 1]) + 1))
    initial_distance = max(2.0, 0.55 * math.sqrt(area / limit))

    best: list[int] = []
    for distance in (initial_distance, initial_distance * 0.75, initial_distance * 0.5, 1.0):
        selected: list[int] = []
        selected_points = np.empty((limit, 2), dtype=np.float32)
        distance_sq = distance * distance
        for index in order:
            point = points_xy[index]
            if selected:
                delta = selected_points[: len(selected)] - point
                if np.any(np.einsum("ij,ij->i", delta, delta) < distance_sq):
                    continue
            selected_points[len(selected)] = point
            selected.append(int(index))
            if len(selected) == limit:
                break
        if len(selected) > len(best):
            best = selected
        if len(best) == limit:
            break
    return np.asarray(best, dtype=np.int32)


def _segment_foreground(gray: UInt8Image, bgr: UInt8Image) -> UInt8Image | None:
    """Detect a centrally-located foreground blob using border-ring background stats.

    Returns a 0/1 uint8 mask, or ``None`` when the border doesn't approximate a
    uniform background (template already tightly crops the target) or no
    plausible blob is found -- callers must fall back to whole-image behaviour.
    """
    height, width = gray.shape
    border = int(np.clip(round(_FG_BORDER_FRACTION * min(height, width)), _FG_BORDER_MIN, _FG_BORDER_MAX))
    if height <= 2 * border or width <= 2 * border:
        return None

    ring_mask = np.zeros((height, width), dtype=bool)
    ring_mask[:border, :] = True
    ring_mask[-border:, :] = True
    ring_mask[:, :border] = True
    ring_mask[:, -border:] = True

    bgr_f = bgr.astype(np.float32)
    z_max = np.zeros((height, width), dtype=np.float32)
    for channel in range(3):
        values = bgr_f[:, :, channel]
        ring_values = values[ring_mask]
        median = float(np.median(ring_values))
        sigma = 1.4826 * float(np.median(np.abs(ring_values - median)))
        if sigma > _FG_MAX_SIGMA:
            return None
        sigma = max(sigma, _FG_MIN_SIGMA)
        np.maximum(z_max, np.abs(values - median) / sigma, out=z_max)

    candidate = (z_max > _FG_Z_THRESHOLD).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if num_labels <= 1:
        return None

    image_area = float(height * width)
    centre_x, centre_y = (width - 1) / 2.0, (height - 1) / 2.0
    max_dist_x = _FG_CENTRAL_FRAC * width
    max_dist_y = _FG_CENTRAL_FRAC * height

    union = np.zeros((height, width), dtype=np.uint8)
    for label in range(1, num_labels):
        left, top, comp_w, comp_h, area = stats[label]
        area_frac = area / image_area
        if not (_FG_MIN_COMPONENT_AREA_FRAC <= area_frac <= _FG_MAX_COMPONENT_AREA_FRAC):
            continue
        centroid_x, centroid_y = centroids[label]
        if abs(centroid_x - centre_x) > max_dist_x or abs(centroid_y - centre_y) > max_dist_y:
            continue
        touches_border = left <= 0 or top <= 0 or left + comp_w >= width or top + comp_h >= height
        if touches_border:
            # A hairline bridge (e.g. a repeated watermark stroke) can connect an
            # otherwise-central blob to the border under 8-connectivity. Keep the
            # component only if a solid core survives a stronger erosion and that
            # core itself stays clear of the border -- a thin tendril alone won't.
            component_mask = (labels_img == label).astype(np.uint8)
            core = cv2.erode(component_mask, _FG_CORE_KERNEL)
            if not np.any(core):
                continue
            core_ys, core_xs = np.nonzero(core)
            if core_xs.min() <= 0 or core_ys.min() <= 0 or core_xs.max() >= width - 1 or core_ys.max() >= height - 1:
                continue
        union[labels_img == label] = 1

    if not np.any(union):
        return None
    if np.count_nonzero(union) / image_area > _FG_MAX_UNION_AREA_FRAC:
        return None
    return union


def _extract_features(gray: UInt8Image, config: _PatternConfig, bgr: UInt8Image) -> _ModelFeatures | None:
    edge_mask, gx, gy, magnitude = _consistent_edges(gray, config)
    height, width = gray.shape

    appearance_mask: UInt8Image | None = None
    blob = _segment_foreground(gray, bgr)
    if blob is not None:
        radius = int(np.clip(round(_FG_BAND_RADIUS_FRACTION * min(height, width)), _FG_BAND_RADIUS_MIN, _FG_BAND_RADIUS_MAX))
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        dilated = cv2.dilate(blob, kernel) > 0
        eroded = cv2.erode(blob, kernel) > 0
        band = (dilated & ~eroded).astype(np.uint8)
        banded_edges = edge_mask & band
        if int(np.count_nonzero(banded_edges)) >= _MIN_FEATURES:
            edge_mask = banded_edges
        appearance_mask = dilated.astype(np.uint8)

    ys, xs = np.nonzero(edge_mask)
    if len(xs) < _MIN_FEATURES:
        return None

    points = np.column_stack((xs, ys)).astype(np.int32)
    strengths = magnitude[ys, xs]
    chosen = _select_scattered(points, strengths, _MAX_FEATURES)
    points = points[chosen]
    xs = points[:, 0]
    ys = points[:, 1]
    magnitudes = np.maximum(magnitude[ys, xs], np.float32(1e-6))
    gradients = np.column_stack((gx[ys, xs] / magnitudes, gy[ys, xs] / magnitudes)).astype(np.float32)
    labels = _quantize_orientations(gradients[:, 0], gradients[:, 1])

    centre = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32)
    offsets = points.astype(np.float32) - centre
    return _ModelFeatures(offsets, gradients, points, labels, width, height, gray, appearance_mask)


def _orientation_response_maps(gray: UInt8Image, config: _PatternConfig) -> FloatImage:
    edge_mask, gx, gy, _ = _consistent_edges(gray, config)
    source_labels = _quantize_orientations(gx, gy)
    responses = np.zeros((_NUM_ORIENTATIONS, gray.shape[0], gray.shape[1]), dtype=np.float32)

    bins = np.arange(_NUM_ORIENTATIONS, dtype=np.float32)
    for template_label in range(_NUM_ORIENTATIONS):
        differences = np.abs(bins - template_label)
        differences = np.minimum(differences, _NUM_ORIENTATIONS - differences)
        lookup = np.abs(np.cos(differences * (np.pi / _NUM_ORIENTATIONS))).astype(np.float32)
        response = lookup[source_labels] * edge_mask
        responses[template_label] = cv2.dilate(response, np.ones((3, 3), np.uint8))
    return responses


def _rotation_matrix(angle: float, scale: float = 1.0) -> FloatImage:
    radians = math.radians(angle)
    cosine = math.cos(radians) * scale
    sine = math.sin(radians) * scale
    # Same visual convention as cv2.getRotationMatrix2D: positive is CCW.
    return np.asarray(((cosine, sine), (-sine, cosine)), dtype=np.float32)


def _transformed_geometry(
    features: _ModelFeatures, angle: float, scale: float
) -> tuple[FloatImage, FloatImage, UInt8Image, FloatImage]:
    spatial_transform = _rotation_matrix(angle, scale)
    direction_transform = _rotation_matrix(angle)
    offsets = features.offsets @ spatial_transform.T
    gradients = features.unit_gradients @ direction_transform.T
    labels = _quantize_orientations(gradients[:, 0], gradients[:, 1])

    half_width = (features.width - 1) / 2.0
    half_height = (features.height - 1) / 2.0
    corners = np.asarray(
        ((-half_width, -half_height), (half_width, -half_height), (half_width, half_height), (-half_width, half_height)),
        dtype=np.float32,
    )
    corners = corners @ spatial_transform.T
    return offsets, gradients, labels, corners


def _build_pose_kernel(features: _ModelFeatures, angle: float, scale: float) -> _PoseKernel:
    offsets, _, labels, corners = _transformed_geometry(features, angle, scale)
    rounded_offsets = np.rint(offsets).astype(np.int32)
    min_x = int(math.floor(float(np.min(corners[:, 0]))))
    max_x = int(math.ceil(float(np.max(corners[:, 0]))))
    min_y = int(math.floor(float(np.min(corners[:, 1]))))
    max_y = int(math.ceil(float(np.max(corners[:, 1]))))
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    kernels: list[FloatImage | None] = []
    for label in range(_NUM_ORIENTATIONS):
        selected = rounded_offsets[labels == label]
        if len(selected) == 0:
            kernels.append(None)
            continue
        kernel = np.zeros((height, width), dtype=np.float32)
        px = selected[:, 0] - min_x
        py = selected[:, 1] - min_y
        valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        np.add.at(kernel, (py[valid], px[valid]), 1.0)
        kernels.append(kernel)
    return _PoseKernel(tuple(kernels), len(offsets), -min_x, -min_y, width, height)


def _pose_score_map(responses: FloatImage, kernel: _PoseKernel) -> FloatImage | None:
    image_height, image_width = responses.shape[1:]
    if kernel.width > image_width or kernel.height > image_height:
        return None
    result: FloatImage | None = None
    for label, template in enumerate(kernel.kernels):
        if template is None:
            continue
        partial = cv2.matchTemplate(responses[label], template, cv2.TM_CCORR)
        if result is None:
            result = partial
        else:
            result += partial
    if result is None:
        return None
    result /= float(kernel.feature_count)
    np.clip(result, 0.0, 1.0, out=result)
    return result


def _sample_interval(start: float, stop: float, step: float, cyclic: bool = False) -> list[float]:
    if math.isclose(start, stop, abs_tol=1e-9):
        return [float(start)]
    values: list[float] = []
    current = start
    effective_stop = stop - (1e-7 if cyclic else 0.0)
    while current <= effective_stop + 1e-9:
        values.append(float(current))
        current += step
    if not cyclic and not math.isclose(values[-1], stop, abs_tol=1e-7):
        values.append(float(stop))
    return values


def _coarse_angles(config: _PatternConfig) -> list[float]:
    if math.isclose(config.angle_extent, 0.0):
        return [config.angle_start]
    return _sample_interval(
        config.angle_start,
        config.angle_start + config.angle_extent,
        _COARSE_ANGLE_STEP,
        cyclic=math.isclose(config.angle_extent, 360.0),
    )


def _coarse_scales(config: _MatchConfig) -> list[float]:
    return _sample_interval(config.scale_min, config.scale_max, _COARSE_SCALE_STEP)


def _top_local_peaks(score_map: FloatImage, threshold: float, limit: int) -> list[tuple[float, int, int]]:
    if score_map.size == 0:
        return []
    local_maximum = score_map >= cv2.dilate(score_map, np.ones((5, 5), np.uint8)) - 1e-7
    ys, xs = np.nonzero(local_maximum & (score_map >= threshold))
    if len(xs) == 0:
        return []
    scores = score_map[ys, xs]
    if len(scores) > limit:
        chosen = np.argpartition(scores, -limit)[-limit:]
        xs, ys, scores = xs[chosen], ys[chosen], scores[chosen]
    order = np.argsort(scores)[::-1]
    return [(float(scores[i]), int(xs[i]), int(ys[i])) for i in order]


def _coarse_search(
    features: _ModelFeatures,
    responses: FloatImage,
    pattern: _PatternConfig,
    matching: _MatchConfig,
    image_factor: float,
) -> list[_Candidate]:
    per_pose_limit = max(4, matching.num_matches * 2)
    candidates: list[_Candidate] = []
    scaled_features = _ModelFeatures(
        features.offsets * image_factor,
        features.unit_gradients,
        features.points,
        features.labels,
        max(1, int(round(features.width * image_factor))),
        max(1, int(round(features.height * image_factor))),
        features.template_gray,
        features.appearance_mask,
    )

    coarse_threshold = max(0.02, matching.min_score * 0.8)
    for scale in _coarse_scales(matching):
        for angle in _coarse_angles(pattern):
            kernel = _build_pose_kernel(scaled_features, angle, scale)
            score_map = _pose_score_map(responses, kernel)
            if score_map is None:
                continue
            for score, left, top in _top_local_peaks(score_map, coarse_threshold, per_pose_limit):
                candidates.append(
                    _Candidate(
                        (left + kernel.anchor_x) / image_factor,
                        (top + kernel.anchor_y) / image_factor,
                        score,
                        angle,
                        scale,
                    )
                )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    global_limit = max(24, matching.num_matches * 10)
    return candidates[:global_limit]


def _angle_allowed(angle: float, config: _PatternConfig) -> bool:
    if math.isclose(config.angle_extent, 360.0):
        return True
    return config.angle_start - 1e-9 <= angle <= config.angle_start + config.angle_extent + 1e-9


def _canonical_angle(angle: float, config: _PatternConfig) -> float:
    if math.isclose(config.angle_extent, 360.0):
        return config.angle_start + ((angle - config.angle_start) % 360.0)
    return min(max(angle, config.angle_start), config.angle_start + config.angle_extent)


def _fine_pose_values(candidate: _Candidate, pattern: _PatternConfig, matching: _MatchConfig) -> tuple[list[float], list[float]]:
    angles: list[float] = []
    for angle in _sample_interval(candidate.angle - _COARSE_ANGLE_STEP, candidate.angle + _COARSE_ANGLE_STEP, _FINE_ANGLE_STEP):
        canonical = _canonical_angle(angle, pattern)
        if _angle_allowed(canonical, pattern) and not any(math.isclose(canonical, item, abs_tol=1e-7) for item in angles):
            angles.append(canonical)

    scale_start = max(matching.scale_min, candidate.scale - _COARSE_SCALE_STEP)
    scale_stop = min(matching.scale_max, candidate.scale + _COARSE_SCALE_STEP)
    scales = _sample_interval(scale_start, scale_stop, _FINE_SCALE_STEP)
    return angles, scales


def _valid_centres(
    centres: FloatImage, corners: FloatImage, image_width: int, image_height: int
) -> NDArray[np.bool_]:
    transformed = centres[:, None, :] + corners[None, :, :]
    return (
        (transformed[:, :, 0] >= 0.0).all(axis=1)
        & (transformed[:, :, 0] <= image_width - 1).all(axis=1)
        & (transformed[:, :, 1] >= 0.0).all(axis=1)
        & (transformed[:, :, 1] <= image_height - 1).all(axis=1)
    )


def _scores_at_centres(
    responses: FloatImage,
    offsets: FloatImage,
    labels: UInt8Image,
    centres: FloatImage,
) -> FloatImage:
    rounded_offsets = np.rint(offsets).astype(np.int32)
    rounded_centres = np.rint(centres).astype(np.int32)
    xs = rounded_centres[:, 0, None] + rounded_offsets[None, :, 0]
    ys = rounded_centres[:, 1, None] + rounded_offsets[None, :, 1]
    values = responses[labels[None, :], ys, xs]
    return values.mean(axis=1, dtype=np.float32)


def _refine_candidate(
    candidate: _Candidate,
    features: _ModelFeatures,
    responses: FloatImage,
    pattern: _PatternConfig,
    matching: _MatchConfig,
) -> _Candidate | None:
    radius = 4 if pattern.num_levels == 1 else 2
    base_x = int(round(candidate.cx))
    base_y = int(round(candidate.cy))
    grid_x, grid_y = np.meshgrid(
        np.arange(base_x - radius, base_x + radius + 1),
        np.arange(base_y - radius, base_y + radius + 1),
    )
    centres = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
    image_height, image_width = responses.shape[1:]
    best: _Candidate | None = None

    angles, scales = _fine_pose_values(candidate, pattern, matching)
    for scale in scales:
        for angle in angles:
            offsets, _, labels, corners = _transformed_geometry(features, angle, scale)
            valid = _valid_centres(centres, corners, image_width, image_height)
            if not np.any(valid):
                continue
            valid_centres = centres[valid]
            scores = _scores_at_centres(responses, offsets, labels, valid_centres)
            index = int(np.argmax(scores))
            score = float(scores[index])
            if best is None or score > best.score:
                centre = valid_centres[index]
                best = _Candidate(float(centre[0]), float(centre[1]), score, float(angle), float(scale))
    return best


def _candidate_polygon(candidate: _Candidate, features: _ModelFeatures) -> FloatImage:
    _, _, _, corners = _transformed_geometry(features, candidate.angle, candidate.scale)
    return corners + np.asarray((candidate.cx, candidate.cy), dtype=np.float32)


def _polygon_iou(first: FloatImage, second: FloatImage) -> float:
    area_first = abs(float(cv2.contourArea(first)))
    area_second = abs(float(cv2.contourArea(second)))
    if area_first <= 0.0 or area_second <= 0.0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = area_first + area_second - float(intersection)
    return float(intersection) / union if union > 0.0 else 0.0


def _resample_candidate_patch(source_gray: UInt8Image, features: _ModelFeatures, candidate: _Candidate) -> UInt8Image:
    """Resample the source region a candidate covers back into template space."""
    rotation = _rotation_matrix(candidate.angle, candidate.scale)
    template_centre = np.array([(features.width - 1) / 2.0, (features.height - 1) / 2.0], dtype=np.float32)
    candidate_centre = np.array([candidate.cx, candidate.cy], dtype=np.float32)
    translation = candidate_centre - rotation @ template_centre
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[:, :2] = rotation
    matrix[:, 2] = translation
    return cv2.warpAffine(
        source_gray,
        matrix,
        (features.width, features.height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _appearance_score(template_gray: UInt8Image, appearance_mask: UInt8Image | None, patch_gray: UInt8Image) -> float:
    """Foreground/background contrast-ratio consistency between the template and a resampled patch.

    Real-world illumination, JPEG drift and near-uniform fill colours make raw
    pixel-to-pixel correlation an unreliable appearance check even for genuinely
    correct matches (interior texture rarely repeats between two crops of the
    same physical mark). Instead this compares each image's own foreground-vs-
    background mean brightness gap: a genuine instance of the mark still shows a
    contrast blob shaped like the template even under different lighting, while
    an unrelated flat/textureless region does not. The ratio is symmetric and
    uses ``abs`` so polarity-inverted matches (deliberately supported elsewhere)
    score the same as direct matches. Returns 1.0 (neutral pass) when no
    reliable foreground mask is available for the template.
    """
    if appearance_mask is None:
        return 1.0
    mask = appearance_mask.astype(bool)
    background = ~mask
    if int(np.count_nonzero(mask)) < _APPEARANCE_MASK_MIN_PIXELS or int(np.count_nonzero(background)) < _APPEARANCE_MASK_MIN_PIXELS:
        return 1.0

    template_contrast = abs(
        float(template_gray[mask].astype(np.float32).mean()) - float(template_gray[background].astype(np.float32).mean())
    )
    patch_contrast = abs(
        float(patch_gray[mask].astype(np.float32).mean()) - float(patch_gray[background].astype(np.float32).mean())
    )
    if template_contrast <= 1e-3:
        return 1.0
    if patch_contrast <= 1e-3:
        return 0.0
    return min(template_contrast, patch_contrast) / max(template_contrast, patch_contrast)


def _nms(candidates: list[_Candidate], features: _ModelFeatures, limit: int) -> list[_Candidate]:
    selected: list[_Candidate] = []
    polygons: list[FloatImage] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        polygon = _candidate_polygon(candidate, features)
        if any(_polygon_iou(polygon, previous) > _NMS_IOU for previous in polygons):
            continue
        selected.append(candidate)
        polygons.append(polygon)
        if len(selected) == limit:
            break
    return selected


def _draw_model_features(image: np.ndarray, features: _ModelFeatures) -> UInt8Image:
    show = _to_bgr(image)
    colors = (
        (255, 80, 80),
        (255, 180, 60),
        (180, 255, 60),
        (60, 255, 120),
        (60, 255, 255),
        (60, 140, 255),
        (160, 60, 255),
        (255, 60, 180),
    )
    radius = 1 if min(features.width, features.height) < 160 else 2
    for (x, y), label in zip(features.points, features.labels, strict=True):
        cv2.circle(show, (int(x), int(y)), radius, colors[int(label)], -1, lineType=cv2.LINE_AA)
    centre = (int(round((features.width - 1) / 2)), int(round((features.height - 1) / 2)))
    cv2.drawMarker(show, centre, (0, 255, 255), cv2.MARKER_CROSS, 11, 1, cv2.LINE_AA)
    return show


def _draw_matches(image: np.ndarray, matches: list[_Candidate], features: _ModelFeatures) -> UInt8Image:
    show = _to_bgr(image)
    palette = ((0, 255, 0), (0, 180, 255), (255, 120, 0), (255, 0, 220), (0, 255, 255))
    for index, match in enumerate(matches):
        color = palette[index % len(palette)]
        polygon = np.rint(_candidate_polygon(match, features)).astype(np.int32)
        cv2.polylines(show, [polygon], True, color, 2, cv2.LINE_AA)
        centre = (int(round(match.cx)), int(round(match.cy)))
        cv2.drawMarker(show, centre, color, cv2.MARKER_CROSS, 13, 2, cv2.LINE_AA)
        label = f"{match.score:.3f}  {match.angle:.1f}deg  x{match.scale:.2f}"
        text_at = (max(0, centre[0] + 6), max(14, centre[1] - 6))
        cv2.putText(show, label, text_at, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return show


@count_time
def get_model_shape(model: np.ndarray, pat_config: Mapping[str, Any] = {}) -> np.ndarray | None:
    """Return a BGR visualization of the shape features extracted from *model*.

    A flat or otherwise unusable template returns ``None``. Invalid arguments
    raise ``TypeError`` or ``ValueError``.
    """

    pattern = _parse_pattern_config(pat_config)
    model_array = _validate_image(model, "model")
    features = _extract_features(_to_gray(model_array), pattern, _to_bgr(model_array))
    if features is None:
        LOGGER.warning("model feature extraction failed: fewer than %d usable edge points", _MIN_FEATURES)
        return None
    return _draw_model_features(model_array, features)


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

    pattern = _parse_pattern_config(pat_config)
    matching = _parse_match_config(match_config)
    model_array = _validate_image(model, "model")
    source_array = _validate_image(src, "src")
    model_gray = _to_gray(model_array)
    source_gray = _to_gray(source_array)
    features = _extract_features(model_gray, pattern, _to_bgr(model_array))
    if features is None:
        LOGGER.warning("matching skipped: model contains fewer than %d usable edge points", _MIN_FEATURES)
        return [], None

    if pattern.num_levels == 1:
        coarse_gray = cv2.pyrDown(source_gray)
        image_factor = 0.5
    else:
        coarse_gray = source_gray
        image_factor = 1.0

    coarse_responses = _orientation_response_maps(coarse_gray, pattern)
    coarse = _coarse_search(features, coarse_responses, pattern, matching, image_factor)
    if not coarse:
        return [], None

    full_responses = _orientation_response_maps(source_gray, pattern)
    refined: list[_Candidate] = []
    for candidate in coarse:
        result = _refine_candidate(candidate, features, full_responses, pattern, matching)
        if result is None or result.score < matching.min_score:
            continue
        patch = _resample_candidate_patch(source_gray, features, result)
        appearance = _appearance_score(features.template_gray, features.appearance_mask, patch)
        if appearance >= _APPEARANCE_MIN_SCORE:
            refined.append(result)
    if not refined:
        return [], None

    matches = _nms(refined, features, matching.num_matches)
    if not matches:
        return [], None

    result_rows = [
        [float(match.cx), float(match.cy), float(match.score), float(match.angle), float(match.scale)]
        for match in matches
    ]
    return result_rows, _draw_matches(source_array, matches, features)
