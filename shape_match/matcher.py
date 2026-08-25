"""Hierarchical coarse-to-fine search, refinement, appearance verification, and NMS."""

from __future__ import annotations

import math
import cv2
import numpy as np
from numpy.typing import NDArray

from shape_match.transforms import (
    build_pose_kernel,
    candidate_polygon,
    polygon_iou,
    rotation_matrix,
    transformed_geometry,
)
from shape_match.types import (
    APPEARANCE_MASK_MIN_PIXELS,
    APPEARANCE_MIN_SCORE,
    COARSE_ANGLE_STEP,
    COARSE_SCALE_STEP,
    FINE_ANGLE_STEP,
    FINE_SCALE_STEP,
    NMS_IOU,
    Candidate,
    FloatImage,
    MatchConfig,
    ModelFeatures,
    PatternConfig,
    PoseKernel,
    UInt8Image,
)


def pose_score_map(responses: FloatImage, kernel: PoseKernel) -> FloatImage | None:
    """Correlate orientation response maps with a multi-channel pose kernel and return score map."""
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


def sample_interval(start: float, stop: float, step: float, cyclic: bool = False) -> list[float]:
    """Sample numbers in [start, stop] with step; cyclic handles full 360-deg wrapping without duplicate endpoints."""
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


def coarse_angles(config: PatternConfig) -> list[float]:
    """Generate coarse search angle sequence based on pattern configuration."""
    if math.isclose(config.angle_extent, 0.0):
        return [config.angle_start]
    return sample_interval(
        config.angle_start,
        config.angle_start + config.angle_extent,
        COARSE_ANGLE_STEP,
        cyclic=math.isclose(config.angle_extent, 360.0),
    )


def coarse_scales(config: MatchConfig) -> list[float]:
    """Generate coarse search scale sequence based on match configuration."""
    return sample_interval(config.scale_min, config.scale_max, COARSE_SCALE_STEP)


def top_local_peaks(score_map: FloatImage, threshold: float, limit: int) -> list[tuple[float, int, int]]:
    """Extract top local maxima from score map exceeding the threshold."""
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


def coarse_search(
    features: ModelFeatures,
    responses: FloatImage,
    pattern: PatternConfig,
    matching: MatchConfig,
    image_factor: float,
) -> list[Candidate]:
    """Perform coarse grid search over discrete poses and extract candidate peak locations."""
    per_pose_limit = max(4, matching.num_matches * 2)
    candidates: list[Candidate] = []
    scaled_features = ModelFeatures(
        offsets=features.offsets * image_factor,
        unit_gradients=features.unit_gradients,
        points=features.points,
        labels=features.labels,
        width=max(1, int(round(features.width * image_factor))),
        height=max(1, int(round(features.height * image_factor))),
        template_gray=features.template_gray,
        appearance_mask=features.appearance_mask,
    )

    coarse_threshold = max(0.02, matching.min_score * 0.8)
    scales = coarse_scales(matching)
    angles = coarse_angles(pattern)
    
    try:
        import torch
        from shape_match.torch_matcher import coarse_search_pytorch, HAS_TORCH
        if HAS_TORCH and torch.cuda.is_available():
            return coarse_search_pytorch(
                scaled_features, responses, pattern, matching, image_factor, scales, angles
            )
    except ImportError:
        pass

    for scale in scales:
        for angle in angles:
            kernel = build_pose_kernel(scaled_features, angle, scale)
            score_map = pose_score_map(responses, kernel)
            if score_map is None:
                continue
            for score, left, top in top_local_peaks(score_map, coarse_threshold, per_pose_limit):
                candidates.append(
                    Candidate(
                        cx=(left + kernel.anchor_x) / image_factor,
                        cy=(top + kernel.anchor_y) / image_factor,
                        score=score,
                        angle=angle,
                        scale=scale,
                    )
                )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    global_limit = max(24, matching.num_matches * 10)
    return candidates[:global_limit]


def angle_allowed(angle: float, config: PatternConfig) -> bool:
    """Check if an angle falls within the allowed search range."""
    if math.isclose(config.angle_extent, 360.0):
        return True
    return config.angle_start - 1e-9 <= angle <= config.angle_start + config.angle_extent + 1e-9


def canonical_angle(angle: float, config: PatternConfig) -> float:
    """Map angle to canonical representation within allowed angle bounds."""
    if math.isclose(config.angle_extent, 360.0):
        return config.angle_start + ((angle - config.angle_start) % 360.0)
    return min(max(angle, config.angle_start), config.angle_start + config.angle_extent)


def fine_pose_values(
    candidate: Candidate, pattern: PatternConfig, matching: MatchConfig
) -> tuple[list[float], list[float]]:
    """Generate fine angle and scale sample values surrounding a coarse candidate pose."""
    angles: list[float] = []
    for angle in sample_interval(
        candidate.angle - COARSE_ANGLE_STEP,
        candidate.angle + COARSE_ANGLE_STEP,
        FINE_ANGLE_STEP,
    ):
        canonical = canonical_angle(angle, pattern)
        if angle_allowed(canonical, pattern) and not any(math.isclose(canonical, item, abs_tol=1e-7) for item in angles):
            angles.append(canonical)

    scale_start = max(matching.scale_min, candidate.scale - COARSE_SCALE_STEP)
    scale_stop = min(matching.scale_max, candidate.scale + COARSE_SCALE_STEP)
    scales = sample_interval(scale_start, scale_stop, FINE_SCALE_STEP)
    return angles, scales


def valid_centres(
    centres: FloatImage, corners: FloatImage, image_width: int, image_height: int
) -> NDArray[np.bool_]:
    """Filter candidate centers where all 4 transformed template corners remain strictly within image bounds."""
    transformed = centres[:, None, :] + corners[None, :, :]
    return (
        (transformed[:, :, 0] >= 0.0).all(axis=1)
        & (transformed[:, :, 0] <= image_width - 1).all(axis=1)
        & (transformed[:, :, 1] >= 0.0).all(axis=1)
        & (transformed[:, :, 1] <= image_height - 1).all(axis=1)
    )


def scores_at_centres(
    responses: FloatImage,
    offsets: FloatImage,
    labels: UInt8Image,
    centres: FloatImage,
) -> FloatImage:
    """Compute average similarity score across all feature points at given center coordinates."""
    rounded_offsets = np.rint(offsets).astype(np.int32)
    rounded_centres = np.rint(centres).astype(np.int32)
    xs = rounded_centres[:, 0, None] + rounded_offsets[None, :, 0]
    ys = rounded_centres[:, 1, None] + rounded_offsets[None, :, 1]
    values = responses[labels[None, :], ys, xs]
    return values.mean(axis=1, dtype=np.float32)


def refine_candidate(
    candidate: Candidate,
    features: ModelFeatures,
    responses: FloatImage,
    pattern: PatternConfig,
    matching: MatchConfig,
) -> Candidate | None:
    """Refine candidate pose by searching a local spatial window and sub-degree/sub-scale neighborhood."""
    radius = 4 if pattern.num_levels == 1 else 2
    base_x = int(round(candidate.cx))
    base_y = int(round(candidate.cy))
    grid_x, grid_y = np.meshgrid(
        np.arange(base_x - radius, base_x + radius + 1),
        np.arange(base_y - radius, base_y + radius + 1),
    )
    centres = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
    image_height, image_width = responses.shape[1:]
    best: Candidate | None = None

    angles, scales = fine_pose_values(candidate, pattern, matching)
    for scale in scales:
        for angle in angles:
            offsets, _, labels, corners = transformed_geometry(features, angle, scale)
            valid = valid_centres(centres, corners, image_width, image_height)
            if not np.any(valid):
                continue
            valid_centres_arr = centres[valid]
            scores = scores_at_centres(responses, offsets, labels, valid_centres_arr)
            index = int(np.argmax(scores))
            score = float(scores[index])
            if best is None or score > best.score:
                centre = valid_centres_arr[index]
                best = Candidate(
                    cx=float(centre[0]),
                    cy=float(centre[1]),
                    score=score,
                    angle=float(angle),
                    scale=float(scale),
                )
    return best


def resample_candidate_patch(
    source_gray: UInt8Image, features: ModelFeatures, candidate: Candidate
) -> UInt8Image:
    """Resample the source image region covered by candidate back into template canonical coordinate frame."""
    rotation = rotation_matrix(candidate.angle, candidate.scale)
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


def appearance_score(
    template_gray: UInt8Image,
    appearance_mask: UInt8Image | None,
    patch_gray: UInt8Image,
) -> float:
    """Compare foreground-to-background contrast ratio between template and candidate patch."""
    if appearance_mask is None:
        return 1.0
    mask = appearance_mask.astype(bool)
    background = ~mask
    if (
        int(np.count_nonzero(mask)) < APPEARANCE_MASK_MIN_PIXELS
        or int(np.count_nonzero(background)) < APPEARANCE_MASK_MIN_PIXELS
    ):
        return 1.0

    template_contrast = abs(
        float(template_gray[mask].astype(np.float32).mean())
        - float(template_gray[background].astype(np.float32).mean())
    )
    patch_contrast = abs(
        float(patch_gray[mask].astype(np.float32).mean())
        - float(patch_gray[background].astype(np.float32).mean())
    )
    if template_contrast <= 1e-3:
        return 1.0
    if patch_contrast <= 1e-3:
        return 0.0
    return min(template_contrast, patch_contrast) / max(template_contrast, patch_contrast)


def nms(candidates: list[Candidate], features: ModelFeatures, limit: int) -> list[Candidate]:
    """Perform non-maximum suppression using quadrilateral polygon IoU."""
    selected: list[Candidate] = []
    polygons: list[FloatImage] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        polygon = candidate_polygon(candidate, features)
        if any(polygon_iou(polygon, previous) > NMS_IOU for previous in polygons):
            continue
        selected.append(candidate)
        polygons.append(polygon)
        if len(selected) == limit:
            break
    return selected
