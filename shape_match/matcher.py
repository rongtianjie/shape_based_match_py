"""Hierarchical coarse-to-fine search, refinement, appearance verification, and NMS."""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np


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
    FINE_ANGLE_RADIUS,
    FINE_ANGLE_STEP,
    FINE_SCALE_RADIUS,
    FINE_SCALE_STEP,
    MIN_FEATURES,
    MIN_VISIBLE_RATIO,
    Candidate,
    FloatImage,
    MatchConfig,
    ModelFeatures,
    PatternConfig,
    PoseKernel,
    ResponseImage,
    UInt8Image,
)


def _vis_map_prefix_sum(
    template_sum: FloatImage,
    pad_top: int,
    pad_left: int,
    orig_h: int,
    orig_w: int,
    out_h: int,
    out_w: int,
) -> FloatImage:
    """Compute visible-feature-count map analytically using a 2-D prefix sum.

    When the FOV mask is a simple rectangle (ones inside the original image,
    zeros in the padding), the correlation ``matchTemplate(mask, template_sum)``
    reduces to a clipped sliding-window sum over ``template_sum``.  Computing
    this via a prefix sum avoids a full-size ``matchTemplate`` call and is
    ~1.5× faster.

    ``out_h`` and ``out_w`` are the spatial dimensions of the matchTemplate
    output (``padded_h - kh + 1``, ``padded_w - kw + 1``).
    """
    kh, kw = template_sum.shape
    cumsum = np.zeros((kh + 1, kw + 1), dtype=np.float64)
    cumsum[1:, 1:] = np.cumsum(
        np.cumsum(template_sum.astype(np.float64), axis=0), axis=1
    )
    ys = np.arange(out_h)
    xs = np.arange(out_w)
    lo_y = np.clip(pad_top - ys, 0, kh)
    hi_y = np.clip(pad_top - ys + orig_h, 0, kh)
    lo_x = np.clip(pad_left - xs, 0, kw)
    hi_x = np.clip(pad_left - xs + orig_w, 0, kw)
    return (
        cumsum[hi_y[:, None], hi_x[None, :]]
        - cumsum[lo_y[:, None], hi_x[None, :]]
        - cumsum[hi_y[:, None], lo_x[None, :]]
        + cumsum[lo_y[:, None], lo_x[None, :]]
    ).astype(np.float32)


def pose_score_map(
    responses: ResponseImage,
    kernel: PoseKernel,
    pad_top: int = 0,
    pad_left: int = 0,
    orig_h: int = 0,
    orig_w: int = 0,
    min_visible_ratio: float = MIN_VISIBLE_RATIO,
) -> FloatImage | None:
    """Correlate orientation response maps with a multi-channel pose kernel and return score map.

    When ``pad_top`` and ``pad_left`` are non-zero the responses are assumed to
    be zero-padded around an original image of size ``(orig_h, orig_w)``.  The
    visible feature count at each output position is computed analytically via
    a prefix sum rather than an extra ``matchTemplate`` call.
    """
    image_height, image_width = responses.shape[1:]
    if kernel.width > image_width or kernel.height > image_height:
        return None
    result: FloatImage | None = None
    template_sum: FloatImage | None = None
    for label, template in enumerate(kernel.kernels):
        if template is None:
            continue
        if template_sum is None:
            template_sum = template.copy()
        else:
            template_sum += template
        source = responses[label]
        if source.dtype == np.uint8:
            # The pose kernel stores integer feature multiplicities.  Keeping
            # both operands in 8-bit form lets OpenCV use its faster integer
            # correlation path; normalize the response before accumulation.
            if float(np.max(template)) <= 255.0:
                template_u8 = np.rint(template).astype(np.uint8)
                partial = cv2.matchTemplate(source, template_u8, cv2.TM_CCORR)
                partial *= np.float32(1.0 / 255.0)
            else:
                source = source.astype(np.float32) * np.float32(1.0 / 255.0)
                partial = cv2.matchTemplate(source, template, cv2.TM_CCORR)
        else:
            partial = cv2.matchTemplate(source, template, cv2.TM_CCORR)
        if result is None:
            result = partial
        else:
            result += partial
    if result is None:
        return None

    use_fov = pad_top > 0 or pad_left > 0
    if use_fov and template_sum is not None:
        out_h, out_w = result.shape
        vis_map = _vis_map_prefix_sum(
            template_sum, pad_top, pad_left, orig_h, orig_w, out_h, out_w
        )
        min_vis = max(1, min(MIN_FEATURES, kernel.feature_count), int(round(min_visible_ratio * kernel.feature_count)))
        vis_safe = np.maximum(vis_map, 1.0)
        result /= vis_safe
        result[vis_map < float(min_vis)] = 0.0
    else:
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
    step = config.angle_step if config.angle_step > 0.0 else COARSE_ANGLE_STEP
    values = sample_interval(
        config.angle_start,
        config.angle_start + config.angle_extent,
        step,
        cyclic=math.isclose(config.angle_extent, 360.0),
    )

    # The default interval is [-5, 5] degrees.  Anchoring a 10-degree grid at
    # the lower bound would otherwise sample only the two endpoints and miss
    # the most common (zero-rotation) pose entirely.  Add zero whenever it is
    # inside the configured interval (including a full-turn interval); keep
    # the list sorted for deterministic CPU/GPU candidate ordering.
    stop = config.angle_start + config.angle_extent
    if config.angle_step == 0.0 and (config.angle_start <= 0.0 <= stop) and not any(
        math.isclose(value, 0.0, abs_tol=1e-7) for value in values
    ):
        values.append(0.0)
        values.sort()
    return values


def coarse_scales(config: MatchConfig) -> list[float]:
    """Generate coarse search scale sequence based on match configuration."""
    return sample_interval(config.scale_min, config.scale_max, COARSE_SCALE_STEP)


def coarse_search_limits(config: MatchConfig) -> tuple[float, int, int]:
    """Translate HALCON-style greediness into coarse-search pruning controls.

    A value of zero explores more low-scoring hypotheses; a value of one
    retains only hypotheses already close to the requested final score.
    Final candidates are always checked against ``minScore``.
    """
    requested = max(1, config.num_matches)
    threshold_factor = 0.2 + 0.8 * config.greediness
    threshold = max(0.02, config.min_score * threshold_factor)
    per_pose_multiplier = max(2, int(math.ceil(5.0 - 4.0 * config.greediness)))
    global_multiplier = max(10, int(math.ceil(40.0 - 40.0 * config.greediness)))
    return (
        threshold,
        max(4, requested * per_pose_multiplier),
        max(24, requested * global_multiplier),
    )


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
    responses: ResponseImage,
    pattern: PatternConfig,
    matching: MatchConfig,
    image_factor: float,
) -> list[Candidate]:
    """Perform coarse grid search over discrete poses and extract candidate peak locations."""
    coarse_threshold, per_pose_limit, global_limit = coarse_search_limits(matching)
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
        use_polarity=features.use_polarity,
    )

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

    pose_specs = [
        (scale, angle, build_pose_kernel(scaled_features, angle, scale))
        for scale in scales
        for angle in angles
    ]
    if not pose_specs:
        return []

    # Pad each side by the maximum anchor offset so that the template centre
    # can reach any pixel of the original image.  The previous implementation
    # used ``kernel.width - 1`` on *each* side (double the necessary amount),
    # which inflated the padded image area by ~20-25 % and slowed every
    # ``matchTemplate`` call proportionally.
    max_pad_left = max(k.anchor_x for _, _, k in pose_specs)
    max_pad_right = max(k.width - 1 - k.anchor_x for _, _, k in pose_specs)
    max_pad_top = max(k.anchor_y for _, _, k in pose_specs)
    max_pad_bottom = max(k.height - 1 - k.anchor_y for _, _, k in pose_specs)
    padded_responses = np.pad(
        responses,
        ((0, 0), (max_pad_top, max_pad_bottom), (max_pad_left, max_pad_right)),
        mode="constant",
        constant_values=0,
    )
    orig_h, orig_w = responses.shape[1:]

    def evaluate_pose(
        pose_spec: tuple[float, float, PoseKernel]
    ) -> list[Candidate]:
        scale, angle, kernel = pose_spec
        score_map = pose_score_map(
            padded_responses, kernel,
            pad_top=max_pad_top, pad_left=max_pad_left,
            orig_h=orig_h, orig_w=orig_w,
        )
        if score_map is None:
            return []
        return [
            Candidate(
                cx=(left + kernel.anchor_x - max_pad_left) / image_factor,
                cy=(top + kernel.anchor_y - max_pad_top) / image_factor,
                score=score,
                angle=angle,
                scale=scale,
            )
            for score, left, top in top_local_peaks(
                score_map, coarse_threshold, per_pose_limit
            )
        ]

    # Each pose reads the same immutable response maps but writes independent
    # score maps.  OpenCV releases the GIL during correlation, so a bounded
    # thread pool speeds up CPU coarse search without copying image data.
    worker_count = min(
        len(pose_specs), max(1, os.cpu_count() or 1), 8
    )
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for pose_candidates in executor.map(evaluate_pose, pose_specs):
                candidates.extend(pose_candidates)
    else:
        for pose_spec in pose_specs:
            candidates.extend(evaluate_pose(pose_spec))

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
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
    angle_radius = max(
        FINE_ANGLE_RADIUS,
        pattern.angle_step * 0.5 if pattern.angle_step > 0.0 else 0.0,
    )
    for angle in sample_interval(
        candidate.angle - angle_radius,
        candidate.angle + angle_radius,
        FINE_ANGLE_STEP,
    ):
        canonical = canonical_angle(angle, pattern)
        if angle_allowed(canonical, pattern) and not any(math.isclose(canonical, item, abs_tol=1e-7) for item in angles):
            angles.append(canonical)

    scale_start = max(matching.scale_min, candidate.scale - FINE_SCALE_RADIUS)
    scale_stop = min(matching.scale_max, candidate.scale + FINE_SCALE_RADIUS)
    scales = sample_interval(scale_start, scale_stop, FINE_SCALE_STEP)
    return angles, scales



def scores_at_centres(
    responses: ResponseImage,
    offsets: FloatImage,
    labels: UInt8Image,
    centres: FloatImage,
    min_visible_ratio: float = MIN_VISIBLE_RATIO,
) -> FloatImage:
    """Compute average similarity score across visible feature points at given center coordinates."""
    image_height, image_width = responses.shape[1:]
    rounded_offsets = np.rint(offsets).astype(np.int32)
    rounded_centres = np.rint(centres).astype(np.int32)
    xs = rounded_centres[:, 0, None] + rounded_offsets[None, :, 0]
    ys = rounded_centres[:, 1, None] + rounded_offsets[None, :, 1]

    in_bounds = (xs >= 0) & (xs < image_width) & (ys >= 0) & (ys < image_height)
    vis_count = in_bounds.sum(axis=1)

    total_features = len(offsets)
    min_vis = max(1, min(MIN_FEATURES, total_features), int(round(min_visible_ratio * total_features)))

    xs_clamped = np.clip(xs, 0, image_width - 1)
    ys_clamped = np.clip(ys, 0, image_height - 1)

    values = responses[labels[None, :], ys_clamped, xs_clamped].astype(np.float32)
    if responses.dtype == np.uint8:
        values *= np.float32(1.0 / 255.0)

    values *= in_bounds.astype(np.float32)
    scores = values.sum(axis=1) / np.maximum(vis_count.astype(np.float32), 1.0)
    scores[vis_count < min_vis] = -1.0
    return scores


def _score_surface_3x3(
    responses: ResponseImage,
    offsets: FloatImage,
    labels: UInt8Image,
    cx: float,
    cy: float,
    min_visible_ratio: float = MIN_VISIBLE_RATIO,
) -> FloatImage:
    """Evaluate bilinear subpixel scores at a 3×3 neighbourhood in one vectorized pass.

    Returns a (3, 3) float64 array where ``surface[row, col]`` is the score at
    ``(cx + col - 1, cy + row - 1)``.  Positions with too few visible features
    are set to ``-inf``.  This is ~2.8× faster than 9 serial calls to
    ``score_at_subpixel_centre``.
    """
    image_height, image_width = responses.shape[1:]
    total_features = len(offsets)
    min_vis = max(1, min(MIN_FEATURES, total_features), int(round(min_visible_ratio * total_features)))

    # Offsets from the 9 neighbourhood positions: shape (9,)
    DX = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0], dtype=np.float32)
    DY = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # Feature positions for each of the 9 candidates: (9, N_features)
    xs = offsets[:, 0] + np.float32(cx) + DX[:, None]
    ys = offsets[:, 1] + np.float32(cy) + DY[:, None]

    in_bounds = (xs >= 0.0) & (xs <= image_width - 1.0) & (ys >= 0.0) & (ys <= image_height - 1.0)
    vis_count = in_bounds.sum(axis=1)  # (9,)

    # Bilinear interpolation
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.minimum(x0 + 1, image_width - 1)
    y1 = np.minimum(y0 + 1, image_height - 1)
    wx = xs - x0
    wy = ys - y0

    # Clamp for safe indexing (masked positions will be zeroed out)
    x0c = np.clip(x0, 0, image_width - 1)
    y0c = np.clip(y0, 0, image_height - 1)
    x1c = np.clip(x1, 0, image_width - 1)
    y1c = np.clip(y1, 0, image_height - 1)

    L = labels[None, :]  # (1, N_features)
    value_scale = np.float32(1.0 / 255.0) if responses.dtype == np.uint8 else np.float32(1.0)
    v00 = responses[L, y0c, x0c].astype(np.float32) * value_scale
    v10 = responses[L, y0c, x1c].astype(np.float32) * value_scale
    v01 = responses[L, y1c, x0c].astype(np.float32) * value_scale
    v11 = responses[L, y1c, x1c].astype(np.float32) * value_scale
    values = v00 * (1.0 - wx) * (1.0 - wy) + v10 * wx * (1.0 - wy) + v01 * (1.0 - wx) * wy + v11 * wx * wy
    values *= in_bounds.astype(np.float32)

    scores = values.sum(axis=1) / np.maximum(vis_count.astype(np.float32), 1.0)
    scores[vis_count < min_vis] = float("-inf")
    return scores.astype(np.float64).reshape(3, 3)


def _parabolic_offset(negative: float, centre: float, positive: float) -> float:
    denominator = negative - 2.0 * centre + positive
    if denominator >= -1e-9:
        return 0.0
    return float(np.clip(0.5 * (negative - positive) / denominator, -0.5, 0.5))


def refine_subpixel_candidate(
    candidate: Candidate,
    features: ModelFeatures,
    responses: ResponseImage,
    mode: int,
) -> Candidate:
    """Refine a candidate centre by interpolation (1) or quadratic least squares (2)."""
    base_x = int(round(candidate.cx))
    base_y = int(round(candidate.cy))
    if mode == 0:
        return Candidate(float(base_x), float(base_y), candidate.score, candidate.angle, candidate.scale)

    offsets, _, labels, _ = transformed_geometry(features, candidate.angle, candidate.scale)
    surface = _score_surface_3x3(responses, offsets, labels, float(base_x), float(base_y))
    if not np.all(np.isfinite(surface)):
        return Candidate(float(base_x), float(base_y), candidate.score, candidate.angle, candidate.scale)
    if float(np.ptp(surface)) < 1e-6:
        return Candidate(float(base_x), float(base_y), candidate.score, candidate.angle, candidate.scale)

    if mode == 1:
        delta_x = _parabolic_offset(surface[1, 0], surface[1, 1], surface[1, 2])
        delta_y = _parabolic_offset(surface[0, 1], surface[1, 1], surface[2, 1])
    else:
        coordinates = np.asarray(
            [(x, y) for y in (-1.0, 0.0, 1.0) for x in (-1.0, 0.0, 1.0)],
            dtype=np.float64,
        )
        design = np.column_stack(
            (
                coordinates[:, 0] ** 2,
                coordinates[:, 1] ** 2,
                coordinates[:, 0] * coordinates[:, 1],
                coordinates[:, 0],
                coordinates[:, 1],
                np.ones(9),
            )
        )
        coefficients, *_ = np.linalg.lstsq(design, surface.ravel(), rcond=None)
        a, b, cross, d, e, _ = coefficients
        hessian = np.asarray(((2.0 * a, cross), (cross, 2.0 * b)), dtype=np.float64)
        if (
            np.all(np.linalg.eigvalsh(hessian) < -1e-9)
            and np.linalg.cond(hessian) < 1e6
        ):
            delta_x, delta_y = np.linalg.solve(hessian, -np.asarray((d, e), dtype=np.float64))
            delta_x = float(np.clip(delta_x, -0.5, 0.5))
            delta_y = float(np.clip(delta_y, -0.5, 0.5))
        else:
            delta_x = delta_y = 0.0

    refined_x = float(base_x + delta_x)
    refined_y = float(base_y + delta_y)
    return Candidate(
        refined_x,
        refined_y,
        candidate.score,
        candidate.angle,
        candidate.scale,
    )


def refine_candidate(
    candidate: Candidate,
    features: ModelFeatures,
    responses: ResponseImage,
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
    best: Candidate | None = None

    angles, scales = fine_pose_values(candidate, pattern, matching)
    for scale in scales:
        for angle in angles:
            offsets, _, labels, _ = transformed_geometry(features, angle, scale)
            scores = scores_at_centres(responses, offsets, labels, centres)
            max_score = float(np.max(scores))
            if max_score < 0.0:
                continue
            index = int(np.argmax(scores))
            score = float(scores[index])
            if best is None or score > best.score:
                centre = centres[index]
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
) -> tuple[UInt8Image, UInt8Image]:
    """Resample the source image region covered by candidate back into template canonical coordinate frame."""
    rotation = rotation_matrix(candidate.angle, candidate.scale)
    template_centre = np.array([(features.width - 1) / 2.0, (features.height - 1) / 2.0], dtype=np.float32)
    candidate_centre = np.array([candidate.cx, candidate.cy], dtype=np.float32)
    translation = candidate_centre - rotation @ template_centre
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[:, :2] = rotation
    matrix[:, 2] = translation
    patch_gray = cv2.warpAffine(
        source_gray,
        matrix,
        (features.width, features.height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    src_mask = np.ones_like(source_gray, dtype=np.uint8)
    patch_valid_mask = cv2.warpAffine(
        src_mask,
        matrix,
        (features.width, features.height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return patch_gray, patch_valid_mask


def appearance_score(
    template_gray: UInt8Image,
    appearance_mask: UInt8Image | None,
    patch_gray: UInt8Image,
    patch_valid_mask: UInt8Image | None = None,
) -> float:
    """Compare foreground-to-background contrast ratio between template and candidate patch."""
    if appearance_mask is None:
        return 1.0
    mask = appearance_mask.astype(bool)
    background = ~mask
    if patch_valid_mask is not None:
        valid = patch_valid_mask.astype(bool)
        mask = mask & valid
        background = background & valid

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


def nms(
    candidates: list[Candidate],
    features: ModelFeatures,
    limit: int,
    max_overlap: float = 0.5,
) -> list[Candidate]:
    """Perform non-maximum suppression using quadrilateral polygon IoU."""
    selected: list[Candidate] = []
    polygons: list[FloatImage] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        polygon = candidate_polygon(candidate, features)
        if any(polygon_iou(polygon, previous) > max_overlap for previous in polygons):
            continue
        selected.append(candidate)
        polygons.append(polygon)
        if limit > 0 and len(selected) == limit:
            break
    return selected
