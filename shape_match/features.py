"""Foreground segmentation, spatial feature dispersion, and model feature extraction."""

from __future__ import annotations

import math
import cv2
import numpy as np
from numpy.typing import NDArray

from shape_match.gradients import consistent_edges, quantize_orientations
from shape_match.types import (
    FG_BAND_RADIUS_FRACTION,
    FG_BAND_RADIUS_MAX,
    FG_BAND_RADIUS_MIN,
    FG_BORDER_FRACTION,
    FG_BORDER_MAX,
    FG_BORDER_MIN,
    FG_CENTRAL_FRAC,
    FG_CORE_KERNEL,
    FG_MAX_COMPONENT_AREA_FRAC,
    FG_MAX_SIGMA,
    FG_MAX_UNION_AREA_FRAC,
    FG_MIN_COMPONENT_AREA_FRAC,
    FG_MIN_SIGMA,
    FG_Z_THRESHOLD,
    MAX_FEATURES,
    MIN_FEATURES,
    FloatImage,
    ModelFeatures,
    PatternConfig,
    UInt8Image,
)


def select_scattered(
    points_xy: NDArray[np.int32], strengths: FloatImage, limit: int
) -> NDArray[np.int32]:
    """Select up to *limit* feature points maximizing spatial dispersion across the template."""
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


def segment_foreground(gray: UInt8Image, bgr: UInt8Image) -> UInt8Image | None:
    """Detect a centrally-located foreground blob using border-ring background statistics.

    Returns a 0/1 uint8 mask, or ``None`` when the border does not approximate a
    uniform background (e.g. template already tightly crops the target) or no
    plausible blob is found.
    """
    height, width = gray.shape
    border = int(np.clip(round(FG_BORDER_FRACTION * min(height, width)), FG_BORDER_MIN, FG_BORDER_MAX))
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
        if sigma > FG_MAX_SIGMA:
            return None
        sigma = max(sigma, FG_MIN_SIGMA)
        np.maximum(z_max, np.abs(values - median) / sigma, out=z_max)

    candidate = (z_max > FG_Z_THRESHOLD).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if num_labels <= 1:
        return None

    image_area = float(height * width)
    centre_x, centre_y = (width - 1) / 2.0, (height - 1) / 2.0
    max_dist_x = FG_CENTRAL_FRAC * width
    max_dist_y = FG_CENTRAL_FRAC * height

    union = np.zeros((height, width), dtype=np.uint8)
    for label in range(1, num_labels):
        left, top, comp_w, comp_h, area = stats[label]
        area_frac = area / image_area
        if not (FG_MIN_COMPONENT_AREA_FRAC <= area_frac <= FG_MAX_COMPONENT_AREA_FRAC):
            continue
        centroid_x, centroid_y = centroids[label]
        if abs(centroid_x - centre_x) > max_dist_x or abs(centroid_y - centre_y) > max_dist_y:
            continue
        touches_border = left <= 0 or top <= 0 or left + comp_w >= width or top + comp_h >= height
        if touches_border:
            # A hairline bridge can connect an otherwise-central blob to the border.
            # Keep component only if a solid core survives erosion and clears the border.
            component_mask = (labels_img == label).astype(np.uint8)
            core = cv2.erode(component_mask, FG_CORE_KERNEL)
            if not np.any(core):
                continue
            core_ys, core_xs = np.nonzero(core)
            if core_xs.min() <= 0 or core_ys.min() <= 0 or core_xs.max() >= width - 1 or core_ys.max() >= height - 1:
                continue
        union[labels_img == label] = 1

    if not np.any(union):
        return None
    if np.count_nonzero(union) / image_area > FG_MAX_UNION_AREA_FRAC:
        return None
    return union


def extract_features(
    gray: UInt8Image, config: PatternConfig, bgr: UInt8Image
) -> ModelFeatures | None:
    """Extract sparse gradient-orientation shape features and appearance mask from template."""
    edge_mask, gx, gy, magnitude = consistent_edges(gray, config)
    height, width = gray.shape

    appearance_mask: UInt8Image | None = None
    blob = segment_foreground(gray, bgr)
    if blob is not None:
        radius = int(
            np.clip(
                round(FG_BAND_RADIUS_FRACTION * min(height, width)),
                FG_BAND_RADIUS_MIN,
                FG_BAND_RADIUS_MAX,
            )
        )
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        dilated = cv2.dilate(blob, kernel) > 0
        eroded = cv2.erode(blob, kernel) > 0
        band = (dilated & ~eroded).astype(np.uint8)
        banded_edges = edge_mask & band
        if int(np.count_nonzero(banded_edges)) >= MIN_FEATURES:
            edge_mask = banded_edges
        appearance_mask = dilated.astype(np.uint8)

    ys, xs = np.nonzero(edge_mask)
    if len(xs) < MIN_FEATURES:
        return None

    points = np.column_stack((xs, ys)).astype(np.int32)
    strengths = magnitude[ys, xs]
    chosen = select_scattered(points, strengths, MAX_FEATURES)
    points = points[chosen]
    xs = points[:, 0]
    ys = points[:, 1]
    magnitudes = np.maximum(magnitude[ys, xs], np.float32(1e-6))
    gradients = np.column_stack((gx[ys, xs] / magnitudes, gy[ys, xs] / magnitudes)).astype(np.float32)
    labels = quantize_orientations(gradients[:, 0], gradients[:, 1])

    centre = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32)
    offsets = points.astype(np.float32) - centre
    return ModelFeatures(
        offsets=offsets,
        unit_gradients=gradients,
        points=points,
        labels=labels,
        width=width,
        height=height,
        template_gray=gray,
        appearance_mask=appearance_mask,
    )
