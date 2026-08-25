"""Geometric transformations, pose kernel construction, and polygon operations."""

from __future__ import annotations

import math
import cv2
import numpy as np

from shape_match.gradients import quantize_orientations
from shape_match.types import (
    NUM_ORIENTATIONS,
    Candidate,
    FloatImage,
    ModelFeatures,
    PoseKernel,
    UInt8Image,
)


def rotation_matrix(angle: float, scale: float = 1.0) -> FloatImage:
    """Generate 2x2 rotation and scale matrix. Positive angle represents CCW rotation."""
    radians = math.radians(angle)
    cosine = math.cos(radians) * scale
    sine = math.sin(radians) * scale
    return np.asarray(((cosine, sine), (-sine, cosine)), dtype=np.float32)


def transformed_geometry(
    features: ModelFeatures, angle: float, scale: float
) -> tuple[FloatImage, FloatImage, UInt8Image, FloatImage]:
    """Transform model feature offsets, gradient directions, and bounding box corners for a pose."""
    spatial_transform = rotation_matrix(angle, scale)
    direction_transform = rotation_matrix(angle)
    offsets = features.offsets @ spatial_transform.T
    gradients = features.unit_gradients @ direction_transform.T
    labels = quantize_orientations(gradients[:, 0], gradients[:, 1])

    half_width = (features.width - 1) / 2.0
    half_height = (features.height - 1) / 2.0
    corners = np.asarray(
        (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ),
        dtype=np.float32,
    )
    corners = corners @ spatial_transform.T
    return offsets, gradients, labels, corners


def build_pose_kernel(features: ModelFeatures, angle: float, scale: float) -> PoseKernel:
    """Construct multi-channel sparse convolution template kernels for a given pose hypothesis."""
    offsets, _, labels, corners = transformed_geometry(features, angle, scale)
    rounded_offsets = np.rint(offsets).astype(np.int32)
    min_x = int(math.floor(float(np.min(corners[:, 0]))))
    max_x = int(math.ceil(float(np.max(corners[:, 0]))))
    min_y = int(math.floor(float(np.min(corners[:, 1]))))
    max_y = int(math.ceil(float(np.max(corners[:, 1]))))
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    kernels: list[FloatImage | None] = []
    for label in range(NUM_ORIENTATIONS):
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

    return PoseKernel(
        kernels=tuple(kernels),
        feature_count=len(offsets),
        anchor_x=-min_x,
        anchor_y=-min_y,
        width=width,
        height=height,
    )


def candidate_polygon(candidate: Candidate, features: ModelFeatures) -> FloatImage:
    """Compute the 4 oriented bounding box vertices for a candidate in source image space."""
    _, _, _, corners = transformed_geometry(features, candidate.angle, candidate.scale)
    return corners + np.asarray((candidate.cx, candidate.cy), dtype=np.float32)


def polygon_iou(first: FloatImage, second: FloatImage) -> float:
    """Calculate Intersection-over-Union (IoU) between two convex quadrilaterals."""
    area_first = abs(float(cv2.contourArea(first)))
    area_second = abs(float(cv2.contourArea(second)))
    if area_first <= 0.0 or area_second <= 0.0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = area_first + area_second - float(intersection)
    return float(intersection) / union if union > 0.0 else 0.0
