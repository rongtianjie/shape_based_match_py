"""Rendering tools for template features and matched bounding boxes/annotations."""

from __future__ import annotations

import cv2
import numpy as np

from shape_match.image import to_bgr
from shape_match.transforms import candidate_polygon
from shape_match.types import Candidate, ModelFeatures, UInt8Image

ORIENTATION_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 80, 80),
    (255, 180, 60),
    (180, 255, 60),
    (60, 255, 120),
    (60, 255, 255),
    (60, 140, 255),
    (160, 60, 255),
    (255, 60, 180),
)

MATCH_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0),
    (0, 180, 255),
    (255, 120, 0),
    (255, 0, 220),
    (0, 255, 255),
)


def draw_model_features(image: np.ndarray, features: ModelFeatures) -> UInt8Image:
    """Render extracted shape features with orientation-colored circles and center crosshair on BGR image."""
    show = to_bgr(image)
    radius = 1 if min(features.width, features.height) < 160 else 2
    for (x, y), label in zip(features.points, features.labels, strict=True):
        cv2.circle(show, (int(x), int(y)), radius, ORIENTATION_PALETTE[int(label)], -1, lineType=cv2.LINE_AA)
    centre = (int(round((features.width - 1) / 2)), int(round((features.height - 1) / 2)))
    cv2.drawMarker(show, centre, (0, 255, 255), cv2.MARKER_CROSS, 11, 1, cv2.LINE_AA)
    return show


def draw_matches(
    image: np.ndarray, matches: list[Candidate], features: ModelFeatures
) -> UInt8Image:
    """Render oriented bounding boxes, center crosshairs, and score/pose text annotations."""
    show = to_bgr(image)
    for index, match in enumerate(matches):
        color = MATCH_PALETTE[index % len(MATCH_PALETTE)]
        polygon = np.rint(candidate_polygon(match, features)).astype(np.int32)
        cv2.polylines(show, [polygon], True, color, 2, cv2.LINE_AA)
        centre = (int(round(match.cx)), int(round(match.cy)))
        cv2.drawMarker(show, centre, color, cv2.MARKER_CROSS, 13, 2, cv2.LINE_AA)
        label = f"{match.score:.3f}  {match.angle:.1f}deg  x{match.scale:.2f}"
        text_at = (max(0, centre[0] + 6), max(14, centre[1] - 6))
        cv2.putText(show, label, text_at, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return show
