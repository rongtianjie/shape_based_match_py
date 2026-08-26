"""Core data types, data structures, and algorithmic constants for shape matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray

# Type aliases for image arrays
FloatImage = NDArray[np.float32]
UInt8Image = NDArray[np.uint8]
ResponseImage = FloatImage | UInt8Image

# Hyperparameters & algorithmic constants
NUM_ORIENTATIONS: int = 8
DIRECTED_NUM_ORIENTATIONS: int = NUM_ORIENTATIONS * 2
MAX_FEATURES: int = 256
MIN_FEATURES: int = 8
MIN_VISIBLE_RATIO: float = 0.2

COARSE_ANGLE_STEP: float = 10.0
FINE_ANGLE_STEP: float = 1.0
COARSE_SCALE_STEP: float = 0.1
FINE_SCALE_STEP: float = 0.01
FINE_ANGLE_RADIUS: float = 5.0
FINE_SCALE_RADIUS: float = 0.05
NMS_IOU: float = 0.5

AUTO_CANNY_LOW_RATIO: float = 0.65
AUTO_CANNY_HIGH_RATIO: float = 1.30

FG_BORDER_FRACTION: float = 0.07
FG_BORDER_MIN: int = 3
FG_BORDER_MAX: int = 20
FG_MAX_SIGMA: float = 20.0
FG_MIN_SIGMA: float = 6.0
FG_Z_THRESHOLD: float = 4.0
FG_MIN_COMPONENT_AREA_FRAC: float = 0.005
FG_MAX_COMPONENT_AREA_FRAC: float = 0.9
FG_MAX_UNION_AREA_FRAC: float = 0.85
FG_CENTRAL_FRAC: float = 0.4
FG_BAND_RADIUS_FRACTION: float = 0.05
FG_BAND_RADIUS_MIN: int = 3
FG_BAND_RADIUS_MAX: int = 15
FG_CORE_KERNEL: UInt8Image = np.ones((5, 5), dtype=np.uint8)

APPEARANCE_MIN_SCORE: float = 0.3
APPEARANCE_MASK_MIN_PIXELS: int = 16


@dataclass(frozen=True)
class PatternConfig:
    """Configuration for shape model extraction and angular search range."""

    contrast_low: int
    contrast_high: int
    angle_start: float
    angle_extent: float
    num_levels: int
    min_contrast: int = 3
    min_cont_len: int = 1
    use_polarity: int = 0
    angle_step: float = 0.0
    auto_contrast: bool = False


@dataclass(frozen=True)
class MatchConfig:
    """Configuration for matching thresholds, match count, and scale search range."""

    num_matches: int
    min_score: float
    scale_min: float
    scale_max: float
    subpixel: int = 1
    max_overlap: float = 0.5
    greediness: float = 0.75


@dataclass(frozen=True)
class ModelFeatures:
    """Extracted sparse shape features for a template model.

    Attributes:
        offsets: (N, 2) float32 coordinates relative to geometric template center (x, y).
        unit_gradients: (N, 2) float32 normalized gradient directions (gx, gy).
        points: (N, 2) int32 pixel coordinates in template image coordinate system.
        labels: (N,) uint8 orientation indices in [0, 7] without polarity
            or [0, 15] with polarity.
        width: Template width in pixels.
        height: Template height in pixels.
        template_gray: Single-channel grayscale template image.
        appearance_mask: Optional binary foreground mask for appearance contrast verification.
    """

    offsets: FloatImage
    unit_gradients: FloatImage
    points: NDArray[np.int32]
    labels: UInt8Image
    width: int
    height: int
    template_gray: UInt8Image
    appearance_mask: UInt8Image | None
    use_polarity: bool = False


@dataclass(frozen=True)
class PoseKernel:
    """Multi-channel convolution template kernel corresponding to a specific pose (angle, scale)."""

    kernels: tuple[FloatImage | None, ...]
    feature_count: int
    anchor_x: int
    anchor_y: int
    width: int
    height: int


@dataclass(frozen=True)
class Candidate:
    """A single matching candidate pose hypothesis.

    Attributes:
        cx: Center X coordinate in source image (pixels).
        cy: Center Y coordinate in source image (pixels).
        score: Normalized gradient orientation similarity score in [0.0, 1.0].
        angle: Rotation angle in degrees (counter-clockwise).
        scale: Scaling factor relative to template.
    """

    cx: float
    cy: float
    score: float
    angle: float
    scale: float

    def to_list(self) -> list[float]:
        """Format as [cx, cy, score, angle, scale] for API output."""
        return [float(self.cx), float(self.cy), float(self.score), float(self.angle), float(self.scale)]


class MatchResult(NamedTuple):
    """Container for match result list and optional visualization image."""

    matches: list[list[float]]
    visualization: UInt8Image | None
