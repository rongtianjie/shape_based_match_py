"""Image validation, normalization, and color space conversions."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from shape_match.types import UInt8Image


def validate_image(image: Any, name: str) -> np.ndarray:
    """Validate that input is a non-empty, finite numeric numpy image (2D gray, 3-ch BGR, or 4-ch BGRA)."""
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


def to_uint8(image: np.ndarray) -> UInt8Image:
    """Convert image to uint8 array with values clipped to [0, 255]."""
    if image.dtype == np.uint8:
        return image.copy()
    return np.clip(image, 0, 255).astype(np.uint8)


def to_gray(image: np.ndarray) -> UInt8Image:
    """Convert input image (gray, BGR, or BGRA) into a 2D uint8 grayscale image."""
    converted = to_uint8(image)
    if converted.ndim == 2:
        return converted
    if converted.shape[2] == 1:
        return converted[:, :, 0]
    if converted.shape[2] == 3:
        return cv2.cvtColor(converted, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(converted, cv2.COLOR_BGRA2GRAY)


def to_bgr(image: np.ndarray) -> UInt8Image:
    """Convert input image (gray, BGR, or BGRA) into a 3-channel uint8 BGR image."""
    converted = to_uint8(image)
    if converted.ndim == 2:
        return cv2.cvtColor(converted, cv2.COLOR_GRAY2BGR)
    if converted.shape[2] == 1:
        return cv2.cvtColor(converted[:, :, 0], cv2.COLOR_GRAY2BGR)
    if converted.shape[2] == 4:
        return cv2.cvtColor(converted, cv2.COLOR_BGRA2BGR)
    return converted
