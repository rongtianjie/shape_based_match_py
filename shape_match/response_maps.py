"""Computation of multi-channel quantized gradient orientation response maps."""

from __future__ import annotations

import cv2
import numpy as np

from shape_match.gradients import consistent_edges, quantize_orientations
from shape_match.types import NUM_ORIENTATIONS, FloatImage, PatternConfig, UInt8Image


def orientation_response_maps(gray: UInt8Image, config: PatternConfig) -> FloatImage:
    """Compute 8-channel orientation response maps with cosine angular distance lookup and 3x3 dilation.

    Spreading responses over a 3x3 neighborhood allows fast coarse matching tolerant to spatial jitter.
    """
    edge_mask, gx, gy, _ = consistent_edges(gray, config)
    source_labels = quantize_orientations(gx, gy)
    responses = np.zeros((NUM_ORIENTATIONS, gray.shape[0], gray.shape[1]), dtype=np.float32)

    bins = np.arange(NUM_ORIENTATIONS, dtype=np.float32)
    for template_label in range(NUM_ORIENTATIONS):
        differences = np.abs(bins - template_label)
        differences = np.minimum(differences, NUM_ORIENTATIONS - differences)
        lookup = np.abs(np.cos(differences * (np.pi / NUM_ORIENTATIONS))).astype(np.float32)
        response = lookup[source_labels] * edge_mask
        responses[template_label] = cv2.dilate(response, np.ones((3, 3), np.uint8))
    return responses
