"""Computation of multi-channel quantized gradient orientation response maps."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

from shape_match.gradients import consistent_edges, quantize_orientations
from shape_match.types import NUM_ORIENTATIONS, PatternConfig, UInt8Image


def _similarity_lookup() -> UInt8Image:
    """Build the orientation similarity table once at module import time."""
    bins = np.arange(NUM_ORIENTATIONS, dtype=np.float32)
    lookup = np.empty((NUM_ORIENTATIONS, NUM_ORIENTATIONS), dtype=np.uint8)
    for template_label in range(NUM_ORIENTATIONS):
        differences = np.minimum(
            np.abs(bins - template_label),
            NUM_ORIENTATIONS - np.abs(bins - template_label),
        )
        lookup[template_label] = np.rint(
            np.abs(np.cos(differences * (np.pi / NUM_ORIENTATIONS))) * 255.0
        ).astype(np.uint8)
    return lookup


_SIMILARITY_LOOKUP = _similarity_lookup()
_DILATION_KERNEL = np.ones((3, 3), np.uint8)


def orientation_response_maps(gray: UInt8Image, config: PatternConfig) -> UInt8Image:
    """Compute 8-channel orientation response maps with cosine angular distance lookup and 3x3 dilation.

    Spreading responses over a 3x3 neighborhood allows fast coarse matching tolerant to spatial jitter.
    """
    edge_mask, gx, gy, _ = consistent_edges(gray, config)
    source_labels = quantize_orientations(gx, gy)
    # Similarities are quantized to 8 bits while keeping the public score
    # convention unchanged: consumers divide sampled values by 255.0.
    responses = np.empty(
        (NUM_ORIENTATIONS, gray.shape[0], gray.shape[1]), dtype=np.uint8
    )

    def build_response(template_label: int) -> UInt8Image:
        response = _SIMILARITY_LOOKUP[template_label, source_labels] * edge_mask
        return cv2.dilate(response, _DILATION_KERNEL)

    worker_count = min(NUM_ORIENTATIONS, max(1, (os.cpu_count() or 1) // 2))
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for template_label, response in enumerate(
                executor.map(build_response, range(NUM_ORIENTATIONS))
            ):
                responses[template_label] = response
    else:
        for template_label in range(NUM_ORIENTATIONS):
            responses[template_label] = build_response(template_label)
    return responses
