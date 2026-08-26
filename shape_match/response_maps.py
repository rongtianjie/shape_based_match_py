"""Computation of multi-channel quantized gradient orientation response maps."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

from shape_match.gradients import consistent_edges, quantize_orientations
from shape_match.types import DIRECTED_NUM_ORIENTATIONS, NUM_ORIENTATIONS, PatternConfig, UInt8Image


def _similarity_lookup(use_polarity: bool = False) -> UInt8Image:
    """Build the orientation similarity table once at module import time."""
    orientation_count = DIRECTED_NUM_ORIENTATIONS if use_polarity else NUM_ORIENTATIONS
    bins = np.arange(orientation_count, dtype=np.float32)
    lookup = np.empty((orientation_count, orientation_count), dtype=np.uint8)
    for template_label in range(orientation_count):
        differences = np.minimum(
            np.abs(bins - template_label),
            orientation_count - np.abs(bins - template_label),
        )
        period = 2.0 * np.pi if use_polarity else np.pi
        cosine = np.cos(differences * (period / orientation_count))
        if use_polarity:
            cosine = np.maximum(cosine, 0.0)
        else:
            cosine = np.abs(cosine)
        lookup[template_label] = np.rint(cosine * 255.0).astype(np.uint8)
    return lookup


_SIMILARITY_LOOKUPS = {
    False: _similarity_lookup(False),
    True: _similarity_lookup(True),
}
_DILATION_KERNEL = np.ones((3, 3), np.uint8)


def orientation_response_maps(gray: UInt8Image, config: PatternConfig) -> UInt8Image:
    """Compute 8-channel orientation response maps with cosine angular distance lookup and 3x3 dilation.

    Spreading responses over a 3x3 neighborhood allows fast coarse matching tolerant to spatial jitter.
    """
    edge_mask, gx, gy, _ = consistent_edges(gray, config)
    use_polarity = bool(config.use_polarity)
    source_labels = quantize_orientations(gx, gy, use_polarity)
    orientation_count = DIRECTED_NUM_ORIENTATIONS if use_polarity else NUM_ORIENTATIONS
    similarity_lookup = _SIMILARITY_LOOKUPS[use_polarity]
    # Similarities are quantized to 8 bits while keeping the public score
    # convention unchanged: consumers divide sampled values by 255.0.
    responses = np.empty(
        (orientation_count, gray.shape[0], gray.shape[1]), dtype=np.uint8
    )

    def build_response(template_label: int) -> UInt8Image:
        response = similarity_lookup[template_label, source_labels] * edge_mask
        return cv2.dilate(response, _DILATION_KERNEL)

    worker_count = min(orientation_count, max(1, (os.cpu_count() or 1) // 2))
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for template_label, response in enumerate(
                executor.map(build_response, range(orientation_count))
            ):
                responses[template_label] = response
    else:
        for template_label in range(orientation_count):
            responses[template_label] = build_response(template_label)
    return responses
