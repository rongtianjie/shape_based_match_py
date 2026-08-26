"""Micro-benchmark for vectorized candidate refinement."""

from __future__ import annotations

import math
import time

import numpy as np


def rotation_matrix(angle: float, scale: float = 1.0) -> np.ndarray:
    radians = math.radians(angle)
    c, s = math.cos(radians) * scale, math.sin(radians) * scale
    return np.array([[c, s], [-s, c]], dtype=np.float32)


def main() -> None:
    num_poses = 121  # poses
    num_points = 200  # points
    num_centres = 25  # centres

    angles = np.random.rand(num_poses) * 360
    scales = np.random.rand(num_poses) * 0.2 + 0.9

    offsets_base = np.random.rand(num_points, 2).astype(np.float32)
    labels_base = np.random.randint(0, 8, size=(num_points,))

    t0 = time.perf_counter()

    # 1. Create transform matrices
    radians = np.deg2rad(angles)
    c = np.cos(radians) * scales
    s = np.sin(radians) * scales
    spatial_transforms = np.empty((num_poses, 2, 2), dtype=np.float32)
    spatial_transforms[:, 0, 0] = c
    spatial_transforms[:, 0, 1] = s
    spatial_transforms[:, 1, 0] = -s
    spatial_transforms[:, 1, 1] = c

    # 2. Transform offsets
    all_offsets = np.einsum("mk,nkj->nmj", offsets_base, spatial_transforms)

    # centres: (num_centres, 2)
    centres = np.random.rand(num_centres, 2).astype(np.float32)
    xs = np.rint(centres[None, :, 0, None] + all_offsets[:, None, :, 0]).astype(np.int32)
    ys = np.rint(centres[None, :, 1, None] + all_offsets[:, None, :, 1]).astype(np.int32)

    responses = np.random.rand(8, 400, 400).astype(np.float32)
    labels_broadcast = np.broadcast_to(labels_base[None, None, :], xs.shape)
    values = responses[labels_broadcast, ys, xs]
    scores = values.mean(axis=2)

    duration = time.perf_counter() - t0
    print(f"Vectorized refine calculated {scores.shape} scores in {duration * 1000:.3f} ms")


if __name__ == "__main__":
    main()
