import time
import numpy as np
import math

def rotation_matrix(angle, scale=1.0):
    radians = math.radians(angle)
    c, s = math.cos(radians) * scale, math.sin(radians) * scale
    return np.array([[c, s], [-s, c]], dtype=np.float32)

N = 121 # poses
M = 200 # points
K = 25 # centres

angles = np.random.rand(N) * 360
scales = np.random.rand(N) * 0.2 + 0.9

offsets_base = np.random.rand(M, 2).astype(np.float32)
labels_base = np.random.randint(0, 8, size=(M,))

t0 = time.time()

# 1. Create transform matrices
radians = np.deg2rad(angles)
c = np.cos(radians) * scales
s = np.sin(radians) * scales
spatial_transforms = np.empty((N, 2, 2), dtype=np.float32)
spatial_transforms[:, 0, 0] = c
spatial_transforms[:, 0, 1] = s
spatial_transforms[:, 1, 0] = -s
spatial_transforms[:, 1, 1] = c

# 2. Transform offsets
all_offsets = np.einsum('mk,nkj->nmj', offsets_base, spatial_transforms)

# centres: (K, 2)
centres = np.random.rand(K, 2).astype(np.float32)
# all_offsets: (N, M, 2)
# We want xs, ys: (N, K, M)
xs = np.rint(centres[None, :, 0, None] + all_offsets[:, None, :, 0]).astype(np.int32)
ys = np.rint(centres[None, :, 1, None] + all_offsets[:, None, :, 1]).astype(np.int32)

responses = np.random.rand(8, 400, 400).astype(np.float32)
# labels_base is (M,)
# We need to index responses[label, y, x]
# labels can be broadcast to (N, K, M)
L = np.broadcast_to(labels_base[None, None, :], xs.shape)
values = responses[L, ys, xs]
scores = values.mean(axis=2)

print(f'Vectorized refine in {time.time() - t0:.5f}s')
