import numpy as np

from shape_match.types import (
    Candidate,
    MatchConfig,
    ModelFeatures,
    PatternConfig,
    ResponseImage,
    MIN_FEATURES,
    MIN_VISIBLE_RATIO,
)
from shape_match.matcher import fine_pose_values
from shape_match.gradients import quantize_orientations

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def refine_candidates_pytorch(
    candidates: list[Candidate],
    features: ModelFeatures,
    responses: ResponseImage,
    pattern: PatternConfig,
    matching: MatchConfig,
    min_visible_ratio: float = MIN_VISIBLE_RATIO,
) -> list[Candidate]:
    if not candidates:
        return []
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    radius = 4 if pattern.num_levels == 1 else 2
    image_height, image_width = responses.shape[1:]
    total_features = len(features.offsets)
    min_vis = max(1, min(MIN_FEATURES, total_features), int(round(min_visible_ratio * total_features)))
    
    refined_candidates = []
    responses_t = torch.from_numpy(responses).to(device=device, dtype=torch.float32)
    if responses.dtype == np.uint8:
        responses_t.mul_(1.0 / 255.0)
    
    for candidate in candidates:
        base_x = int(round(candidate.cx))
        base_y = int(round(candidate.cy))
        grid_x, grid_y = np.meshgrid(
            np.arange(base_x - radius, base_x + radius + 1),
            np.arange(base_y - radius, base_y + radius + 1),
        )
        centres = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
        
        angles, scales = fine_pose_values(candidate, pattern, matching)
        if not angles or not scales:
            continue
            
        N = len(angles) * len(scales)
        angles_arr = np.repeat(angles, len(scales))
        scales_arr = np.tile(scales, len(angles))
        
        radians = np.deg2rad(angles_arr)
        c = np.cos(radians) * scales_arr
        s = np.sin(radians) * scales_arr
        
        spatial_transforms = np.empty((N, 2, 2), dtype=np.float32)
        spatial_transforms[:, 0, 0] = c
        spatial_transforms[:, 0, 1] = s
        spatial_transforms[:, 1, 0] = -s
        spatial_transforms[:, 1, 1] = c
        
        dir_c = np.cos(radians)
        dir_s = np.sin(radians)
        dir_transforms = np.empty((N, 2, 2), dtype=np.float32)
        dir_transforms[:, 0, 0] = dir_c
        dir_transforms[:, 0, 1] = dir_s
        dir_transforms[:, 1, 0] = -dir_s
        dir_transforms[:, 1, 1] = dir_c
        
        all_offsets = np.einsum('mk,njk->nmj', features.offsets, spatial_transforms)
        
        all_gradients = np.einsum('mk,njk->nmj', features.unit_gradients, dir_transforms)
        gx = all_gradients[:, :, 0].ravel()
        gy = all_gradients[:, :, 1].ravel()
        # Keep the GPU refinement labels exactly aligned with the CPU path.
        # The response maps use polarity-independent, half-bin shifted
        # quantization over pi; rounding 0..360 degree phases into 45 degree
        # bins (the previous implementation) produced different labels and
        # could score a real match as background.
        labels = quantize_orientations(
            gx.astype(np.float32), gy.astype(np.float32), features.use_polarity
        )
        labels = labels.reshape(N, -1)
        
        K = len(centres)
        
        centres_t = torch.from_numpy(centres).to(device)
        offsets_t = torch.from_numpy(all_offsets).to(device)
        labels_t = torch.from_numpy(labels).to(torch.long).to(device)
        
        xs_raw = torch.round(centres_t[None, :, 0, None] + offsets_t[:, None, :, 0]).long()
        ys_raw = torch.round(centres_t[None, :, 1, None] + offsets_t[:, None, :, 1]).long()
        
        in_bounds = (xs_raw >= 0) & (xs_raw < image_width) & (ys_raw >= 0) & (ys_raw < image_height)
        vis_count = in_bounds.sum(dim=2)
        
        L = labels_t[:, None, :].expand(N, K, -1)
        
        xs = torch.clamp(xs_raw, 0, image_width - 1)
        ys = torch.clamp(ys_raw, 0, image_height - 1)
        
        values = responses_t[L, ys, xs] * in_bounds.float()
        scores = values.sum(dim=2) / torch.clamp(vis_count.float(), min=1.0)
        scores[vis_count < min_vis] = -1.0
        
        max_score = torch.max(scores)
        if max_score >= matching.min_score:
            flat_idx = torch.argmax(scores).item()
            best_n = flat_idx // K
            best_k = flat_idx % K
            
            refined_candidates.append(Candidate(
                cx=float(centres[best_k, 0]),
                cy=float(centres[best_k, 1]),
                score=float(max_score.item()),
                angle=float(angles_arr[best_n]),
                scale=float(scales_arr[best_n]),
            ))
            
    return refined_candidates
