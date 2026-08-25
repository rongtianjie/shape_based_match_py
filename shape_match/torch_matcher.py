import math
import numpy as np
from typing import List

from shape_match.types import Candidate, MatchConfig, ModelFeatures, PatternConfig, FloatImage
from shape_match.transforms import build_pose_kernel

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def coarse_search_pytorch(
    features: ModelFeatures,
    responses: FloatImage,
    pattern: PatternConfig,
    matching: MatchConfig,
    image_factor: float,
    coarse_scales: list[float],
    coarse_angles: list[float],
) -> list[Candidate]:
    per_pose_limit = max(4, matching.num_matches * 2)
    global_limit = max(24, matching.num_matches * 10)
    coarse_threshold = max(0.02, matching.min_score * 0.8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pose_kernels = []
    for scale in coarse_scales:
        for angle in coarse_angles:
            kernel = build_pose_kernel(features, angle, scale)
            pose_kernels.append((scale, angle, kernel))
            
    if not pose_kernels:
        return []

    max_anchor_x = max(k.anchor_x for _, _, k in pose_kernels)
    max_anchor_y = max(k.anchor_y for _, _, k in pose_kernels)
    max_right = max(k.width - k.anchor_x for _, _, k in pose_kernels)
    max_bottom = max(k.height - k.anchor_y for _, _, k in pose_kernels)

    pad_width = max_anchor_x + max_right
    pad_height = max_anchor_y + max_bottom
    
    # if image is smaller than kernel, return empty
    if responses.shape[1] < pad_height or responses.shape[2] < pad_width:
        return []
    
    num_poses = len(pose_kernels)
    num_orientations = len(pose_kernels[0][2].kernels)
    
    weights = np.zeros((num_poses, num_orientations, pad_height, pad_width), dtype=np.float32)
    
    for i, (scale, angle, k) in enumerate(pose_kernels):
        start_y = max_anchor_y - k.anchor_y
        start_x = max_anchor_x - k.anchor_x
        inv_count = 1.0 / float(k.feature_count) if k.feature_count > 0 else 0.0
        for j, template in enumerate(k.kernels):
            if template is not None:
                weights[i, j, start_y : start_y + k.height, start_x : start_x + k.width] = template * inv_count
                
    weights_tensor = torch.from_numpy(weights).to(device)
    responses_tensor = torch.from_numpy(responses).unsqueeze(0).to(device)
    
    candidates = []
    with torch.no_grad():
        out = F.conv2d(responses_tensor, weights_tensor).squeeze(0) # (num_poses, out_H, out_W)
        out = torch.clamp(out, 0.0, 1.0)
        
        # local peaks using max pool
        max_pooled = F.max_pool2d(out.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
        is_peak = (out >= max_pooled - 1e-7) & (out >= coarse_threshold)
        
        out_masked = out.clone()
        out_masked[~is_peak] = -1.0
        
        out_H, out_W = out.shape[1:]
        k_limit = min(per_pose_limit, out_H * out_W)
        
        topk_scores, topk_indices = torch.topk(out_masked.view(num_poses, -1), k_limit, dim=1)
        
        valid_mask = topk_scores >= coarse_threshold
        pose_indices, k_indices = torch.where(valid_mask)
        
        scores = topk_scores[pose_indices, k_indices].cpu().numpy()
        flat_spatial_indices = topk_indices[pose_indices, k_indices].cpu().numpy()
        pose_indices = pose_indices.cpu().numpy()
        
        ys = flat_spatial_indices // out_W
        xs = flat_spatial_indices % out_W
        
        for idx in range(len(scores)):
            p_idx = pose_indices[idx]
            scale, angle, _ = pose_kernels[p_idx]
            left = xs[idx]
            top = ys[idx]
            score = float(scores[idx])
            
            cx = (left + max_anchor_x) / image_factor
            cy = (top + max_anchor_y) / image_factor
            candidates.append(Candidate(cx=cx, cy=cy, score=score, angle=angle, scale=scale))
            
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:global_limit]
