import numpy as np

from shape_match.types import (
    Candidate,
    MatchConfig,
    ModelFeatures,
    PatternConfig,
    PoseKernel,
    ResponseImage,
    MIN_FEATURES,
    MIN_VISIBLE_RATIO,
)
from shape_match.transforms import build_pose_kernel
from shape_match.matcher import coarse_search_limits

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _sparse_pose_score_maps(
    responses: ResponseImage,
    pose_kernels: list[tuple[float, float, PoseKernel]],
    pad_left: int,
    pad_right: int,
    pad_top: int,
    pad_bottom: int,
    device: "torch.device",
    min_visible_ratio: float = MIN_VISIBLE_RATIO,
) -> "torch.Tensor":
    """Accumulate sparse pose kernels with zero padding and visible count normalization."""
    responses_tensor = torch.from_numpy(np.ascontiguousarray(responses)).to(
        device=device, dtype=torch.float32
    )
    if responses.dtype == np.uint8:
        responses_tensor.mul_(1.0 / 255.0)

    orig_h, orig_w = responses.shape[1:]
    padded_responses = F.pad(
        responses_tensor,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0.0,
    )
    fov_mask = torch.zeros(
        (orig_h + pad_top + pad_bottom, orig_w + pad_left + pad_right),
        device=device,
        dtype=torch.float32,
    )
    fov_mask[pad_top : pad_top + orig_h, pad_left : pad_left + orig_w] = 1.0

    score_maps: list["torch.Tensor"] = []

    for _, _, kernel in pose_kernels:
        output_height = orig_h + pad_top + pad_bottom - kernel.height + 1
        output_width = orig_w + pad_left + pad_right - kernel.width + 1
        score_sum = torch.zeros(
            (output_height, output_width), device=device, dtype=torch.float32
        )
        vis_sum = torch.zeros(
            (output_height, output_width), device=device, dtype=torch.float32
        )
        min_vis = max(1, min(MIN_FEATURES, kernel.feature_count), int(round(min_visible_ratio * kernel.feature_count)))

        for label, template in enumerate(kernel.kernels):
            if template is None:
                continue
            ys, xs = np.nonzero(template)
            values = template[ys, xs]
            source = padded_responses[label]
            for y, x, value in zip(ys, xs, values):
                score_sum.add_(
                    source[
                        int(y) : int(y) + output_height,
                        int(x) : int(x) + output_width,
                    ],
                    alpha=float(value),
                )
                vis_sum.add_(
                    fov_mask[
                        int(y) : int(y) + output_height,
                        int(x) : int(x) + output_width,
                    ],
                    alpha=float(value),
                )
        score_map = score_sum / torch.clamp(vis_sum, min=1.0)
        score_map[vis_sum < float(min_vis)] = 0.0
        score_maps.append(score_map)

    return torch.stack(score_maps)


def coarse_search_pytorch(
    features: ModelFeatures,
    responses: ResponseImage,
    pattern: PatternConfig,
    matching: MatchConfig,
    image_factor: float,
    coarse_scales: list[float],
    coarse_angles: list[float],
) -> list[Candidate]:
    coarse_threshold, per_pose_limit, global_limit = coarse_search_limits(matching)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pose_kernels = []
    for scale in coarse_scales:
        for angle in coarse_angles:
            kernel = build_pose_kernel(features, angle, scale)
            pose_kernels.append((scale, angle, kernel))
            
    if not pose_kernels:
        return []

    max_pad_left = max(k.anchor_x for _, _, k in pose_kernels)
    max_pad_right = max(k.width - 1 - k.anchor_x for _, _, k in pose_kernels)
    max_pad_top = max(k.anchor_y for _, _, k in pose_kernels)
    max_pad_bottom = max(k.height - 1 - k.anchor_y for _, _, k in pose_kernels)
    
    num_poses = len(pose_kernels)

    with torch.no_grad():
        out = _sparse_pose_score_maps(
            responses,
            pose_kernels,
            max_pad_left,
            max_pad_right,
            max_pad_top,
            max_pad_bottom,
            device,
        )

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
        
        candidates = []
        for idx in range(len(scores)):
            p_idx = pose_indices[idx]
            scale, angle, kernel = pose_kernels[p_idx]
            left = xs[idx]
            top = ys[idx]
            score = float(scores[idx])
            
            cx = (left + kernel.anchor_x - max_pad_left) / image_factor
            cy = (top + kernel.anchor_y - max_pad_top) / image_factor
            candidates.append(Candidate(cx=cx, cy=cy, score=score, angle=angle, scale=scale))
            
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:global_limit]

