import numpy as np

from shape_match.types import (
    Candidate,
    MatchConfig,
    ModelFeatures,
    PatternConfig,
    PoseKernel,
    ResponseImage,
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
    max_anchor_x: int,
    max_anchor_y: int,
    pad_width: int,
    pad_height: int,
    device: "torch.device",
) -> "torch.Tensor":
    """Accumulate sparse pose kernels without launching a dense convolution.

    A pose kernel contains at most one non-zero value per extracted feature,
    while a dense convolution evaluates every pixel in its bounding box.  A
    shifted view of the response map is therefore the exact same correlation
    operation and is substantially cheaper for shape templates.
    """
    responses_tensor = torch.from_numpy(np.ascontiguousarray(responses)).to(
        device=device, dtype=torch.float32
    )
    if responses.dtype == np.uint8:
        responses_tensor.mul_(1.0 / 255.0)
    output_height = responses.shape[1] - pad_height + 1
    output_width = responses.shape[2] - pad_width + 1
    score_maps: list["torch.Tensor"] = []

    for _, _, kernel in pose_kernels:
        score_map = torch.zeros(
            (output_height, output_width), device=device, dtype=torch.float32
        )
        start_y = max_anchor_y - kernel.anchor_y
        start_x = max_anchor_x - kernel.anchor_x
        inverse_count = 1.0 / float(kernel.feature_count)

        for label, template in enumerate(kernel.kernels):
            if template is None:
                continue
            ys, xs = np.nonzero(template)
            values = template[ys, xs] * inverse_count
            source = responses_tensor[label]
            for y, x, value in zip(ys, xs, values):
                score_map.add_(
                    source[
                        start_y + int(y) : start_y + int(y) + output_height,
                        start_x + int(x) : start_x + int(x) + output_width,
                    ],
                    alpha=float(value),
                )
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

    with torch.no_grad():
        if device.type == "cuda":
            # The kernels are sparse (normally <= 256 feature points), so
            # shifted accumulation avoids the prohibitively expensive dense
            # 200x200-ish convolution used by the previous implementation.
            out = _sparse_pose_score_maps(
                responses,
                pose_kernels,
                max_anchor_x,
                max_anchor_y,
                pad_width,
                pad_height,
                device,
            )
        else:
            # Keep a dense CPU fallback.  The public matcher only selects the
            # PyTorch path when CUDA is available, but this makes the helper
            # behave sensibly when called directly in a CPU-only environment.
            weights = np.zeros(
                (num_poses, num_orientations, pad_height, pad_width),
                dtype=np.float32,
            )
            for i, (_, _, kernel) in enumerate(pose_kernels):
                start_y = max_anchor_y - kernel.anchor_y
                start_x = max_anchor_x - kernel.anchor_x
                inverse_count = 1.0 / float(kernel.feature_count)
                for j, template in enumerate(kernel.kernels):
                    if template is not None:
                        weights[
                            i,
                            j,
                            start_y : start_y + kernel.height,
                            start_x : start_x + kernel.width,
                        ] = template * inverse_count
            weights_tensor = torch.from_numpy(weights).to(device)
            responses_tensor = torch.from_numpy(responses).unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            if responses.dtype == np.uint8:
                responses_tensor.mul_(1.0 / 255.0)
            out = F.conv2d(responses_tensor, weights_tensor).squeeze(0)

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
            scale, angle, _ = pose_kernels[p_idx]
            left = xs[idx]
            top = ys[idx]
            score = float(scores[idx])
            
            cx = (left + max_anchor_x) / image_factor
            cy = (top + max_anchor_y) / image_factor
            candidates.append(Candidate(cx=cx, cy=cy, score=score, angle=angle, scale=scale))
            
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:global_limit]
