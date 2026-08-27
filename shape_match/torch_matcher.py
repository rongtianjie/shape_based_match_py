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


def coarse_search_pytorch(
    features: ModelFeatures,
    responses: ResponseImage,
    pattern: PatternConfig,
    matching: MatchConfig,
    image_factor: float,
    coarse_scales: list[float],
    coarse_angles: list[float],
    min_visible_ratio: float = MIN_VISIBLE_RATIO,
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

    responses_tensor = torch.from_numpy(np.ascontiguousarray(responses)).to(
        device=device, dtype=torch.float32
    )
    if responses.dtype == np.uint8:
        responses_tensor.mul_(1.0 / 255.0)

    orig_h, orig_w = responses.shape[1:]
    padded_responses = F.pad(
        responses_tensor,
        (max_pad_left, max_pad_right, max_pad_top, max_pad_bottom),
        mode="constant",
        value=0.0,
    )
    fov_mask = torch.zeros(
        (padded_responses.shape[1], padded_responses.shape[2]),
        device=device,
        dtype=torch.float32,
    )
    fov_mask[max_pad_top : max_pad_top + orig_h, max_pad_left : max_pad_left + orig_w] = 1.0

    candidates = []

    with torch.no_grad():
        for scale, angle, kernel in pose_kernels:
            out_h = padded_responses.shape[1] - kernel.height + 1
            out_w = padded_responses.shape[2] - kernel.width + 1
            if out_h <= 0 or out_w <= 0:
                continue

            score_sum = torch.zeros((out_h, out_w), device=device, dtype=torch.float32)
            vis_sum = torch.zeros((out_h, out_w), device=device, dtype=torch.float32)
            min_vis = max(
                1,
                min(MIN_FEATURES, kernel.feature_count),
                int(round(min_visible_ratio * kernel.feature_count)),
            )

            for label, template_k in enumerate(kernel.kernels):
                if template_k is None:
                    continue
                ys, xs = np.nonzero(template_k)
                values = template_k[ys, xs]
                source = padded_responses[label]
                for y, x, value in zip(ys, xs, values):
                    score_sum.add_(
                        source[int(y) : int(y) + out_h, int(x) : int(x) + out_w],
                        alpha=float(value),
                    )
                    vis_sum.add_(
                        fov_mask[int(y) : int(y) + out_h, int(x) : int(x) + out_w],
                        alpha=float(value),
                    )

            score_map = score_sum / torch.clamp(vis_sum, min=1.0)
            score_map[vis_sum < float(min_vis)] = 0.0
            score_map = torch.clamp(score_map, 0.0, 1.0)

            # Local peaks using max pool
            max_pooled = F.max_pool2d(
                score_map.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2
            ).squeeze(0).squeeze(0)
            is_peak = (score_map >= max_pooled - 1e-7) & (score_map >= coarse_threshold)

            peak_ys, peak_xs = torch.where(is_peak)
            if len(peak_xs) > 0:
                peak_scores = score_map[peak_ys, peak_xs]
                if len(peak_scores) > per_pose_limit:
                    topk_scores, topk_idx = torch.topk(peak_scores, per_pose_limit)
                    peak_scores = topk_scores
                    peak_ys = peak_ys[topk_idx]
                    peak_xs = peak_xs[topk_idx]

                scores_np = peak_scores.cpu().numpy()
                xs_np = peak_xs.cpu().numpy()
                ys_np = peak_ys.cpu().numpy()

                for idx in range(len(scores_np)):
                    cx = (xs_np[idx] + kernel.anchor_x - max_pad_left) / image_factor
                    cy = (ys_np[idx] + kernel.anchor_y - max_pad_top) / image_factor
                    candidates.append(
                        Candidate(
                            cx=cx,
                            cy=cy,
                            score=float(scores_np[idx]),
                            angle=angle,
                            scale=scale,
                        )
                    )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:global_limit]




