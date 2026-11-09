"""Depth-checked RGB prior for the RF3DGS stage-one renderer."""
import torch
import torch.nn.functional as F


def camera_rays(height, width, fx, fy, device):
    y, x = torch.meshgrid(torch.arange(height, device=device),
                          torch.arange(width, device=device), indexing="ij")
    # The centered splat rasterizer maps NDC zero to (size - 1) / 2.
    return torch.stack(((x - (width - 1) / 2) / fx,
                        (y - (height - 1) / 2) / fy, torch.ones_like(x)))


def visible_sample(points, relative_pose, rgb, depth, fx, fy, tolerance, border_margin=0):
    """Gather source pixels that see the same surface as the target."""
    _, h, w = rgb.shape
    xyz = (relative_pose[:3, :3] @ points.flatten(1) +
           relative_pose[:3, 3:4]).reshape_as(points)
    z = xyz[2]
    u = fx * xyz[0] / z + (w - 1) / 2
    v = fy * xyz[1] / z + (h - 1) / 2
    valid = ((z > 0) & torch.isfinite(z) & (u >= border_margin) & (u <= w - 1 - border_margin) &
             (v >= border_margin) & (v <= h - 1 - border_margin))
    grid = torch.stack((2 * (u + 0.5) / w - 1, 2 * (v + 0.5) / h - 1), -1)
    grid = torch.where(valid[..., None], grid, 0)[None]
    # Check the full interpolation footprint to reject occluding depth edges.
    x0 = torch.where(valid, u, 0).floor().long()
    y0 = torch.where(valid, v, 0).floor().long()
    x1 = torch.where(valid, u, 0).ceil().long()
    y1 = torch.where(valid, v, 0).ceil().long()
    for y, x in ((y0, x0), (y0, x1), (y1, x0), (y1, x1)):
        d = depth[0, y, x]
        valid = valid & torch.isfinite(d) & (d > 0) & ((d - z).abs() <= tolerance * z)
    rgb = F.grid_sample(rgb[None], grid, align_corners=False, padding_mode="border")[0]
    return rgb, valid


def robust_prior(reference, samples):
    """Use the second darkest valid source, capped by reference luminance."""
    weights = reference.new_tensor([0.2126, 0.7152, 0.0722])[:, None, None]
    first = torch.full_like(reference[0], float("inf"))
    second = first.clone()
    darkest_rgb = reference
    prior = reference
    for rgb, valid in samples:
        score = (weights * rgb).sum(0)
        new_first = valid & (score < first)
        candidate = torch.where(new_first, first, score)
        update = valid & (candidate < second)
        prior = torch.where(update, torch.where(new_first, darkest_rgb, rgb), prior)
        second = torch.where(update, candidate, second)
        darkest_rgb = torch.where(new_first, rgb, darkest_rgb)
        first = torch.where(new_first, score, first)
    use_prior = second < (reference * weights).sum(0)
    return torch.where(use_prior, prior, reference), use_prior


def consensus_prior(reference, samples):
    """Select a complete RGB sample at the lower quartile of depth-valid views."""
    colors, scores, masks = [], [], []
    weights = reference.new_tensor([0.2126, 0.7152, 0.0722])[:, None, None]
    for rgb, valid in samples:
        colors.append(rgb)
        masks.append(valid)
        scores.append((rgb * weights).sum(0).masked_fill(~valid, float("inf")))
    if len(colors) < 2:
        return reference, torch.zeros_like(reference[0], dtype=torch.bool)
    support = torch.stack(masks).sum(0)
    # For 2--8 valid sources retain the original second-darkest rule.
    rank = ((support + 3) // 4).clamp_min(2) - 1
    ordered, indices = torch.stack(scores).sort(dim=0, stable=True)
    value = ordered.gather(0, rank[None])[0]
    index = indices.gather(0, rank[None])
    rgb = torch.stack(colors).gather(0, index[None].expand(1, 3, *index.shape[-2:]))[0]
    use = (support >= 2) & (value < (reference * weights).sum(0))
    return torch.where(use, rgb, reference), use
