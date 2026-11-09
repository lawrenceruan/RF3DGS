"""Optional frequency-separated supervision; no changes when disabled."""
from pathlib import Path
import torch
import torch.nn.functional as F


def lowpass(image, sigma=2.0):
    radius = int(3 * sigma + .5)
    x = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    k = torch.exp(-.5 * (x / sigma).square()); k = k / k.sum()
    c = image.shape[-3]; batch = image.unsqueeze(0) if image.ndim == 3 else image
    value = F.conv2d(F.pad(batch, (radius, radius, 0, 0), mode='reflect'),
                     k.reshape(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    value = F.conv2d(F.pad(value, (0, 0, radius, radius), mode='reflect'),
                     k.reshape(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return value[0] if image.ndim == 3 else value


def depth_interior(depth, tolerance=.03, radius=3):
    d = depth.unsqueeze(0) if depth.ndim == 3 else depth
    good = torch.isfinite(d) & (d > 0)
    clean = torch.where(good, d, torch.zeros_like(d))
    hi = F.max_pool2d(clean, 2*radius+1, 1, radius)
    lo = -F.max_pool2d(-clean, 2*radius+1, 1, radius)
    mask = good & ((hi-lo) <= tolerance * clean.clamp_min(1e-6))
    return mask[0, 0]


def texture_agreement(reference_hp, warped_hp, valid):
    """Local signed correlation and relative residual; weak texture gives no vote."""
    w = reference_hp.new_tensor([.299, .587, .114])[:, None, None]
    a = (reference_hp*w).sum(0)[None,None]
    b = (warped_hp*w).sum(0)[None,None]
    pool = lambda x: F.avg_pool2d(x, 5, 1, 2)
    aa, bb, ab = pool(a*a), pool(b*b), pool(a*b)
    corr = ab / (aa*bb).clamp_min(1e-12).sqrt()
    ratio = (bb+1e-8)/(aa+1e-8)
    residual = pool((a-b).square())/(aa+bb+1e-8)
    support = pool(valid.float()[None,None]) > .999
    accept = support & (aa > .003**2) & (corr > .6) & (ratio > .25) & (ratio < 4)
    return torch.where(accept, torch.exp(-3*residual), 0)[0,0]


class FrequencyTargets:
    def __init__(self, cameras, cache):
        cache = Path(cache)
        if not cache.is_dir(): raise ValueError('frequency_prior requires a prepared frequency_cache directory')
        self.targets = {}
        for camera in cameras:
            record = torch.load(cache/(Path(camera.image_name).stem+'.pt'), map_location='cpu', weights_only=True)
            if record['image_name'] != camera.image_name: raise ValueError('Frequency cache name mismatch')
            confidence = record['confidence'].float()
            image = camera.original_image[:3].detach()
            prior = camera.original_image_pseudo_warped[:3].detach()
            if confidence.shape != image.shape[-2:] or not torch.isfinite(confidence).all():
                raise ValueError('Invalid frequency confidence map')
            if (confidence < 0).any() or (confidence > 1).any(): raise ValueError('Invalid confidence range')
            with torch.no_grad():
                smooth = lowpass(image)
                self.targets[camera.image_name] = tuple(x.cpu() for x in (smooth, lowpass(prior), image-smooth, confidence))

    def get(self, name, device):
        return tuple(x.to(device) for x in self.targets[name])


def detail_loss(predicted_hp, target_hp, confidence):
    # Full-image normalization makes sparse or uncertain support contribute less.
    return ((predicted_hp-target_hp).abs()*confidence[None]).mean()
