"""Experimental, fitting-view-only correspondence confidence. Disabled by default."""
from pathlib import Path
import torch
import torch.nn.functional as F
from custom_utils.frequency_prior import lowpass


def patch_vote(reference, sample, valid, window=9, threshold=.75, energy=.0015):
    weights = reference.new_tensor([.299, .587, .114])[:, None, None]
    a = (reference * weights).sum(0)[None, None]
    b = (sample * weights).sum(0)[None, None]
    pool = lambda x: F.avg_pool2d(x, window, 1, window//2)
    # Center local patches so slowly varying reflected illumination cannot win NCC.
    aa = (pool(a*a)-pool(a).square()).clamp_min(0)
    bb = (pool(b*b)-pool(b).square()).clamp_min(0)
    ab = pool(a*b)-pool(a)*pool(b)
    corr = ab / (aa*bb).clamp_min(1e-12).sqrt()
    ratio = (bb+1e-8)/(aa+1e-8)
    residual = pool((a-b).square())/(aa+bb+1e-8)
    good = (pool(valid.float()[None, None]) > .999) & (aa > energy**2)
    good &= (corr > threshold) & (ratio > .4) & (ratio < 2.5)
    return torch.where(good, torch.exp(-3*residual), 0)[0, 0]


def brightness_guard(observed, prior):
    # Conservative veto for areas the original prior substantially darkens.
    # This is a heuristic, not a transmission/reflection classifier.
    excess = (lowpass(observed, 4)-lowpass(prior, 4)).clamp_min(0).amax(0)
    return ((.08-excess)/.05).clamp(0, 1)


class CorrespondenceTargets:
    def __init__(self, cameras, cache):
        self.targets = {}
        for camera in cameras:
            record = torch.load(Path(cache)/(Path(camera.image_name).stem+'.pt'),
                                map_location='cpu', weights_only=True)
            image = camera.original_image[:3].detach()
            confidence = record['confidence'].float()
            if record['image_name'] != camera.image_name or confidence.shape != image.shape[-2:]:
                raise ValueError('Correspondence cache mismatch')
            if not torch.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
                raise ValueError('Invalid correspondence confidence')
            with torch.no_grad():
                guard = brightness_guard(image, camera.original_image_pseudo_warped[:3])
                self.targets[camera.image_name] = ((image-lowpass(image)).cpu(),
                                                  (confidence.to(image.device)*guard).cpu())

    def get(self, name, device):
        return tuple(t.to(device) for t in self.targets[name])
