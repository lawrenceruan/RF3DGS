#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

import kornia

C1 = 0.01 ** 2
C2 = 0.03 ** 2

def sample_pixels(image, num_samples):
    """
    Randomly sample a set of pixels from an image.
    
    Args:
        image (torch.Tensor): The input image with shape [C, H, W].
        num_samples (int): The number of pixels to sample.
        
    Returns:
        torch.Tensor: A tensor of sampled pixel values with shape [num_samples, C].
    """
    # Get the image dimensions
    C, H, W = image.shape
    
    # Generate random row and column indices
    row_idx = torch.randint(0, H, (num_samples,))
    col_idx = torch.randint(0, W, (num_samples,))
    
    # Extract the sampled pixel values
    sampled_pixels = image[:, row_idx, col_idx]
    
    return sampled_pixels

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

def l1_loss_min(network_output, gt_pseudo_warped, gt_pseudo_learned):
    l1_loss_gt_pseudo_warped = torch.abs((network_output - gt_pseudo_warped))
    l1_loss_gt_pseudo_learned = torch.abs((network_output - gt_pseudo_learned))
    l1_loss_min = torch.minimum(l1_loss_gt_pseudo_learned, l1_loss_gt_pseudo_warped)
    return l1_loss_min.mean()

def l2_loss_min(network_output, gt_image, pseudo_gt_image_warped):
    l2_loss_gt = (network_output - gt_image) ** 2
    l2_loss_gt_pseudo_learned = (network_output - pseudo_gt_image_warped) ** 2
    l2_loss_min = torch.minimum(l2_loss_gt, l2_loss_gt_pseudo_learned)
    return l2_loss_min.mean()

def normalized_enhanced_loss(network_output, gt, pseudo_gt):
    # Base components
    l2_gt = (network_output - gt)**2
    l2_pseudo = (network_output - pseudo_gt)**2
    
    # 1. Adaptive normalization for minimum L2 term
    min_l2 = torch.minimum(l2_gt, l2_pseudo)
    norm_min_l2 = min_l2 / (torch.mean(min_l2.detach()) + 1e-8)  # Prevent division by zero

    # 2. Edge loss with gradient normalization
    def compute_grad(x):
        return torch.abs(x - F.avg_pool2d(x, 3, stride=1, padding=1))
    
    grad_output = compute_grad(network_output)
    grad_pseudo = compute_grad(pseudo_gt)
    edge_loss = (grad_output - grad_pseudo)**2
    norm_edge = edge_loss / (torch.mean(grad_pseudo**2) + 1e-8)  # Normalize by reference gradients

    # 3. Reflection-aware masking with dynamic scaling
    reflection_mask = torch.sigmoid(10 * (torch.mean(torch.abs(gt - pseudo_gt), dim=1, keepdim=True) - 0.1))
    reflection_term = reflection_mask * l2_pseudo
    norm_reflection = reflection_term / (torch.mean(reflection_mask) + 1e-8)

    # 4. Auto-balanced combination
    total_loss = 0.4*norm_min_l2 + 0.3*norm_edge + 0.3*norm_reflection
    return total_loss.mean()

def l2_loss_min_sample(network_output, gt_pseudo_warped, gt_pseudo_learned, num_samples=100):
    l2_loss_gt_pseudo_warped = (network_output - gt_pseudo_warped) ** 2
    l2_loss_gt_pseudo_learned = (network_output - gt_pseudo_learned) ** 2
    l2_loss_min = torch.minimum(l2_loss_gt_pseudo_learned, l2_loss_gt_pseudo_warped)
    l2_loss_min = sample_pixels(l2_loss_min, num_samples=num_samples)
    return l2_loss_min.mean()

def l1_loss_sample(network_output, gt, num_samples=100, weight_label='normal'):
    if weight_label=='highlight':
        weight = gt
        l1_loss = torch.abs(network_output - gt) * weight
    elif weight_label=='lowlight':
        weight = torch.log(1 + 1 / (gt + 1e-6))  # Logarithmic weighting
        l1_loss = torch.abs(network_output - gt) * weight
    else:
        l1_loss = torch.abs(network_output - gt)
    l1_loss = sample_pixels(l1_loss, num_samples=num_samples)
    return l1_loss.mean()

def l2_loss_sample(network_output, gt, num_samples=100, weight_label='normal'):
    if weight_label=='highlight':
        weight = gt
        l2_loss = (network_output - gt) ** 2 * weight
    elif weight_label=='lowlight':
        weight = torch.log(1 + 1 / (gt + 1e-6))  # Logarithmic weighting
        l2_loss = (network_output - gt) ** 2 * weight
    else:
        l2_loss = (network_output - gt) ** 2
    l2_loss = sample_pixels(l2_loss, num_samples=num_samples)
    return l2_loss.mean()

def huber_loss_sample(network_output, gt, num_samples=100):
    huber_loss = torch.nn.functional.huber_loss(sample_pixels(network_output, num_samples=num_samples),
                                                sample_pixels(gt, num_samples=num_samples))
    return huber_loss.mean()

def l1_loss_min_sample(network_output, gt_1, gt_2):
    l1_loss_gt_1 = torch.abs((network_output - gt_1))
    l1_loss_gt_2 = torch.abs((network_output - gt_2))
    l1_loss_min = torch.minimum(l1_loss_gt_1, l1_loss_gt_2)
    l1_loss_min = sample_pixels(l1_loss_min, num_samples=100)
    return l1_loss_min.mean()

def l1_loss_min_triple(network_output, gt_1, gt_2, gt_3):
    l1_loss_gt_1 = torch.abs((network_output - gt_1))
    l1_loss_gt_2 = torch.abs((network_output - gt_2))
    l1_loss_gt_3 = torch.abs((network_output - gt_3))
    l1_loss_min = torch.minimum(torch.minimum(l1_loss_gt_1, l1_loss_gt_2), l1_loss_gt_3)
    return l1_loss_min.mean()

def l1_grad_loss_min(network_output, gt_pseudo_warped, gt_pseudo_learned):
    l1_loss_gt_pseudo_warped = torch.abs(kornia.filters.sobel(network_output) - kornia.filters.sobel(gt_pseudo_warped))
    l1_loss_gt_pseudo_learned = torch.abs(kornia.filters.sobel(network_output) - kornia.filters.sobel(gt_pseudo_learned))
    l1_loss_min = torch.minimum(l1_loss_gt_pseudo_warped, l1_loss_gt_pseudo_learned)
    return l1_loss_min.mean()

def l1_loss_min_triple(network_output, gt, gt_pseudo_warped, gt_pseudo_learned):
    l1_loss_gt = torch.abs((network_output - gt))
    l1_loss_gt_pseudo_warped = torch.abs((network_output - gt_pseudo_warped))
    l1_loss_gt_pseudo_learned = torch.abs((network_output - gt_pseudo_learned))
    l1_loss_min = torch.minimum(torch.minimum(l1_loss_gt_pseudo_learned, l1_loss_gt_pseudo_warped), l1_loss_gt)
    return l1_loss_min.mean()

def l2_loss_min_triple(network_output, gt, gt_pseudo_warped, gt_pseudo_learned):
    l2_loss_gt = (network_output - gt) ** 2
    l2_loss_gt_pseudo_warped = (network_output - gt_pseudo_warped) ** 2
    l2_loss_gt_pseudo_learned = (network_output - gt_pseudo_learned) ** 2
    l2_loss_min = torch.minimum(torch.minimum(l2_loss_gt_pseudo_learned, l2_loss_gt_pseudo_warped), l2_loss_gt)
    return l2_loss_min.mean()

def ssim_max(img1, img2, img3, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    loss_1 = _ssim(img1, img2, window, window_size, channel, size_average)
    loss_2 = _ssim(img1, img3, window, window_size, channel, size_average)
    
    return torch.maximum(loss_1, loss_2).mean()

def ssim_sample(img1, img2, window_size=11, size_average=True, num_samples=100):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_sample(img1, img2, window, window_size, channel, size_average, num_samples)

def _ssim_sample(img1, img2, window, window_size, channel, size_average=True, num_samples=100):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        ssim_map = sample_pixels(ssim_map, num_samples=num_samples)
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
