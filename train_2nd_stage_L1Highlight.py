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

import os
import torch
from random import randint
from custom_utils.training_checkpoint import save_training_checkpoint, restore_training_checkpoint
from utils.loss_utils import l1_loss, l2_loss, ssim, ssim_max
from utils.loss_utils import l1_loss_min, l2_loss_min, l1_loss_sample, l2_loss_sample, l1_loss_min_sample, ssim_sample, l1_loss_min_triple, l2_loss_min_triple, normalized_enhanced_loss
from gaussian_renderer import render, network_gui
from pathlib import Path
from PIL import Image
from torch.nn import functional as F
from utils.general_utils import PILtoTorch
import sys
from scene.gaussian_model import GaussianModel
from scene.scene_2nd_stage import Scene
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

SPARSE_ADAM_AVAILABLE = False

## WEN Qiang
from custom_utils.frequency_prior import FrequencyTargets, lowpass, detail_loss
from custom_utils.correspondence_prior import CorrespondenceTargets
from custom_utils.sampled_ssim import ssim_sample_patches
from custom_utils.run_metadata import record_training_args
from custom_utils.ex_loss import ExclusionLoss, ExclusionLoss_sample
from custom_utils.tv_loss import total_variation_loss as criterion_tv
from custom_utils.bilateral_loss import BilateralLoss
from custom_utils.his_match import DifferentiableHistogramMatchingLoss as HistogramMatchingLoss

criterion_ex = ExclusionLoss()
criterion_ex_sample = ExclusionLoss_sample()
criterion_huber = torch.nn.HuberLoss(reduction='mean', delta=1.0)
criterion_bilateral_depth = BilateralLoss(gamma=2)
criterion_bilateral_beta = BilateralLoss(gamma=2)
criterion_hismatch = HistogramMatchingLoss().cuda()

def depth_weight(iteration):
    return 0.01

def tv_weight(iteration):
    if iteration < 1000:
        return 1
    else:
        return 0.0

def exc_weight(iteration):
    if iteration < 1000:
        return 0.01
    elif iteration > 1000 and iteration < 3000:
        return 0.1
    else:
        return 0.01

def pooled_rgb(image):
    return F.avg_pool2d(image, 8, stride=8, ceil_mode=True, count_include_pad=False)


@torch.no_grad()
def prior_color_targets(cameras, prior_root):
    targets = {}
    for cam in cameras:
        visible = cam.original_image_pseudo_warped.cuda()
        with Image.open(Path(prior_root) / 'pseudo_GT' / Path(cam.image_name).name) as im:
            legacy = PILtoTorch(im.convert("RGB"), (cam.image_width, cam.image_height)).to(visible)
        agree = (visible - legacy).abs().amax(0, keepdim=True) <= 0.05
        trusted = pooled_rgb(agree.float()) >= 0.75
        target = pooled_rgb((visible + legacy) * 0.5)
        targets[cam.image_name] = (target, trusted)
    return targets


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, prior_color_weight=0.0, histogram_weight=0.0, full_ssim=False, atr_until=3000, depth_smooth_weight=10.0, frequency_prior=False, frequency_cache="", frequency_weight=0.1, correspondence_prior=False, correspondence_cache="", correspondence_weight=0.1, sampled_ssim_patches=False):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians_T = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    gaussians_R = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    gaussians_Beta_R = GaussianModel(1, opt.optimizer_type)
    
    scene = Scene(dataset, gaussians_T, gaussians_R, gaussians_Beta_R)
        
    targets = {}
    if prior_color_weight:
        targets = prior_color_targets(scene.getTrainCameras(), dataset.pseudo_source_path)

    correspondence_targets = CorrespondenceTargets(scene.getTrainCameras(), correspondence_cache) if correspondence_prior else None
    frequency_targets = FrequencyTargets(scene.getTrainCameras(), frequency_cache) if frequency_prior else None

    gaussians_T.training_setup(opt)
    gaussians_R.training_setup(opt)
    gaussians_Beta_R.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=30000)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    if checkpoint:
        first_iter, viewpoint_stack = restore_training_checkpoint(
            checkpoint, (gaussians_T, gaussians_R, gaussians_Beta_R), opt,
            scene.getTrainCameras())
        if first_iter >= opt.iterations:
            raise ValueError("Checkpoint has already reached the requested iteration count")

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians_T.update_learning_rate(iteration)
        gaussians_R.update_learning_rate(iteration)
        gaussians_Beta_R.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians_T.oneupSHdegree()
            gaussians_R.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        rand_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg_T = render(viewpoint_cam, gaussians_T, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image_T, viewspace_point_tensor_T, visibility_filter_T, radii_T = render_pkg_T["render"], render_pkg_T["viewspace_points"], render_pkg_T["visibility_filter"], render_pkg_T["radii"]

        render_pkg_R = render(viewpoint_cam, gaussians_R, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image_R, viewspace_point_tensor_R, visibility_filter_R, radii_R = render_pkg_R["render"], render_pkg_R["viewspace_points"], render_pkg_R["visibility_filter"], render_pkg_R["radii"]

        render_pkg_Beta_R = render(viewpoint_cam, gaussians_Beta_R, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image_Beta_R, viewspace_point_tensor_Beta_R, visibility_filter_Beta_R, radii_Beta_R = render_pkg_Beta_R["render"], render_pkg_Beta_R["viewspace_points"], render_pkg_Beta_R["visibility_filter"], render_pkg_Beta_R["radii"]
        beta_map_one_channel_R = torch.sigmoid(image_Beta_R[0])

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        pseudo_gt_image_warped = viewpoint_cam.original_image_pseudo_warped.cuda()
        pseudo_gt_image_learned = viewpoint_cam.original_image_pseudo_learned.cuda()

        num_samples = 3000

        fake_full = image_T + image_R * beta_map_one_channel_R

        Ll1 = l1_loss(fake_full, gt_image)
        if full_ssim:
            # Preserve the CPU draws that the sampled loss would have consumed.
            torch.randint(0, gt_image.shape[-2], (100,))
            torch.randint(0, gt_image.shape[-1], (100,))
            ssim_value = ssim(fake_full, gt_image)
        elif sampled_ssim_patches:
            ssim_value = ssim_sample_patches(fake_full, gt_image)
        else:
            ssim_value = ssim_sample(fake_full, gt_image)
        
        loss_full = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        ## Add more loss information for reconstructing image R
        fake_full_R = image_T.detach() + image_R * beta_map_one_channel_R

        loss_full_R = l1_loss_sample(fake_full_R, gt_image, num_samples=num_samples, weight_label='highlight')
        
        loss = loss_full + loss_full_R

        # # Depth regularization
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg_T["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss_T = image_T.new_zeros(())
        prior_for_exclusion = pseudo_gt_image_warped
        if frequency_prior:
            low_gt, low_prior, target_hp, confidence = frequency_targets.get(viewpoint_cam.image_name, image_T.device)
            low_T = lowpass(image_T)
            prior_for_exclusion = low_prior
            if iteration < atr_until:
                loss_T = l2_loss_min(low_T, low_gt, low_prior)
            ramp = min(1.0, max(0.0, (iteration-1000)/2000))
            loss_T = loss_T + frequency_weight*ramp*detail_loss(image_T-low_T, target_hp, confidence)
        elif iteration < atr_until:
            loss_T = l2_loss_min(image_T, gt_image, pseudo_gt_image_warped)
        if correspondence_prior:
            target_hp, confidence = correspondence_targets.get(viewpoint_cam.image_name, image_T.device)
            ramp = min(1.0, max(0.0, (iteration-1000)/2000))
            loss_T = loss_T + correspondence_weight*ramp*detail_loss(image_T-lowpass(image_T), target_hp, confidence)
        if histogram_weight:
            loss_T = loss_T + histogram_weight * criterion_hismatch(
                image_T.unsqueeze(0), gt_image.unsqueeze(0))

        if prior_color_weight and iteration >= 3000:
            target, trusted = targets[viewpoint_cam.image_name]
            # Penalize broad color drift, with a 3% dead band for exposure differences.
            drift = ((pooled_rgb(image_T) - target).abs() - 0.03).clamp_min(0)
            loss_T += prior_color_weight * (drift * trusted).sum() / (3 * trusted.sum().clamp_min(1))

        loss_T_depth = criterion_bilateral_depth(render_pkg_T['depth'][0].unsqueeze(0),
                                                 render_pkg_T['depth'][0].unsqueeze(0).unsqueeze(-1)) * depth_smooth_weight

        loss_R_depth = criterion_bilateral_depth(render_pkg_R['depth'][0].unsqueeze(0),
                                                 render_pkg_R['depth'][0].unsqueeze(0).unsqueeze(-1)) * depth_smooth_weight

        loss_beta = criterion_tv(beta_map_one_channel_R.unsqueeze(0))
                    
        loss_exc = (criterion_ex(image_T.unsqueeze(0), image_R.unsqueeze(0)) + 
                    criterion_ex(prior_for_exclusion.unsqueeze(0), image_R.unsqueeze(0))) * exc_weight(iteration)

        loss += loss_T + loss_T_depth + loss_R_depth + loss_exc + loss_beta

        if iteration % 500 == 0:

            print(f'loss_T: {loss_T.item()}, loss_T_depth: {loss_T_depth.item()}, loss_R_depth: {loss_R_depth.item()},\
                    loss_exc: {loss_exc.item()}, loss_beta: {loss_beta.item()}')

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians_T.max_radii2D[visibility_filter_T] = torch.max(gaussians_T.max_radii2D[visibility_filter_T], radii_T[visibility_filter_T])
                gaussians_T.add_densification_stats(viewspace_point_tensor_T, visibility_filter_T)

                gaussians_R.max_radii2D[visibility_filter_R] = torch.max(gaussians_R.max_radii2D[visibility_filter_R], radii_R[visibility_filter_R])
                gaussians_R.add_densification_stats(viewspace_point_tensor_R, visibility_filter_R)

                gaussians_Beta_R.max_radii2D[visibility_filter_Beta_R] = torch.max(gaussians_Beta_R.max_radii2D[visibility_filter_Beta_R], radii_Beta_R[visibility_filter_Beta_R])
                gaussians_Beta_R.add_densification_stats(viewspace_point_tensor_Beta_R, visibility_filter_Beta_R)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians_T.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii_T)
                    gaussians_R.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii_R)
                    gaussians_Beta_R.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii_Beta_R)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians_T.reset_opacity()
                    gaussians_R.reset_opacity()
                    gaussians_Beta_R.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians_T.optimizer.step()
                gaussians_T.optimizer.zero_grad(set_to_none = True)

                gaussians_R.optimizer.step()
                gaussians_R.optimizer.zero_grad(set_to_none = True)

                gaussians_Beta_R.optimizer.step()
                gaussians_Beta_R.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                save_training_checkpoint(
                    Path(scene.model_path) / f"chkpnt{iteration}.pth",
                    (gaussians_T, gaussians_R, gaussians_Beta_R), iteration,
                    scene.getTrainCameras(), viewpoint_stack,
                    optimizer_step_pending=iteration == opt.iterations)

def prepare_output_and_logger(args):    
    args.model_path = os.path.join("./output/", args.scene_name)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 3_000, 5_000, 7_000, 10_000, 15_000, 20_000, 30_000, 40_000, 50_000, 60_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 3_000, 5_000, 7_000, 10_000, 15_000, 20_000, 30_000, 40_000, 50_000, 60_000])

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--prior_color_weight", type=float, default=0.0)
    parser.add_argument("--histogram_weight", type=float, default=0.0)
    parser.add_argument("--full_ssim", action="store_true")
    parser.add_argument("--atr_until", type=int, default=3000)
    parser.add_argument("--depth_smooth_weight", type=float, default=10.0)
    parser.add_argument("--frequency_prior", action="store_true", help="Optional confidence-gated frequency-separated prior")
    parser.add_argument("--frequency_cache", default="", help="Fitting-view confidence directory from prepare_frequency.py")
    parser.add_argument("--frequency_weight", type=float, default=0.1)
    parser.add_argument("--correspondence_prior", action="store_true", help="Experimental validated correspondence detail; legacy losses retained")
    parser.add_argument("--correspondence_cache", default="")
    parser.add_argument("--correspondence_weight", type=float, default=.1)
    parser.add_argument("--sampled_ssim_patches", action="store_true", help="Optional computation of sampled SSIM only at its sampled neighborhoods")
    args = parser.parse_args(sys.argv[1:])
    if args.full_ssim and args.sampled_ssim_patches:
        parser.error("Choose full-image SSIM or sampled SSIM patches, not both")
    if args.frequency_weight < 0 or (args.frequency_prior and not args.frequency_cache):
        parser.error("frequency_prior requires a cache and nonnegative weight")
    if args.correspondence_weight < 0 or (args.correspondence_prior and not args.correspondence_cache):
        parser.error("correspondence_prior requires a cache and nonnegative weight")
    if args.correspondence_prior and args.frequency_prior:
        parser.error("Test correspondence and frequency modes separately")
    if args.histogram_weight < 0 or args.atr_until < 0 or args.depth_smooth_weight < 0:
        parser.error("Loss weights and iteration limits must be nonnegative")
    if args.prior_color_weight < 0 or (args.prior_color_weight and args.pseudo_warp_folder not in ("pseudo_GT_visible", "pseudo_GT_consensus")):
        parser.error("prior_color_weight requires a nonnegative weight and a depth-checked prior when enabled")
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    record_training_args(args, __file__, sys.argv[1:])

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.prior_color_weight, args.histogram_weight, args.full_ssim, args.atr_until, args.depth_smooth_weight, args.frequency_prior, args.frequency_cache, args.frequency_weight, args.correspondence_prior, args.correspondence_cache, args.correspondence_weight, args.sampled_ssim_patches)

    # All done
    print("\nTraining complete.")
