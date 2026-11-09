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
from scene.scene_1st_stage import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import math

from nerf.run_nerf_helpers import *
import cv2

from pathlib import Path
from custom_utils.visible_prior import camera_rays, visible_sample, robust_prior, consensus_prior
from functools import reduce

NORM_MAX = -math.inf
NORM_MIN = math.inf

def are_matrices_similar(matrix1, matrix2, threshold):
    # Ensure the matrices are NumPy arrays
    matrix1 = np.array(matrix1.cpu())
    matrix2 = np.array(matrix2.cpu())
    
    # Compute the difference
    difference = matrix1 - matrix2
    
    # Calculate the Frobenius norm of the difference
    norm = np.linalg.norm(difference)

    global NORM_MAX
    global NORM_MIN

    if norm > NORM_MAX:
        NORM_MAX = norm
    if norm < NORM_MIN:
        NORM_MIN = norm
    
    # Check if the norm is within the threshold
    return norm < threshold


def get_pixel_points(depth, coords, K, pose):

    depth = depth.view(-1, 1)
    pixel_uv = (
        torch.cat(
            (
                coords,
                torch.tensor(1).view(1, 1).expand(coords.size(0), -1).to(coords.device),
            ),
            dim=-1,
        ).to(depth.device)
        * depth
    ).unsqueeze(-1)
    points_cam = torch.inverse(K).unsqueeze(0).to(pixel_uv.device) @ pixel_uv

    points_cam = torch.cat(
        (
            points_cam,
            torch.tensor(1)
            .view(1, 1, 1)
            .expand(points_cam.size(0), 1, 1)
            .to(points_cam.device),
        ),
        dim=1,
    )

    points_world = pose.unsqueeze(0).to(points_cam.device) @ points_cam
    points_world = points_world.squeeze(-1)[..., :3]

    return points_world


def project_points_image(points, pose, K, image_width, image_height):
    points = torch.cat(
        (
            points,
            torch.tensor([1]).view(1, 1).expand(points.size(0), -1).to(points.device),
        ),
        dim=1,
    )

    w2c = torch.inverse(pose)
    points_c = w2c.unsqueeze(0) @ points.unsqueeze(-1)
    points_c = points_c[:, :3]

    points_img = K.unsqueeze(0) @ points_c
    points_img = points_img.squeeze(-1)

    points_uv = points_img[:, :2] / points_img[:, 2:]

    valid_points_image_mask_x = torch.logical_and(
        (points_uv[:, 0] > 0), (points_uv[:, 0] < (image_width - 1))
    )
    valid_points_image_mask_y = torch.logical_and(
        (points_uv[:, 1] > 0), (points_uv[:, 1] < (image_height - 1))
    )

    valid_points_image_mask = torch.all(
        torch.stack((valid_points_image_mask_x, valid_points_image_mask_y), dim=-1),
        dim=-1,
    )

    points_uv = points_uv.view(image_height, image_width, 2)
    valid_points_image_mask = valid_points_image_mask.view(image_height, image_width)

    return points_uv, valid_points_image_mask


def project_points_image_show(
    points, pose, K, image_width, image_height, image=None, path=None
):
    points = torch.cat(
        (
            points,
            torch.tensor([1]).view(1, 1).expand(points.size(0), -1).to(points.device),
        ),
        dim=1,
    )

    w2c = torch.inverse(pose)
    points_c = w2c.unsqueeze(0) @ points.unsqueeze(-1)
    points_c = points_c[:, :3]

    points_img = K.unsqueeze(0) @ points_c
    points_img = points_img.squeeze(-1)

    points_uv = points_img[:, :2] / points_img[:, 2:]

    if image is not None:

        image_show = image.clone()

        valid_points_image_mask_x = torch.logical_and(
            (points_uv[:, 0] > 0), (points_uv[:, 0] < (image_width - 1))
        )
        valid_points_image_mask_y = torch.logical_and(
            (points_uv[:, 1] > 0), (points_uv[:, 1] < (image_height - 1))
        )

        valid_points_image_mask = torch.all(
            torch.stack((valid_points_image_mask_x, valid_points_image_mask_y), dim=-1),
            dim=-1,
        )

        valid_points_image = points_uv[valid_points_image_mask]

        valid_points_image = valid_points_image.long()
        image_show[0][(valid_points_image[:, 1], valid_points_image[:, 0])] = 0
        image_show[1][(valid_points_image[:, 1], valid_points_image[:, 0])] = 1
        image_show[2][(valid_points_image[:, 1], valid_points_image[:, 0])] = 0

        gt_cv2 = np.uint8(
            torch.clamp(image_show.permute(1, 2, 0)[..., [2, 1, 0]], 0, 255)
            .cpu()
            .numpy()
            * 255
        )
        cv2.imwrite(path, gt_cv2)

    return points_uv


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, threshold, file_name, pixel_center_offset=0.0):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))

    gt_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :]
        gt_list.append(gt)

    image_height = gt_list[0].size(1)
    image_width = gt_list[0].size(2)

    camera_sample = views[0]
    focal_x = fov2focal(camera_sample.FoVx, camera_sample.image_width)
    focal_y = fov2focal(camera_sample.FoVy, camera_sample.image_height)

    K = (
        torch.tensor(
            [
                [focal_x, 0, 0.5 * camera_sample.image_width],
                [0, focal_y, 0.5 * camera_sample.image_height],
                [0, 0, 1],
            ]
        )
        .float()
        .to(background.device)
    )

    coords = torch.stack(
        torch.meshgrid(
            torch.linspace(
                0, camera_sample.image_height - 1, camera_sample.image_height
            ),
            torch.linspace(0, camera_sample.image_width - 1, camera_sample.image_width),
        ),
        -1,
    )[
        ..., [1, 0]
    ]  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)

    pose_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        R_w2c = torch.from_numpy(np.transpose(view.R)).float().to(background.device)
        T_w2c = torch.from_numpy(view.T).float().view(3, 1).to(background.device)
        RT_w2c = torch.cat(
            (
                torch.cat((R_w2c, T_w2c), dim=-1),
                torch.tensor([0, 0, 0, 1]).float().view(1, 4).to(R_w2c.device),
            ),
            dim=0,
        )

        RT_c2w = torch.inverse(RT_w2c)
        pose_list.append(RT_c2w)

    points_np_list = []
    # rays_o_np_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        depth_path = os.path.join(render_path, 'Depth_pth', view.image_name.split('.')[0] + ".pth")
        depth = torch.load(depth_path)
        depth = depth.permute(1, 2, 0)

        # get points
        points_world = get_pixel_points(depth, coords, K, pose_list[idx])

        warp_image_list = [gt_list[idx]]

        invalid_mask_lsit = []

        for img_idx, img_pose in enumerate(pose_list):
            
            if img_idx is not idx and are_matrices_similar(pose_list[idx], pose_list[img_idx], threshold):

                points_uv, valid_points_image_mask = project_points_image(
                    points_world,
                    img_pose,
                    K,
                    image_width,
                    image_height,
                )

                # warp the images

                source_image = gt_list[img_idx].unsqueeze(0)

                points_uv_x = (points_uv[..., 0] + pixel_center_offset) / image_width * 2 - 1
                points_uv_y = (points_uv[..., 1] + pixel_center_offset) / image_height * 2 - 1
                points_uv_xy = torch.stack(
                    (points_uv_x, points_uv_y), dim=-1
                ).unsqueeze(0)

                image_warp = torch.nn.functional.grid_sample(
                    source_image,
                    points_uv_xy,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=None,
                ).squeeze(0)

                image_warp[:, torch.logical_not(valid_points_image_mask)] = 1

                warp_image_list.append(image_warp)
                invalid_mask_lsit.append(torch.logical_not(valid_points_image_mask))

        warp_image_list = torch.stack(warp_image_list)
        min_warp_image = torch.min(warp_image_list, dim=0)[0]
        invalid_mask = reduce(torch.logical_xor, invalid_mask_lsit)

        min_warp_image_save_path = os.path.join(
            render_path, "{0:05d}".format(idx) + "_warp_min.png"
        )

        invalid_mask_save_path = os.path.join(
            render_path, "{0:05d}".format(idx) + "_invalid_mask.png"
        )

        pseudo_gt_save_dir = os.path.join(
            render_path, file_name
        )

        if not os.path.exists(pseudo_gt_save_dir):
            os.makedirs(pseudo_gt_save_dir)
        
        min_warp_image_cv2 = np.uint8(
            torch.clamp(min_warp_image.permute(1, 2, 0)[..., [2, 1, 0]], 0, 1)
            .cpu()
            .numpy()
            * 255
        )

        invalid_mask_cv2 = np.uint8(
            (1 - torch.clamp(invalid_mask, 0, 1))
            .cpu()
            .numpy()
            * 255
        )

        # cv2.imwrite(min_warp_image_save_path, min_warp_image_cv2)
        # cv2.imwrite(invalid_mask_save_path, invalid_mask_cv2)
        cv2.imwrite(os.path.join(pseudo_gt_save_dir, view.image_name), min_warp_image_cv2)
        # cv2.imwrite(os.path.join(pseudo_gt_save_dir, view.image_name.replace('frame', 'mask')), invalid_mask_cv2)


def render_visible_set(model_path, name, iteration, views, threshold, folder, tolerance, device, consensus=False, border_margin=0):
    if not views:
        return
    root = Path(model_path) / name / f"ours_{iteration}"
    poses, depths, focals = [], [], []
    for view in views:
        w2c = torch.eye(4, device=device)
        w2c[:3, :3] = torch.as_tensor(view.R.T, device=device)
        w2c[:3, 3] = torch.as_tensor(view.T, device=device)
        poses.append(w2c)
        depth = torch.load(root / "Depth_pth" / (view.image_name.split(".")[0] + ".pth"),
                           map_location="cpu")
        if depth.shape != (1, view.image_height, view.image_width):
            raise ValueError(f"Depth/image size mismatch for {view.image_name}")
        depths.append(depth)
        focals.append((fov2focal(view.FoVx, view.image_width),
                       fov2focal(view.FoVy, view.image_height)))
    c2w = [torch.linalg.inv(pose) for pose in poses]
    out = root / folder
    out.mkdir(parents=True, exist_ok=True)
    for i, view in enumerate(tqdm(views, desc="Visible priors")):
        target = view.original_image[:3].to(device)
        depth = depths[i].to(device)
        rays = camera_rays(*target.shape[-2:], *focals[i], device)
        points = rays * torch.where(torch.isfinite(depth) & (depth > 0), depth, float("nan"))

        def samples(target_points):
            for j, src in enumerate(views):
                if j != i and torch.linalg.norm(c2w[i] - c2w[j]) < threshold:
                    yield visible_sample(target_points, poses[j] @ c2w[i], src.original_image[:3].to(device),
                                         depths[j].to(device), *focals[j], tolerance, border_margin)

        if consensus:
            prior = torch.empty_like(target)
            used = torch.empty_like(target[0], dtype=torch.bool)
            for y in range(0, target.shape[1], 96):
                rows = slice(y, y + 96)
                prior[:, rows], used[rows] = consensus_prior(target[:, rows], samples(points[:, rows]))
        else:
            prior, used = robust_prior(target, samples(points))
        path = out / Path(view.image_name).name
        bgr = (prior[[2, 1, 0]].permute(1, 2, 0).clamp(0, 1) * 255).round().byte()
        if not cv2.imwrite(str(path), bgr.cpu().numpy()):
            raise OSError(f"Could not write prior {path}")
        print(f"{view.image_name}: auxiliary RGB selected on {used.float().mean().item():.1%} of pixels")


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    threshold: float,
    file_name: str,
    prior_mode="legacy",
    depth_tolerance=0.05,
    source_border_margin=0,
    legacy_pixel_sampling=True,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if prior_mode in ("visible", "consensus"):
            if not skip_train:
                render_visible_set(dataset.model_path, "train", scene.loaded_iter,
                                   scene.getTrainCameras(), threshold, file_name,
                                   depth_tolerance, background.device, consensus=prior_mode == "consensus",
                                   border_margin=source_border_margin)
            return  # Stage two consumes supervision only for fitting views.

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                threshold,
                file_name,
                pixel_center_offset=0.0 if legacy_pixel_sampling else 0.5,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                threshold,
                file_name,
                pixel_center_offset=0.0 if legacy_pixel_sampling else 0.5,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--threshold", default=100, type=float)
    parser.add_argument("--file_name", default='', type=str)
    parser.add_argument("--prior_mode", choices=["legacy", "visible", "consensus"], default="legacy")
    parser.add_argument("--depth_tolerance", type=float, default=0.05)
    parser.add_argument("--source_border_margin", type=int, default=0,
                        help="Exclude this many rim pixels from source views in depth-checked warps.")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--legacy_pixel_sampling", action="store_true",
                          help="Use historical pixel sampling (the default for the legacy prior).")
    sampling.add_argument("--corrected_pixel_sampling", action="store_true",
                          help="Opt in to corrected pixel-center sampling for the legacy prior; experimental.")
    args = get_combined_args(parser)
    if (args.legacy_pixel_sampling or args.corrected_pixel_sampling) and args.prior_mode != "legacy":
        parser.error("pixel sampling options apply only to the legacy prior")
    if args.source_border_margin < 0 or (args.source_border_margin and args.prior_mode == "legacy"):
        parser.error("source_border_margin requires a nonnegative margin and a depth-checked prior")
    if args.prior_mode in ("visible", "consensus"):
        if args.skip_train or not 0 < args.depth_tolerance < 1:
            parser.error("visible priors require training views and 0 < depth_tolerance < 1")
        args.file_name = args.file_name or ("pseudo_GT_" + args.prior_mode)
        if Path(args.file_name).name != args.file_name or args.file_name in {".", "..", "pseudo_GT", "T", "R", "GT", "Depth_png", "Depth_pth"}:
            parser.error("choose a separate output directory for visible priors")
    if args.prior_mode == "legacy":
        print("Legacy prior pixel sampling: " + ("corrected (opt-in)" if args.corrected_pixel_sampling else "historical (default)"))
    print("Warping " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.threshold,
        args.file_name,
        args.prior_mode,
        args.depth_tolerance,
        args.source_border_margin,
        not args.corrected_pixel_sampling,
    )
