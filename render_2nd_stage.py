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
from scene.scene_2nd_stage import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import cv2
import numpy as np


def render_set(model_path, name, iteration, views, gaussians_T, gaussians_R, gaussians_Beta_R, pipeline, background, train_test_exp, separate_sh):
    render_path_T = os.path.join(model_path, name, "ours_{}".format(iteration), 'T')
    render_path_R = os.path.join(model_path, name, "ours_{}".format(iteration), 'R')
    render_path_Beta_R = os.path.join(model_path, name, "ours_{}".format(iteration), 'Beta_R')
    render_path_GT = os.path.join(model_path, name, "ours_{}".format(iteration), 'GT')
    render_path_Depth_png = os.path.join(model_path, name, "ours_{}".format(iteration), 'Depth_png')
    render_path_Depth_pth = os.path.join(model_path, name, "ours_{}".format(iteration), 'Depth_pth')

    makedirs(render_path_T, exist_ok=True)
    makedirs(render_path_R, exist_ok=True)
    makedirs(render_path_Beta_R, exist_ok=True)
    makedirs(render_path_GT, exist_ok=True)
    makedirs(render_path_Depth_png, exist_ok=True)
    makedirs(render_path_Depth_pth, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg_T = render(view, gaussians_T, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        render_pkg_R = render(view, gaussians_R, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        render_pkg_Beta_R = render(view, gaussians_Beta_R, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        image_T = render_pkg_T['render']
        image_R = render_pkg_R['render']
        map_Beta_R = render_pkg_Beta_R['render'][0]
        depth_T = render_pkg_T['depth']
        gt = view.original_image[0:3, :, :]

        img_name = view.image_name.split('.')[0]

        torchvision.utils.save_image(image_T, os.path.join(render_path_T, img_name + ".png"))
        torchvision.utils.save_image(image_R, os.path.join(render_path_R, img_name + ".png"))
        torchvision.utils.save_image(map_Beta_R, os.path.join(render_path_Beta_R, img_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(render_path_GT, img_name + ".png"))

        ## Normalize the depth map
        depth_T = torch.clamp(depth_T, min=depth_T.quantile(0.01), max=depth_T.quantile(0.99))
        depth_T = 1.0 / (depth_T + 1e-8)

        depth_T = (depth_T - torch.min(depth_T)) / (torch.max(depth_T) - torch.min(depth_T))

        # Convert depth_T to a numpy array
        depth_T_np = depth_T[0].cpu().numpy()

        # Apply a colormap
        depth_colormap = cv2.applyColorMap((np.nan_to_num(depth_T_np).clip(0, 1) * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)

        # Save the colormap image
        cv2.imwrite(os.path.join(render_path_Depth_png, img_name + "_depth.png"), depth_colormap)

        torch.save(
            1 / render_pkg_T["depth"],
            os.path.join(render_path_Depth_pth, img_name + "_depth.pth"),
        )

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians_T = GaussianModel(dataset.sh_degree)
        gaussians_R = GaussianModel(dataset.sh_degree)
        gaussians_Beta_R = GaussianModel(1)
        scene = Scene(dataset, gaussians_T, gaussians_R, gaussians_Beta_R, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians_T, gaussians_R, gaussians_Beta_R, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians_T, gaussians_R, gaussians_Beta_R, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)