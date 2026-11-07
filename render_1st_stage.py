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
from scene.gaussian_model import GaussianModel
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
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def render_set(model_path, name, iteration, views, gaussians, learnable_images, pipeline, background, train_test_exp, separate_sh):
    render_path_T = os.path.join(model_path, name, "ours_{}".format(iteration), 'T')
    render_path_R = os.path.join(model_path, name, "ours_{}".format(iteration), 'R')
    render_path_GT = os.path.join(model_path, name, "ours_{}".format(iteration), 'GT')
    render_path_Depth_jpg = os.path.join(model_path, name, "ours_{}".format(iteration), 'Depth_png')
    render_path_Depth_pth = os.path.join(model_path, name, "ours_{}".format(iteration), 'Depth_pth')

    makedirs(render_path_T, exist_ok=True)
    makedirs(render_path_R, exist_ok=True)
    makedirs(render_path_GT, exist_ok=True)
    makedirs(render_path_Depth_jpg, exist_ok=True)
    makedirs(render_path_Depth_pth, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg_T = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        image_T = render_pkg_T['render']
        image_R = learnable_images(view.img_idx)
        depth = render_pkg_T['depth']
        gt = view.original_image[0:3, :, :]

        img_name = view.image_name.split('.')[0]

        torchvision.utils.save_image(image_T, os.path.join(render_path_T, view.image_name))
        torchvision.utils.save_image(image_R, os.path.join(render_path_R, view.image_name))
        torchvision.utils.save_image(gt, os.path.join(render_path_GT, view.image_name))

        ## Normalize the depth map
        depth = torch.clamp(depth, min=depth.quantile(0.01), max=depth.quantile(0.99))
        depth = 1.0 / (depth + 1e-8)

        depth = (depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth))

        # Convert depth to a numpy array
        depth_np = depth[0].cpu().numpy()

        # Apply a colormap
        depth_colormap = cm.plasma(depth_np)

        # Save the colormap image
        plt.imsave(os.path.join(render_path_Depth_jpg, img_name + ".png"), depth_colormap)

        torch.save(
            1 / render_pkg_T["depth"],
            os.path.join(render_path_Depth_pth, img_name + ".pth"),
        )

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, scene.learnable_images, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, scene.learnable_images, pipeline, background, dataset.train_test_exp, separate_sh)

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