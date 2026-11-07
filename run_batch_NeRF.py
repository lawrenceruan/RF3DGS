import os
import numpy as np
from natsort import natsorted

import torch
torch.manual_seed(0)

CUDA_device = 0

scene_dir = '/disk1/wenqiang/Documents/code/3DGS/vggsfm/NeRF_data'
scene_names = natsorted(os.listdir(scene_dir))

setting_name = 'NeRF/Stage'

for scene_name in scene_names:
    data_path = os.path.join(scene_dir, scene_name)

    # First Stage
    ## Train
    os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 train_1st_stage.py \
                -s {data_path} \
                --sh_degree 3 \
                --densification_interval 500 \
                --scene_name {setting_name}_1st_stage/{scene_name} \
                --iterations 5000')

    ## Render
    os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 render_1st_stage.py \
                -s {data_path} \
                -m output/{setting_name}_1st_stage/{scene_name} \
                --iteration 5000')

    ## Depth-based Warping
    os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 warp_images_1st_stage.py \
                -s {data_path} \
                --model_path output/{setting_name}_1st_stage/{scene_name} \
                --iteration 5000 \
                --skip_test \
                --threshold 10 \
                --file_name pseudo_GT')
    
    ## Depth processing for the second stage
    os.system(f'python3  utils/make_depth_scale.py \
                --base_dir {data_path} \
                --depths_dir output/{setting_name}_1st_stage/{scene_name}/train/ours_5000/Depth_png')

    # Second Stage
    ## Train
    os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 train_2nd_stage_nerf.py \
                -s {data_path} \
                --pseudo_source_path output/{setting_name}_1st_stage/{scene_name}/train/ours_5000 \
                --sh_degree 3 \
                --densification_interval 500 \
                --scene_name {setting_name}_2nd_stage/{scene_name} \
                --iterations 30000')

    ## Render
    os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 render_2nd_stage.py \
                -m output/{setting_name}_2nd_stage/{scene_name} \
                --iteration 1000')

    os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 render_2nd_stage.py \
                -m output/{setting_name}_2nd_stage/{scene_name} \
                --iteration 3000')

    os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 render_2nd_stage.py \
                -m output/{setting_name}_2nd_stage/{scene_name} \
                --iteration 5000')

    os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 render_2nd_stage.py \
                -m output/{setting_name}_2nd_stage/{scene_name} \
                --iteration 10000')

    os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 render_2nd_stage.py \
                -m output/{setting_name}_2nd_stage/{scene_name} \
                --iteration 15000')

    os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 render_2nd_stage.py \
                -m output/{setting_name}_2nd_stage/{scene_name} \
                --iteration 30000')
