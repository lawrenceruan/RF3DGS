import os
import numpy as np
from natsort import natsorted

import torch
torch.manual_seed(0)

CUDA_device = 0

big_scenes = ['Shanghai_2']

for big_scene in big_scenes:
    scene_dir = f'/disk1/wenqiang/Documents/code/3DGS/vggsfm/{big_scene}'
    scene_names = natsorted(os.listdir(scene_dir))

    setting_name = f'{big_scene}/Stage'

    for scene_name in scene_names:
        data_path = os.path.join(scene_dir, scene_name)

        # First Stage
        stage_1_iters = 5000
        
        ## Train
        # os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 train_1st_stage.py \
        #             -s {data_path} \
        #             --sh_degree 3 \
        #             --densification_interval 500 \
        #             --scene_name {setting_name}_1st_stage/{scene_name} \
        #             --iterations {stage_1_iters}')

        ## Render
        os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 render_1st_stage.py \
                    -s {data_path} \
                    -m output/{setting_name}_1st_stage/{scene_name} \
                    --iteration {stage_1_iters}')

        ## Depth-based Warping
        # os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 warp_images_1st_stage.py \
        #             -s {data_path} \
        #             --model_path output/{setting_name}_1st_stage/{scene_name} \
        #             --iteration {stage_1_iters} \
        #             --skip_test \
        #             --threshold 10 \
        #             --file_name pseudo_GT')
        
        ## Depth processing for the second stage
        # os.system(f'python3  utils/make_depth_scale.py \
        #             --base_dir {data_path} \
        #             --depths_dir output/{setting_name}_1st_stage/{scene_name}/train/ours_{stage_1_iters}/Depth_png')
        
        # Second Stage
        stage_2_iters = 30000
        ## Train
        # os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 train_2nd_stage_L1Highlight.py \
        #             -s {data_path} \
        #             --pseudo_source_path output/{setting_name}_1st_stage/{scene_name}/train/ours_{stage_1_iters} \
        #             --sh_degree 3 \
        #             --densification_interval 500 \
        #             --scene_name {setting_name}_2nd_stage/{scene_name} \
        #             --iterations {stage_2_iters}')

        # ## Render
        # os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 render_2nd_stage.py \
        #             -m output/{setting_name}_2nd_stage/{scene_name} \
        #             --iteration 3000')

        # os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 render_2nd_stage.py \
        #             -m output/{setting_name}_2nd_stage/{scene_name} \
        #             --iteration 5000')

        # os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 render_2nd_stage.py \
        #             -m output/{setting_name}_2nd_stage/{scene_name} \
        #             --iteration 10000')

        # os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 render_2nd_stage.py \
        #             -m output/{setting_name}_2nd_stage/{scene_name} \
        #             --iteration 15000')

        # os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 render_2nd_stage.py \
        #             -m output/{setting_name}_2nd_stage/{scene_name} \
        #             --iteration 30000')