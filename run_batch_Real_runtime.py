import os
import numpy as np
from natsort import natsorted

import torch
torch.manual_seed(0)

import time

CUDA_device = 0

big_scenes = ['filtered_scenes']

for big_scene in big_scenes:
    scene_dir = f'/disk1/wenqiang/Documents/code/3DGS/vggsfm/{big_scene}'
    scene_names = ['Japan14_resize']

    setting_name = f'{big_scene}/Stage'

    for scene_name in scene_names:
        data_path = os.path.join(scene_dir, scene_name)

        # First Stage
        stage_1_iters = 5000
        
        # start_time = time.time()

        '''
        ## Train
        os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 train_1st_stage.py \
                    -s {data_path} \
                    --sh_degree 3 \
                    --densification_interval 500 \
                    --scene_name {setting_name}_1st_stage/{scene_name} \
                    --iterations {stage_1_iters}')
        ## Render
        os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 render_1st_stage.py \
                    -s {data_path} \
                    -m output/{setting_name}_1st_stage/{scene_name} \
                    --iteration {stage_1_iters}')
        ## Depth-based Warping
        os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 warp_images_1st_stage.py \
                    -s {data_path} \
                    --model_path output/{setting_name}_1st_stage/{scene_name} \
                    --iteration {stage_1_iters} \
                    --skip_test \
                    --threshold 10 \
                    --file_name pseudo_GT')
        
        # Second Stage
        stage_2_iters = 30000
        ## Train
        os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 train_2nd_stage_L1Highlight.py \
                    -s {data_path} \
                    --pseudo_source_path output/{setting_name}_1st_stage/{scene_name}/train/ours_{stage_1_iters} \
                    --sh_degree 3 \
                    --densification_interval 500 \
                    --scene_name {setting_name}_2nd_stage/{scene_name} \
                    --iterations {stage_2_iters}')
        '''

        ## Render
        os.system(f'CUDA_VISIBLE_DEVICES={CUDA_device} OMP_NUM_THREADS=8 python3 render_2nd_stage_runtime.py \
                    -m output/{setting_name}_2nd_stage/{scene_name} \
                    --iteration 30000')

        # end_time = time.time()

        # # Calculate the elapsed time in seconds
        # elapsed_time_seconds = end_time - start_time

        # # Convert the elapsed time to minutes
        # elapsed_time_minutes = elapsed_time_seconds / 60

        # # Print the result
        # print(f"Program runtime: {elapsed_time_seconds:.2f} seconds")
        # print(f"Program runtime: {elapsed_time_minutes:.2f} minutes")