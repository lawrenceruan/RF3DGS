import os
import cv2
import pyiqa
import torch
import lpips
import numpy as np

metric_psnr = pyiqa.create_metric('psnr', device=torch.device('cuda'))
metric_ssim = pyiqa.create_metric('ssim', device=torch.device('cuda'))
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

def img_resize(img):
    orig_w, orig_h = img.size
    global WIDTH
    scale = orig_w / WIDTH
    resolution = (int(orig_w / scale), int(orig_h / scale))

    return img.resize(resolution)

def get_psnr(im1, im2):
    im1_th = torch.from_numpy((im1.astype(np.float32) / 255.).transpose(2,0,1)).unsqueeze(0)
    im2_th = torch.from_numpy((im2.astype(np.float32) / 255.).transpose(2,0,1)).unsqueeze(0)

    return metric_psnr(im1_th, im2_th)


def get_ssim(im1, im2):
    im1_th = torch.from_numpy((im1.astype(np.float32) / 255.).transpose(2,0,1)).unsqueeze(0)
    im2_th = torch.from_numpy((im2.astype(np.float32) / 255.).transpose(2,0,1)).unsqueeze(0)

    return metric_ssim(im1_th, im2_th)


def get_lpips(im1, im2):
    im1_th = torch.from_numpy((im1.astype(np.float32) / 255.).transpose(2,0,1)).unsqueeze(0)
    im2_th = torch.from_numpy((im2.astype(np.float32) / 255.).transpose(2,0,1)).unsqueeze(0)

    im1_th = im1_th * 2 - 1
    im2_th = im2_th * 2 - 1

    return loss_fn_alex.forward(im1_th, im2_th)

scene_names = ['bag_reflection_scene_2',
              'Car_reflection_scene_2',
              'trunk_reflection_scene_2']

for scene_name in scene_names:

    case_path = f'/disk1/wenqiang/Documents/code/3DGS/ours/012_AllStage/output/Unity/Dataset/{scene_name}/Stage_2nd_stage/' 

    case_name_list = os.listdir(case_path)

    case_count = 0

    input_psnr_avg_case = 0
    input_ssim_avg_case = 0
    input_lpips_avg_case = 0

    ours_psnr_avg_case = 0
    ours_ssim_avg_case = 0
    ours_lpips_avg_case = 0

    official_3DGS_psnr_avg_case = 0
    official_3DGS_ssim_avg_case = 0
    official_3DGS_lpips_avg_case = 0

    for case_name in case_name_list:
        input_path = os.path.join(case_path, case_name, 'train/ours_10000/GT')
        ours_path = os.path.join(case_path, case_name, 'train/ours_10000/T')
        official_3DGS_path = os.path.join(f'/disk1/wenqiang/Documents/code/3DGS/ours/gaussian-splatting/output/Unity/Dataset/{scene_name}/Stage', case_name, 'train/ours_10000/renders')
        gt_path = f'/disk1/wenqiang/Documents/code/3DGS/vggsfm/Unity/Dataset/GT/{scene_name}/images'

        img_count = 0

        input_psnr_avg_img = 0
        input_ssim_avg_img = 0
        input_lpips_avg_img = 0

        ours_psnr_avg_img = 0
        ours_ssim_avg_img = 0
        ours_lpips_avg_img = 0

        official_3DGS_psnr_avg_img = 0
        official_3DGS_ssim_avg_img = 0
        official_3DGS_lpips_avg_img = 0

        for frame_name in os.listdir(input_path):
            input_img = cv2.cvtColor(cv2.imread(os.path.join(input_path, frame_name)), cv2.COLOR_BGR2RGB)
            ours_img = cv2.cvtColor(cv2.imread(os.path.join(ours_path, frame_name)), cv2.COLOR_BGR2RGB)
            official_3DGS_img = cv2.cvtColor(cv2.imread(os.path.join(official_3DGS_path, frame_name)), cv2.COLOR_BGR2RGB)
            gt_img = cv2.cvtColor(cv2.imread(os.path.join(gt_path, frame_name.replace('png', 'jpg'))), cv2.COLOR_BGR2RGB)

            input_psnr_avg_img += get_psnr(input_img, gt_img)
            input_ssim_avg_img += get_ssim(input_img, gt_img)
            input_lpips_avg_img += get_lpips(input_img, gt_img)

            ours_psnr_avg_img += get_psnr(ours_img, gt_img)
            ours_ssim_avg_img += get_ssim(ours_img, gt_img)
            ours_lpips_avg_img += get_lpips(ours_img, gt_img)

            official_3DGS_psnr_avg_img += get_psnr(official_3DGS_img, gt_img)
            official_3DGS_ssim_avg_img += get_ssim(official_3DGS_img, gt_img)
            official_3DGS_lpips_avg_img += get_lpips(official_3DGS_img, gt_img)

            img_count += 1

        input_psnr_avg_img = input_psnr_avg_img / img_count
        input_ssim_avg_img = input_ssim_avg_img / img_count
        input_lpips_avg_img = input_lpips_avg_img / img_count

        ours_psnr_avg_img = ours_psnr_avg_img / img_count
        ours_ssim_avg_img = ours_ssim_avg_img / img_count
        ours_lpips_avg_img = ours_lpips_avg_img / img_count

        official_3DGS_psnr_avg_img = official_3DGS_psnr_avg_img / img_count
        official_3DGS_ssim_avg_img = official_3DGS_ssim_avg_img / img_count
        official_3DGS_lpips_avg_img = official_3DGS_lpips_avg_img / img_count

        with open('scores.txt', 'a') as f:

            print('input_psnr_avg_img: ', input_psnr_avg_img, file=f)
            print('input_ssim_avg_img: ', input_ssim_avg_img, file=f)
            print('input_lpips_avg_img: ', input_lpips_avg_img, file=f)

            print('--------------------------', file=f)

            print('ours_psnr_avg_img: ', ours_psnr_avg_img, file=f)
            print('ours_ssim_avg_img: ', ours_ssim_avg_img, file=f)
            print('ours_lpips_avg_img: ', ours_lpips_avg_img, file=f)

            print('--------------------------', file=f)

            print('official_3DGS_psnr_avg_img: ', official_3DGS_psnr_avg_img, file=f)
            print('official_3DGS_ssim_avg_img: ', official_3DGS_ssim_avg_img, file=f)
            print('official_3DGS_lpips_avg_img: ', official_3DGS_lpips_avg_img, file=f)

            print('case_name: ', case_name, file=f)

            print('--------------------------', file=f)
            print('--------------------------', file=f)
            print('--------------------------', file=f)

        input_psnr_avg_case += input_psnr_avg_img
        input_ssim_avg_case += input_ssim_avg_img
        input_lpips_avg_case += input_lpips_avg_img

        ours_psnr_avg_case += ours_psnr_avg_img
        ours_ssim_avg_case += ours_ssim_avg_img
        ours_lpips_avg_case += ours_lpips_avg_img

        official_3DGS_psnr_avg_case += official_3DGS_psnr_avg_img
        official_3DGS_ssim_avg_case += official_3DGS_ssim_avg_img
        official_3DGS_lpips_avg_case += official_3DGS_lpips_avg_img

        case_count += 1

    input_psnr_avg_case = input_psnr_avg_case / case_count
    input_ssim_avg_case = input_ssim_avg_case / case_count
    input_lpips_avg_case = input_lpips_avg_case / case_count

    ours_psnr_avg_case = ours_psnr_avg_case / case_count
    ours_ssim_avg_case = ours_ssim_avg_case / case_count
    ours_lpips_avg_case = ours_lpips_avg_case / case_count

    official_3DGS_psnr_avg_case = official_3DGS_psnr_avg_case / case_count
    official_3DGS_ssim_avg_case = official_3DGS_ssim_avg_case / case_count
    official_3DGS_lpips_avg_case = official_3DGS_lpips_avg_case / case_count

    with open('scores.txt', 'a') as f:

        print('input_psnr_avg_case: ', input_psnr_avg_case, file=f)
        print('input_ssim_avg_case: ', input_ssim_avg_case, file=f)
        print('input_lpips_avg_case: ', input_lpips_avg_case, file=f)

        print('--------------------------', file=f)

        print('ours_psnr_avg_case: ', ours_psnr_avg_case, file=f)
        print('ours_ssim_avg_case: ', ours_ssim_avg_case, file=f)
        print('ours_lpips_avg_case: ', ours_lpips_avg_case, file=f)

        print('--------------------------', file=f)

        print('official_3DGS_psnr_avg_case: ', official_3DGS_psnr_avg_case, file=f)
        print('official_3DGS_ssim_avg_case: ', official_3DGS_ssim_avg_case, file=f)
        print('official_3DGS_lpips_avg_case: ', official_3DGS_lpips_avg_case, file=f)

        print('scene_name: ', scene_name, file=f)

        print('##########################', file=f)
        print('##########################', file=f)
        print('##########################', file=f)