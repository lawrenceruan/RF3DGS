CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 python3 train.py \
-s /disk1/wenqiang/Documents/code/3DGS/vggsfm/hkust_data/HKUST_89_resize \
--scene_name HKUST_89_resize \
--pseudo_source_path ../007_1stStage/output/HKUST_89_resize/train/ours_3000/pseudo_GT \
-d /disk1/wenqiang/Documents/code/3DGS/ours/007_1stStage/output/HKUST_89_resize/train/ours_3000/Depth_png
