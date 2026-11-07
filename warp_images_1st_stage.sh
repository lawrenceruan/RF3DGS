CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 python warp_images.py \
-s /disk1/wenqiang/Documents/code/3DGS/vggsfm/hkust_data/HKUST_82_resize \
-m output/HKUST_82_resize \
--iteration 3000 \
--skip_test \
--threshold 100 \
--file_name pseudo_GT