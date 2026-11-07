CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python3 run.py \
--encoder vitl \
--pred-only \
--grayscale \
--img-path /disk1/wenqiang/Documents/code/3DGS/vggsfm/hkust_data/HKUST_89_resize/images \
--outdir /disk1/wenqiang/Documents/code/3DGS/vggsfm/hkust_data/HKUST_89_resize/depths
