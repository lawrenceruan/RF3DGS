#!/usr/bin/env bash
set -euo pipefail
# Usage: bash run_scene.sh /absolute/path/to/prepared_scene unique_run_name
DATA=${1:?Supply a prepared scene directory}
RUN=${2:?Supply a unique run name}
if [[ ! -d "$DATA/images" || ! -d "$DATA/sparse" ]]; then
  echo "Expected images/ and sparse/ under DATA" >&2; exit 1
fi
if [[ -e "output/$RUN" ]]; then
  echo "Output already exists; choose a new run name." >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Stage 1: initial geometry, 5,000 iterations
python train_1st_stage.py -s "$DATA" \
  --scene_name "${RUN}/stage1" --iterations 5000 \
  --sh_degree 3 --densification_interval 500 --disable_viewer

# Render the initial transmission and depth
python render_1st_stage.py -s "$DATA" \
  -m "output/${RUN}/stage1" --iteration 5000 --skip_test

# Construct transmission pseudo-labels
python warp_images_1st_stage.py -s "$DATA" \
  -m "output/${RUN}/stage1" --iteration 5000 \
  --skip_test --threshold 10 --file_name pseudo_GT

# Stage 2: ATR optimization, 30,000 iterations
python train_2nd_stage_L1Highlight.py -s "$DATA" \
  --pseudo_source_path "output/${RUN}/stage1/train/ours_5000" \
  --scene_name "${RUN}/stage2" --iterations 30000 \
  --sh_degree 3 --densification_interval 500 --disable_viewer

# Render the transmission, reflection, and blending factor
python render_2nd_stage.py -s "$DATA" \
  -m "output/${RUN}/stage2" --iteration 30000 --skip_test
