#!/usr/bin/env bash
# Grade the stereo-depth eval (164 pairs, 328 points) produced by
# eval_stereo_depth.sh. Pairs point0/point1 EPEs from the per-shard
# epe_results_*.json, joins to data/evals/depth/stereo_depth/dataset.json
# for ground-truth depths, and prints accuracy + bootstrap 95% CI + binomial
# SEM stratified by dataset_name (kitti_1000 vs kitti_500_flipud).
#
# Edit the values below; do not pass anything on the CLI.

set -euo pipefail

# Default: babyview-170m. Override with e.g.:
#   CKPT=awwkl/zwm-bvd-170m/model.pt bash scripts/eval/depth/grade_stereo_depth.sh
CKPT="${CKPT:-awwkl/zwm-babyview-170m/model.pt}"
MODEL_SLUG="$(echo "$CKPT" | tr '/' '_' | sed 's/\.pt$//')"
ROOT_DIR="viz/eval/depth/stereo_depth/std_2_zoom_4/${MODEL_SLUG}_mask_ratio_0.9"

DATA_PATH=data/evals/depth/stereo_depth/dataset.json

python -m zwm.eval.depth.grade_stereo_depth \
    --root_dir "$ROOT_DIR" \
    --data_path "$DATA_PATH"
