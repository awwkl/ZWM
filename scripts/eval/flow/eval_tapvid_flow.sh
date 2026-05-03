#!/usr/bin/env bash
# Full TapVID-DAVIS benchmark eval: 38881 query/target point pairs across all
# 30 DAVIS videos. End-Point Error (vs ground-truth tracks) via counterfactual
# point-tracking (optical flow) — perturb the query patch, predict frame1
# with/without the perturbation, locate the perturbation in the RGB-diff
# heatmap.
#
# Reference hardware: 1x NVIDIA A40 (48 GB), bfloat16. Expect ~18 hours for
# the 170M model end-to-end on one GPU. To shard across multiple GPUs, copy
# this file and edit START_IDX / NUM_POINTS per shard.
#
# For a quick (~30 min) smoketest, use eval_tapvid_flow_demo.sh instead.
#
# Prereq (one-time): extract frames from the TapVID-DAVIS pickle
#   python scripts/eval/flow/extract_tapvid_davis_pkl.py \
#       --pkl_path /path/to/tapvid_davis.pkl \
#       --out_dir data/evals/flow/tapvid_davis_first/frames/
#
# Edit the values below to change the run; do not pass anything on the CLI.

set -xeuo pipefail

CKPT=awwkl/zwm-babyview-170m/model.pt
START_IDX=0
NUM_POINTS=38881

# Recipe knobs — fixed; this is the published configuration.
NUM_ROLLOUTS=5
ZOOM_ITERS=4
STD=2
FRAME_GAP=-1
MASK_RATIO=0.9

DATA_PATH=data/evals/flow/tapvid_davis_first/dataset.json
OUT_DIR=viz/eval/flow/tapvid_davis_first/std_${STD}_zoom_${ZOOM_ITERS}

python -m zwm.eval.flow.eval_tapvid_flow \
    --model_name "$CKPT" \
    --data_path "$DATA_PATH" \
    --out_dir "$OUT_DIR" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --perturb_std "$STD" \
    --zoom_iters "$ZOOM_ITERS" \
    --mask_ratio "$MASK_RATIO" \
    --frame_gap "$FRAME_GAP" \
    --no_blur \
    --squish \
    --flat_points_start_idx "$START_IDX" \
    --num_flat_points_to_process "$NUM_POINTS" \
    --log_interval 10 \
    --viz_interval 1000 \
    --device cuda \
    --compile
