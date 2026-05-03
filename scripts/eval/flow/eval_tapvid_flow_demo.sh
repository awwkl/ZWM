#!/usr/bin/env bash
# DEMO / smoketest of the TapVID-DAVIS counterfactual point-tracking (optical
# flow) eval.
# Runs the recipe on the bundled 30-entry demo subset (one query/target pair
# per DAVIS video, frame gaps spanning [0, 15]) and visualizes every point.
#
# Reference hardware: 1x NVIDIA A40 (48 GB), bfloat16. Expect ~30 min total —
# ~60 s/point, the first few dominated by torch.compile warmup.
#
# This is a smoketest of the eval pipeline, not a usable benchmark number.
# For the full 38881-point benchmark, use eval_tapvid_flow.sh.
#
# Prereq (one-time): extract frames from the TapVID-DAVIS pickle
#   python scripts/eval/flow/extract_tapvid_davis_pkl.py \
#       --pkl_path /path/to/tapvid_davis.pkl \
#       --out_dir data/evals/flow/tapvid_davis_first/frames/
#
# Edit the values below to change the run; do not pass anything on the CLI.

set -xeuo pipefail

CKPT=awwkl/zwm-babyview-170m/model.pt

# Recipe knobs — same as the full eval, so demo behavior matches.
NUM_ROLLOUTS=5
ZOOM_ITERS=4
STD=2
FRAME_GAP=-1
MASK_RATIO=0.9

DATA_PATH=data/evals/flow/tapvid_davis_first/dataset_demo.json
OUT_DIR=viz/eval/flow/tapvid_davis_first_demo/std_${STD}_zoom_${ZOOM_ITERS}

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
    --flat_points_start_idx 0 \
    --num_flat_points_to_process 30 \
    --log_interval 5 \
    --viz_all \
    --device cuda \
    --compile
