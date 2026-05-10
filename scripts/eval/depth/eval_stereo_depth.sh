#!/usr/bin/env bash
# Stereo-depth counterfactual point-tracking eval: 328 (query, target) point
# entries spanning 164 stereo image pairs from KITTI 1000 + KITTI 500-flipud,
# organized as (point0, point1) per image. Reuses the flow EPE engine — for
# each point, perturb the query frame around the query point, predict the
# stereo-mate frame with/without the perturbation, and locate the perturbation
# via the L2 norm of the RGB-prediction difference. EPE against the ground-
# truth correspondence is the per-point metric. Comparing point0_epe vs
# point1_epe orders depths (closer point -> larger EPE).
#
# Reference hardware: 1x NVIDIA A40 (48 GB), bfloat16. ~9 minutes for 328
# points (extrapolated from the flow eval at zoom_iters=4).
#
# Note: stereo uids have et - st = 0; with FRAME_GAP=-1 the engine's dynamic
# policy clamps to 5 (it prints `Data Frame Gap: 0 => 5 (policy=...)` once).
#
# Prereq (one-time): download the prebuilt dataset from HuggingFace into
# data/evals/depth/. ~135 MB.
#
#   huggingface-cli download awwkl/zwm-stereo-depth-eval \
#       --repo-type dataset --local-dir data/evals/depth/
#
# Maintainer-only regeneration path (instead of the download above):
#   python personal_scripts/prepare_depth_dataset.py --copy_images
#   python personal_scripts/upload_depth_dataset_to_hf.py
#
# Edit the values below to change the run; do not pass anything on the CLI.

set -xeuo pipefail

# Default: babyview-170m. Override with e.g. `CKPT=awwkl/zwm-bvd-170m/model.pt bash ...`.
CKPT="${CKPT:-awwkl/zwm-babyview-170m/model.pt}"
START_IDX=0
NUM_POINTS=328

# Recipe knobs — fixed; matches the published flow eval configuration.
NUM_ROLLOUTS=5
ZOOM_ITERS=4
STD=2
FRAME_GAP=-1
MASK_RATIO=0.9

DATA_PATH=data/evals/depth/stereo_depth/dataset.json
OUT_DIR=viz/eval/depth/stereo_depth/std_${STD}_zoom_${ZOOM_ITERS}

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
