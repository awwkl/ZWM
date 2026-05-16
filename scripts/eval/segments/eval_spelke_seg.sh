#!/usr/bin/env bash
# SpelkeBench object-segmentation eval. For each GT segment, perturb the
# centroid patch, predict next-frame RGB, compute optical flow (SEA-RAFT)
# between the perturbed prediction and the unperturbed frame, and threshold
# the dot-product-with-perturbation heatmap into a binary segment. Per-image
# AP/AR/IoU is computed by the grade script.
#
# Reference hardware: 4x A40 (48 GB), bfloat16. Shards keys across --gpus
# via subprocess.Popen + CUDA_VISIBLE_DEVICES.
#
# Prereq (one-time): download the prebuilt dataset from HuggingFace into
# data/evals/segments/. ~? MB.
#
#   huggingface-cli download awwkl/zwm-spelke-bench \
#       --repo-type dataset --local-dir data/evals/segments/
#
# Maintainer-only regeneration path (instead of the download above):
#   python personal_scripts/prepare_spelke_bench_dataset.py
#   python personal_scripts/upload_spelke_bench_to_hf.py
#
# Edit the values below to change the run; do not pass anything on the CLI.

set -xeuo pipefail

# Default: babyview-170m. Override with e.g. `CKPT=awwkl/zwm-bvd-170m/model.pt bash ...`.
CKPT="${CKPT:-awwkl/zwm-babyview-170m/model.pt}"
GPUS=(4 5 6 7)  # Override with e.g. `GPUS=(0 1)` for a 2-GPU run.

# Recipe knobs — fixed; matches the SpelkeNet/SpelkeBench CWM configuration.
NUM_SEQ_PATCHES=16
NUM_SEEDS=3
NUM_DIRS=8
NUM_ZOOM_ITERS=0
NUM_ZOOM_DIRS=5
MIN_MAG_ZOOM=10.0
MAX_MAG_ZOOM=25.0
MIN_MAG=25.0
MAX_MAG=35.0

DATA_PATH=data/evals/segments/spelke_bench.h5
RECIPE="seq${NUM_SEQ_PATCHES}_seeds${NUM_SEEDS}_dirs${NUM_DIRS}_zoom${NUM_ZOOM_ITERS}"
OUT_DIR=viz/eval/segments/spelke_bench/${RECIPE}

python -m zwm.eval.segments.eval_spelke_seg_parallel \
    --gpus "${GPUS[@]}" \
    --model_name "$CKPT" \
    --dataset_path "$DATA_PATH" \
    --output_dir "$OUT_DIR" \
    --num_seq_patches "$NUM_SEQ_PATCHES" \
    --num_seeds "$NUM_SEEDS" \
    --num_dirs "$NUM_DIRS" \
    --num_zoom_iters "$NUM_ZOOM_ITERS" \
    --num_zoom_dirs "$NUM_ZOOM_DIRS" \
    --min_mag_zoom "$MIN_MAG_ZOOM" \
    --max_mag_zoom "$MAX_MAG_ZOOM" \
    --min_mag "$MIN_MAG" \
    --max_mag "$MAX_MAG"
