#!/usr/bin/env bash
# Intuitive-physics benchmark eval. For each (category, video_id, seed),
# condition on frame_02 plus revealed patches around the factual grounding
# point, predict frame_03 via ZWMPredictor.factual_prediction, and save the
# predicted PNG. Grade with grade_intuitive_physics.sh.
#
# Reference hardware: 8x A40 (48 GB), bfloat16. Shards items across --gpus
# via subprocess.Popen + CUDA_VISIBLE_DEVICES.
#
# Prereq (one-time): download the prebuilt dataset from HuggingFace into
# data/evals/intuitive_physics/.
#
#   huggingface-cli download awwkl/zwm-intuitive-physics \
#       --repo-type dataset --local-dir data/evals/intuitive_physics/
#
# Maintainer-only regeneration path (instead of the download above):
#   python personal_scripts/upload_intuitive_physics_to_hf.py
#
# Edit the values below to change the run; do not pass anything on the CLI.

set -xeuo pipefail

# Default: babyview-170m. Override with e.g. `CKPT=awwkl/zwm-bvd-170m/model.pt bash ...`.
CKPT="${CKPT:-awwkl/zwm-babyview-170m/model.pt}"
GPUS=(0 1 2 3 4 5 6 7)  # Override by editing here.
SEEDS=(0 1 2 3 4 5 6 7)  # 8 seeds per item (matches CWM publication).

FRAME_GAP="${FRAME_GAP:-10}"
SQUARE_LENGTH_IN_PATCHES="${SQUARE_LENGTH_IN_PATCHES:-4}"
ITEMS_LIMIT="${ITEMS_LIMIT:-0}"  # >0 for smoke tests (first N items).

DATA_DIR=data/evals/intuitive_physics
RECIPE="seeds${#SEEDS[@]}_gap${FRAME_GAP}"
OUT_DIR=viz/eval/intuitive_physics/${RECIPE}

python -m zwm.eval.intuitive_physics.eval_intuitive_physics_parallel \
    --gpus "${GPUS[@]}" \
    --model_name "$CKPT" \
    --dataset_dir "$DATA_DIR" \
    --output_dir "$OUT_DIR" \
    --seeds "${SEEDS[@]}" \
    --frame_gap "$FRAME_GAP" \
    --square_length_in_patches "$SQUARE_LENGTH_IN_PATCHES" \
    --items_limit "$ITEMS_LIMIT"
