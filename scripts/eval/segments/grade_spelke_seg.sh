#!/usr/bin/env bash
# Grade the SpelkeBench segmentation eval produced by eval_spelke_seg.sh.
# Globs the per-image .h5 files under the model's output dir and prints
# mean AP / AR / IoU across the dataset.
#
# Edit the values below; do not pass anything on the CLI.

set -euo pipefail

# Default: babyview-170m. Override with e.g.:
#   CKPT=awwkl/zwm-bvd-170m/model.pt bash scripts/eval/segments/grade_spelke_seg.sh
CKPT="${CKPT:-awwkl/zwm-babyview-170m/model.pt}"
MODEL_SLUG="$(echo "$CKPT" | tr '/' '_' | sed 's/\.pt$//')"

# Must match the RECIPE used by eval_spelke_seg.sh.
RECIPE=seq16_seeds3_dirs8_zoom0
INPUT_DIR=viz/eval/segments/spelke_bench/${RECIPE}/${MODEL_SLUG}

# Visualize a random subset of predictions (0 disables). Override with
# `NUM_VIZ=0 bash ...` for a metrics-only run.
NUM_VIZ="${NUM_VIZ:-10}"

DATA_PATH=data/evals/segments/spelke_bench.h5

python -m zwm.eval.segments.grade_spelke_seg \
    --input_dir "$INPUT_DIR" \
    --dataset_path "$DATA_PATH" \
    --num_viz "$NUM_VIZ"
