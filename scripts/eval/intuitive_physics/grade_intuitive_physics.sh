#!/usr/bin/env bash
# Grade the intuitive-physics eval produced by eval_intuitive_physics.sh.
# Prints a 5-row per-category accuracy table (MSE and LPIPS).
#
# Edit the values below; do not pass anything on the CLI.

set -euo pipefail

# Default: babyview-170m. Override with e.g.:
#   CKPT=awwkl/zwm-bvd-170m/model.pt bash scripts/eval/intuitive_physics/grade_intuitive_physics.sh
CKPT="${CKPT:-awwkl/zwm-babyview-170m/model.pt}"
MODEL_SLUG="$(echo "$CKPT" | tr '/' '_' | sed 's/\.pt$//')"

# Must match the RECIPE used by eval_intuitive_physics.sh.
RECIPE="${RECIPE:-seeds8_gap10}"
PRED_DIR=viz/eval/intuitive_physics/${RECIPE}/${MODEL_SLUG}/pred

DATA_DIR=data/evals/intuitive_physics
OUT_CSV="${OUT_CSV:-viz/eval/intuitive_physics/${RECIPE}/${MODEL_SLUG}/accuracy.csv}"

python -m zwm.eval.intuitive_physics.grade_intuitive_physics \
    --pred_dir "$PRED_DIR" \
    --dataset_dir "$DATA_DIR" \
    --out_csv "$OUT_CSV"
