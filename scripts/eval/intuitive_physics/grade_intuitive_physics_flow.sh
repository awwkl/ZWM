#!/usr/bin/env bash
# Grade the intuitive-physics eval with the FLOW-based motion-fidelity metric
# (per-object EPE / normalized %-error / graded closeness).
# Companion to grade_intuitive_physics.sh (pixel MSE/LPIPS).
#
# Edit the values below; do not pass anything on the CLI.

set -euo pipefail

# Default: babyview-170m. Override with e.g.:
#   CKPT=awwkl/zwm-bvd-170m/model.pt bash scripts/eval/intuitive_physics/grade_intuitive_physics_flow.sh
CKPT="${CKPT:-awwkl/zwm-babyview-170m/model.pt}"
MODEL_SLUG="$(echo "$CKPT" | tr '/' '_' | sed 's/\.pt$//')"

# Must match the RECIPE used by eval_intuitive_physics.sh.
RECIPE="${RECIPE:-seeds8_gap10}"
PRED_DIR=viz/eval/intuitive_physics/${RECIPE}/${MODEL_SLUG}/pred

DATA_DIR=data/evals/intuitive_physics
OUT_CSV="${OUT_CSV:-viz/eval/intuitive_physics/${RECIPE}/${MODEL_SLUG}/flow_accuracy.csv}"
ROWS_CSV="${ROWS_CSV:-viz/eval/intuitive_physics/${RECIPE}/${MODEL_SLUG}/flow_rows.csv}"

python -m zwm.eval.intuitive_physics.grade_intuitive_physics_flow \
    --pred_dir "$PRED_DIR" \
    --dataset_dir "$DATA_DIR" \
    --out_csv "$OUT_CSV" \
    --rows_csv "$ROWS_CSV"
