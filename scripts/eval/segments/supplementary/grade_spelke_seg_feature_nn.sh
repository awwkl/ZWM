#!/usr/bin/env bash
# Grade ONE Feature-NN cell (a single layer x tau) produced by
# eval_spelke_seg_feature_nn.sh, and optionally write overlay PNGs. Cell files
# store only segment_pred; GT + RGB image are read from the source dataset by
# grade_spelke_seg_feature_nn.py. For the whole layer x tau table at once, use
# grade_spelke_seg_feature_nn_table.py.
#
# Set LAYER / TAU / FEATURE_NORM to the cell you want. Edit the values below; do
# not pass anything on the CLI.

set -euo pipefail

# Run from repo root and make the by-path scripts importable, regardless of launch dir.
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Default: bvd-1b. Override with e.g. `CKPT=awwkl/zwm-bvd-170m/model.pt bash ...`.
CKPT="${CKPT:-awwkl/zwm-bvd-1b/model.pt}"
MODEL_SLUG="$(echo "$CKPT" | tr '/' '_' | sed 's/\.pt$//')"

# Must match a (layer, tau) cell produced by eval_spelke_seg_feature_nn.sh. For the
# whole layer x tau table at once, use grade_spelke_seg_feature_nn_table.py instead.
LAYER="${LAYER:-24}"
FEATURE_NORM="${FEATURE_NORM:-layernorm}"
TAU="${TAU:-0.2}"
INPUT_DIR=viz/eval/segments/spelke_bench_featurenn/${MODEL_SLUG}_featurenn_layer${LAYER}_${FEATURE_NORM}_tau${TAU}

# Visualize a random subset of predictions (0 disables). Override with
# `NUM_VIZ=0 bash ...` for a metrics-only run.
NUM_VIZ="${NUM_VIZ:-10}"

DATA_PATH=data/evals/segments/spelke_bench.h5

python scripts/eval/segments/supplementary/grade_spelke_seg_feature_nn.py \
    --input_dir "$INPUT_DIR" \
    --dataset_path "$DATA_PATH" \
    --num_viz "$NUM_VIZ"
