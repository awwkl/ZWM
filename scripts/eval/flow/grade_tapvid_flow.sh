#!/usr/bin/env bash
# Grade the FULL TapVID-DAVIS eval (38881 points) produced by
# eval_tapvid_flow.sh. Aggregates per-shard prediction JSONs, joins against
# the TapVID-DAVIS ground-truth pickle, and prints the official metrics
# (AD / MD / Pct / AJ / OA / OF1).
#
# This script does NOT pass --sample — it grades over the full DAVIS
# trajectory space. Only meaningful when ROOT_DIR holds predictions for the
# full 38881-point set; for a demo/partial run, use grade_tapvid_flow_demo.sh.
#
# Edit the values below; do not pass anything on the CLI.

set -euo pipefail

# Path to the downloaded TapVID-DAVIS pickle — same one used to extract frames.
# Place the downloaded pickle at this path before running.
PKL_PATH=data/evals/flow/tapvid_davis_first/tapvid_davis.pkl

# Output dir of the full eval run to grade. Pick a model by uncommenting the
# corresponding ROOT_DIR line. The active line wins; comment the other one out.
# Each path matches the OUT_DIR scheme in eval_tapvid_flow.sh after the eval
# has been run with that CKPT.

# --- babyview-170m (default) ---
ROOT_DIR=viz/eval/flow/tapvid_davis_first/std_2_zoom_4/awwkl_zwm-babyview-170m_model_mask_ratio_0.9

# --- bvd-170m ---
# ROOT_DIR=viz/eval/flow/tapvid_davis_first/std_2_zoom_4/awwkl_zwm-bvd-170m_model_mask_ratio_0.9

# Threshold on `occ_metric` for predicting occlusion.
OCC_THRESH=0.4

python -m zwm.eval.flow.grade_tapvid_flow \
    --root_dir "$ROOT_DIR" \
    --pkl_path "$PKL_PATH" \
    --occ_thresh "$OCC_THRESH"
