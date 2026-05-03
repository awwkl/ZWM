#!/usr/bin/env bash
# Grade the DEMO TapVID-DAVIS eval (30 points) produced by
# eval_tapvid_flow_demo.sh. Aggregates the demo-run prediction JSON, joins
# against the TapVID-DAVIS ground-truth pickle, and prints the official
# metrics (AD / MD / Pct / AJ / OA / OF1).
#
# Demo metrics are useful only for sanity-checking that the pipeline ran;
# they are not benchmark numbers. The 30-point demo only fills 30 of the
# 38881 (n,t,2) slots in the trajectory grid, so AD/Pct come out heavily
# diluted (un-evaluated frames trivially score 0 because pred=gt=0). For
# benchmark numbers use the full eval + grade_tapvid_flow.sh.
#
# Edit the values below; do not pass anything on the CLI.

set -euo pipefail

# Path to the downloaded TapVID-DAVIS pickle — same one used to extract frames.
# Place the downloaded pickle at this path before running.
PKL_PATH=data/evals/flow/tapvid_davis_first/tapvid_davis.pkl

# Output dir of the demo eval run (matches eval_tapvid_flow_demo.sh OUT_DIR).
ROOT_DIR=viz/eval/flow/tapvid_davis_first_demo/std_2_zoom_4/awwkl_zwm-babyview-170m_model_mask_ratio_0.9

# Threshold on `occ_metric` for predicting occlusion.
OCC_THRESH=0.4

python -m zwm.eval.flow.grade_tapvid_flow \
    --root_dir "$ROOT_DIR" \
    --pkl_path "$PKL_PATH" \
    --occ_thresh "$OCC_THRESH" \
    --sample
