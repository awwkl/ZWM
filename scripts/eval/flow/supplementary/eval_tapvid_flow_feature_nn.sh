#!/usr/bin/env bash
# Supplementary: Feature-NN (cosine patch-feature matching) optical-flow
# eval for BabyZWM on the FULL TapVID-DAVIS benchmark (38881 query/target
# pairs). This reads flow from BabyZWM's OWN frozen patch features with the
# exact nearest-neighbor probe used for the DINOv3 / V-JEPA2 baselines — a
# matched, label-free comparison that isolates representation from readout.
# This is NOT BabyZWM's native perturb-compare mechanism (see
# scripts/eval/flow/eval_tapvid_flow.sh for that).
#
# Sweeps a set of feature layers IN SERIES (one full eval per layer). Each
# layer's results land in a separate dir (..._featurenn_layer{L}), so runs
# never collide and each is self-describing. The middle layer (12 of the
# 24-layer 170M model) is the headline and is run first; the rest step every 4
# layers across the stack, plus -1 = final layer after the final LayerNorm.
#
# Grade each layer with scripts/eval/flow/grade_tapvid_flow.sh, pointing its
# ROOT_DIR at OUT_DIR/<slug>_featurenn_layer$L.
#
# Edit the values below to change the run; do not pass anything on the CLI.

set -xeuo pipefail

# conda activate zwm

# Run from the repo root regardless of where this is launched from: the paths
# below (DATA_PATH, OUT_DIR, the python module) are relative to it. Derived from
# this script's own location (scripts/eval/flow/supplementary -> 4 levels up).
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1

# Pin to one GPU (one GPU per node). The process sees only this device as
# cuda:0, so --device cuda lands on it. Override per node: GPU=3 bash <script>.
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

# Path relative to out/. Override per run: CKPT=<path> bash <script>.
CKPT="${CKPT:-awwkl/zwm-bvd-170m/model.pt}"
ENCODING=in_context          # in_context = both frames in one forward (ZWM's native two-frame setting); or independent
# Middle layer first (headline), then every 4 layers + final post-LayerNorm.
# 170M (24-layer) default; for the 1B (48-layer): LAYERS="24 0 8 16 32 40 -1" bash <script>.
LAYERS=(${LAYERS:-12 0 4 8 16 20 -1})
START_IDX=0
NUM_POINTS=38881

# Resolution-boosting zoom — matched to the baselines and the native eval.
ZOOM_ITERS=4

DATA_PATH=data/evals/flow/tapvid_davis_first/dataset.json
OUT_DIR=viz/eval/flow/tapvid_davis_first_featurenn/zoom_${ZOOM_ITERS}

for FEATURE_LAYER in "${LAYERS[@]}"; do
    echo "=== Feature-NN flow eval: layer ${FEATURE_LAYER} ==="
    python scripts/eval/flow/supplementary/eval_tapvid_flow_feature_nn.py \
        --model_name "$CKPT" \
        --data_path "$DATA_PATH" \
        --out_dir "$OUT_DIR" \
        --encoding "$ENCODING" \
        --feature_layer "$FEATURE_LAYER" \
        --zoom_iters "$ZOOM_ITERS" \
        --squish \
        --flat_points_start_idx "$START_IDX" \
        --num_flat_points_to_process "$NUM_POINTS" \
        --log_interval 10 \
        --viz_interval 1000 \
        --device cuda
done
