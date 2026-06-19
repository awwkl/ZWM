#!/usr/bin/env bash
# Supplementary: Feature-NN (cosine patch-feature matching) relative-depth
# eval for BabyZWM on the stereo-depth benchmark (164 pairs, 328 points;
# KITTI 1000 + KITTI 500-flipud). Relative depth is read as binocular
# correspondence (disparity) between the two simultaneous stereo views — the
# SAME cosine nearest-neighbor probe as the flow eval, just pointed at the
# stereo dataset (exactly how the depth eval reuses the flow engine). This
# reads from BabyZWM's OWN frozen patch features, matched and label-free
# against the DINOv3 / V-JEPA2 baselines, isolating representation from readout.
#
# Sweeps feature layers IN SERIES, one eval per layer (middle layer 12 first,
# then every 4 layers + final post-LayerNorm). Each layer's results land in a
# separate dir (..._featurenn_layer{L}).
#
# Grade each layer with scripts/eval/depth/grade_stereo_depth.sh after pointing
# its ROOT_DIR at OUT_DIR/<slug>_featurenn_layer$L.
#
# Prereq (one-time): the stereo dataset must be present at DATA_PATH below
# (see scripts/eval/depth/eval_stereo_depth.sh for the download command).
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

CKPT="${CKPT:-awwkl/zwm-bvd-170m/model.pt}"
ENCODING=in_context          # in_context = both frames in one forward (ZWM's native two-frame setting); or independent
LAYERS=(${LAYERS:-12 0 4 8 16 20 -1})   # 170M default; 1B: LAYERS="24 0 8 16 32 40 -1" bash <script>
START_IDX=0
NUM_POINTS=328

ZOOM_ITERS=4

DATA_PATH=data/evals/depth/stereo_depth/dataset.json
OUT_DIR=viz/eval/depth/stereo_depth_featurenn/zoom_${ZOOM_ITERS}

for FEATURE_LAYER in "${LAYERS[@]}"; do
    echo "=== Feature-NN stereo-depth eval: layer ${FEATURE_LAYER} ==="
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
