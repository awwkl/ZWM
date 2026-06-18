#!/usr/bin/env bash
# Supplementary: Feature-NN (cosine patch-feature matching) segmentation eval
# for BabyZWM on SpelkeBench. For each GT segment, take the patch feature at the
# segment's seed centroid, score every patch by cosine similarity to it,
# threshold at TAU, and keep the connected component containing the seed. This
# reads segmentation from BabyZWM's OWN frozen patch features with the exact rule
# used for the DINOv3 / V-JEPA2 point-seg baselines — a matched, label-free
# comparison that isolates representation from readout. This is NOT BabyZWM's
# native perturb-compare mechanism (see scripts/eval/segments/eval_spelke_seg.sh).
#
# Sweeps feature LAYER x bipartition TAU. Layers run IN SERIES (one forward pass
# per layer per image); for each layer, ALL taus are emitted from that single pass
# (tau only thresholds the cosine map). Every (layer, tau) lands in its own dir
# (..._featurenn_layer{L}_{norm}_tau{T}), so runs never collide and each is
# self-describing. Default model is bvd-1b (48 layers); override with CKPT=.
#
# Build the layer x tau table with:
#   python scripts/eval/segments/supplementary/grade_spelke_seg_feature_nn_table.py
# or grade a single (layer, tau) cell with grade_spelke_seg_feature_nn.sh.
#
# Edit the values below to change the run; do not pass anything on the CLI.

set -xeuo pipefail

# conda activate zwm

# Run from the repo root regardless of where this is launched from: the paths
# below (DATA_PATH, OUT_DIR, the python script) are relative to it. Derived from
# this script's own location (scripts/eval/segments/supplementary -> 4 levels up).
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1

# Make the by-path worker importable from any env / launch dir (no editable install needed).
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Pin to one GPU. The process sees only this device as cuda:0, so --device cuda
# lands on it. Override per node: GPU=3 bash <script>.
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

CKPT="${CKPT:-awwkl/zwm-bvd-1b/model.pt}"
FEATURE_NORM=layernorm

# layer x tau sweep. Each LAYER is one forward pass per image; ALL TAUS are emitted
# from that single pass (tau only thresholds the cosine map), each into its own
# ..._layer{L}_{norm}_tau{T} dir. Tabulate with grade_spelke_seg_feature_nn_table.py.
# tau sweep (override with TAUS="..."); all emitted from one forward pass per image.
TAUS=(${TAUS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9})
# Default fits the 48-layer 1B models. Override per model (24-layer 170M needs
# in-range layers), e.g. LAYERS="0 4 8 12 16 20 23 -1" bash <script>.
LAYERS=(${LAYERS:-0 8 16 24 32 40 47 -1})

DATA_PATH=data/evals/segments/spelke_bench.h5
OUT_DIR=viz/eval/segments/spelke_bench_featurenn

for FEATURE_LAYER in "${LAYERS[@]}"; do
    echo "=== Feature-NN seg eval: layer ${FEATURE_LAYER}, taus ${TAUS[*]} ==="
    python scripts/eval/segments/supplementary/eval_spelke_seg_feature_nn.py \
        --model_name "$CKPT" \
        --dataset_path "$DATA_PATH" \
        --output_dir "$OUT_DIR" \
        --feature_layer "$FEATURE_LAYER" \
        --feature_norm "$FEATURE_NORM" \
        --bipartition_tau "${TAUS[@]}" \
        --device cuda
done
