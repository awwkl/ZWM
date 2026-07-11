#!/usr/bin/env bash
# BASELINE (not ZWM): Feature-NN segmentation probe for ResNet-50 (ImageNet-supervised)
# on SpelkeBench, through the EXACT same probe as the ZWM feature-NN eval. ResNet is
# convolutional, so the sweep is over conv STAGE (the ResNet analog of a layer) x
# bipartition TAU; all taus come from one forward per image. Output lands in the same
# root as the ZWM probe, so the shared table grader compares them.
#
# Needs `transformers` -> run in `ccwm` (or `dinov3`), NOT `zwm`. No CRF.

set -xeuo pipefail

# conda activate ccwm

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

VARIANT="${VARIANT:-resnet50}"
FEATURE_NORM=layernorm
FEAT_RES="${FEAT_RES:-256}"   # stage grids = feat_res/stride -> at 256: 64,64,32,16,8

TAUS=(${TAUS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9})
# ResNet-50 conv stages (its "layers"): 0,1 = stride-4 (64x64); 2 = stride-8 (32x32);
# 3 = stride-16 (16x16); 4 = stride-32 (8x8).
LAYERS=(${LAYERS:-0 1 2 3 4})

DATA_PATH=data/evals/segments/spelke_bench.h5
OUT_DIR=viz/eval/segments/spelke_bench_featurenn

for FEATURE_LAYER in "${LAYERS[@]}"; do
    echo "=== ResNet-50 (res ${FEAT_RES}) feature-NN seg: stage ${FEATURE_LAYER}, taus ${TAUS[*]} ==="
    python scripts/eval/segments/baselines/eval_spelke_seg_resnet50.py \
        --model_variant "$VARIANT" \
        --dataset_path "$DATA_PATH" \
        --output_dir "$OUT_DIR" \
        --feature_layer "$FEATURE_LAYER" \
        --feature_norm "$FEATURE_NORM" \
        --bipartition_tau "${TAUS[@]}" \
        --feat_res "$FEAT_RES" \
        --device cuda
done
