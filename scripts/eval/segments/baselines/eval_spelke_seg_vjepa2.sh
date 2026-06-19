#!/usr/bin/env bash
# BASELINE (not ZWM): Feature-NN segmentation probe for V-JEPA 2 on SpelkeBench,
# run through the EXACT same probe as the ZWM feature-NN eval (and the DINOv3
# baseline). Sweeps feature LAYER x bipartition TAU; all taus from one forward per
# image. Output lands in the SAME root as the ZWM probe (distinct per-model slug),
# so the shared table grader compares them.
#
# Needs `transformers` (AutoVideoProcessor) -> run in the `ccwm` (or `dinov3`) env,
# NOT `zwm`. No CRF. Run once per resolution (FEAT_RES=512 matched-grid 32x32;
# FEAT_RES=256 matched-input 16x16). Edit the values below; do not pass on the CLI.

set -xeuo pipefail

# conda activate ccwm

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

# Variant: vjepa2_vitl | vjepa2_vitg | vjepa2_vitl_babyview. Override: VARIANT=... bash <script>.
VARIANT="${VARIANT:-vjepa2_vitl}"
FEATURE_NORM=layernorm
# V-JEPA 2 is trained at crop 256; use its NATIVE 256 (-> 16x16 grid, matched input to ZWM).
# 512 runs but is 2x its training res (out-of-distribution, unlike the resolution-flexible
# DINOv3) so it is NOT recommended. Override with FEAT_RES=512 only as an OOD curiosity.
FEAT_RES="${FEAT_RES:-256}"

TAUS=(${TAUS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9})
# Default fits ViT-L (24 blocks); ViT-g has more (use e.g. LAYERS="0 8 16 24 32 39 -1"). -1 = final.
LAYERS=(${LAYERS:-0 4 8 12 16 20 23 -1})

DATA_PATH=data/evals/segments/spelke_bench.h5
OUT_DIR=viz/eval/segments/spelke_bench_featurenn

for FEATURE_LAYER in "${LAYERS[@]}"; do
    echo "=== V-JEPA2 ${VARIANT} (res ${FEAT_RES}) feature-NN seg: layer ${FEATURE_LAYER}, taus ${TAUS[*]} ==="
    python scripts/eval/segments/baselines/eval_spelke_seg_vjepa2.py \
        --model_variant "$VARIANT" \
        --dataset_path "$DATA_PATH" \
        --output_dir "$OUT_DIR" \
        --feature_layer "$FEATURE_LAYER" \
        --feature_norm "$FEATURE_NORM" \
        --bipartition_tau "${TAUS[@]}" \
        --feat_res "$FEAT_RES" \
        --device cuda
done
