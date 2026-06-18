#!/usr/bin/env bash
# BASELINE (not ZWM): Feature-NN segmentation probe for DINOv3 on SpelkeBench, run
# through the EXACT same probe as the ZWM feature-NN eval (scripts/eval/segments/
# supplementary/) so the two are directly comparable. Sweeps feature LAYER x
# bipartition TAU; all taus come from one forward per image. Output lands in the
# SAME root as the ZWM probe (distinct per-model slug dirs), so the shared table
# grader compares them.
#
# Needs `transformers` -> run in the `ccwm` (or `dinov3`) env, NOT `zwm`.
# No CRF. Edit the values below; do not pass anything on the CLI.

set -xeuo pipefail

# conda activate ccwm   # (has transformers + zwm importable; `zwm` env has no transformers)

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

# Variant: dinov3_s16 | dinov3_b16 | dinov3_l16 | dinov3_l16_babyview. Override: VARIANT=... bash <script>.
VARIANT="${VARIANT:-dinov3_l16}"
FEATURE_NORM=layernorm
# DINOv3 input res (patch-16): 512 -> 32x32 grid (matched to ZWM 256/8, the default);
# FEAT_RES=256 -> 16x16 grid (matched input pixels) for a fairness robustness run.
FEAT_RES="${FEAT_RES:-512}"

# tau sweep (override TAUS="..."); all emitted from one forward pass per image.
TAUS=(${TAUS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9})
# Layers to sweep. Default fits ViT-L (24 blocks); for ViT-S/B (12 blocks) use e.g.
# LAYERS="0 2 4 6 8 10 11 -1". -1 = final post-LayerNorm.
LAYERS=(${LAYERS:-0 4 8 12 16 20 23 -1})

DATA_PATH=data/evals/segments/spelke_bench.h5
OUT_DIR=viz/eval/segments/spelke_bench_featurenn

for FEATURE_LAYER in "${LAYERS[@]}"; do
    echo "=== DINOv3 ${VARIANT} feature-NN seg: layer ${FEATURE_LAYER}, taus ${TAUS[*]} ==="
    python scripts/eval/segments/baselines/eval_spelke_seg_dinov3.py \
        --model_variant "$VARIANT" \
        --dataset_path "$DATA_PATH" \
        --output_dir "$OUT_DIR" \
        --feature_layer "$FEATURE_LAYER" \
        --feature_norm "$FEATURE_NORM" \
        --bipartition_tau "${TAUS[@]}" \
        --feat_res "$FEAT_RES" \
        --device cuda
done
