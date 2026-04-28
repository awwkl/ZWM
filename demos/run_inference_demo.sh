#!/usr/bin/env bash
# End-to-end inference demo: runs ZWM factual prediction on the bundled
# demo videos in data/demo_videos/ and saves visualization images.
#
# Usage:
#   bash demos/run_inference_demo.sh
#
# Output:
#   viz/zwm_factual_predictions/<MODEL_NAME>/iter_*.png
#     Each PNG is a 7-panel figure: frame0, frame1 (ground truth), prediction,
#     prediction (unmasked), frame1 with mask, frame0 raw, frame1 raw.
#   viz/zwm_factual_predictions/<MODEL_NAME>/loss_value.txt
#     Mean MSE between the predicted and ground-truth frame1.
#
# Expected run time: ~1–2 minutes on a single NVIDIA A100 (after one-time
# checkpoint download from HuggingFace, ~30s for the 170M model).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_NAME="${MODEL_NAME:-awwkl/zwm-bvd-170m/model.pt}"
VIDEOS_DIR="${VIDEOS_DIR:-data/demo_videos/}"
N_SAMPLES="${N_SAMPLES:-10}"

# ZWMPredictor loads checkpoints from local out/. If the file is missing,
# fetch it from HuggingFace first.
HF_REPO="${MODEL_NAME%/*}"
HF_FILE="${MODEL_NAME##*/}"
LOCAL_CKPT="out/${MODEL_NAME}"
if [ ! -f "$LOCAL_CKPT" ]; then
    echo "Checkpoint not found at $LOCAL_CKPT — downloading $HF_REPO/$HF_FILE from HuggingFace..."
    python scripts/hf_model_download.py "$HF_REPO" --filename "$HF_FILE"
fi

python -m zwm.inv.inv_zwm_factual_prediction \
    --model_name "$MODEL_NAME" \
    --videos_dir "$VIDEOS_DIR" \
    --n_samples_to_eval "$N_SAMPLES" \
    --num_viz "$N_SAMPLES" \
    --frame1_mask_ratio 0.90
