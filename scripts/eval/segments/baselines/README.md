# Feature-NN segmentation — representation baselines (NOT ZWM)

Runs the **representation baselines** (DINOv3, and later V-JEPA2) through the *exact
same* matched feature-NN segmentation probe as the ZWM eval in
`../supplementary/`, so the numbers are directly comparable. These are **not** ZWM
models — they're frozen image/video backbones read with the identical readout.

The backbone-agnostic probe (cosine-NN segment rule, source-dataset loaders, lean
per-cell h5 IO) is shared from `zwm.eval.segments.feature_nn`; each script here only
adds its backbone's feature extractor (HF `transformers`). Output goes to the same
root as the ZWM probe (`viz/eval/segments/spelke_bench_featurenn/`) under a distinct
per-model slug, so the shared table grader compares everything in one place.

## Environment

Needs `transformers` — run in the **`ccwm`** (or `dinov3`) conda env, **not `zwm`**
(which has no transformers). DINOv3 weights are already cached under
`~/.cache/huggingface`. **No CRF** (so no `pydensecrf`/`promerge`).

## Run

```bash
conda activate ccwm
# DINOv3 ViT-L, layers {0,4,8,12,16,20,23,-1} x taus {0.1..0.9}; GPU=N to pick device
bash scripts/eval/segments/baselines/eval_spelke_seg_dinov3.sh
# other variants / layers:
VARIANT=dinov3_l16_babyview bash scripts/eval/segments/baselines/eval_spelke_seg_dinov3.sh
VARIANT=dinov3_b16 LAYERS="0 2 4 6 8 10 11 -1" bash scripts/eval/segments/baselines/eval_spelke_seg_dinov3.sh
```

DINOv3 is patch-16, fed at `--feat_res 512` → a **32×32 grid**, matching ZWM's
256/8 grid; masks/GT stay at `--out_res 256` like the ZWM eval.

## Grade (shared with ZWM)

```bash
python scripts/eval/segments/supplementary/grade_spelke_seg_feature_nn_table.py \
    --model_name dinov3-vitl16-r512 --metric IoU     # slug = the output-dir tag
```

Slug = `<variant>-r<feat_res>`, e.g. `dinov3-vitl16-r512` (matched grid, 32×32),
`dinov3-vitl16-r256` (matched input, 16×16), `dinov3-vitl16-babyview-r512`, etc.
This re-runs DINOv3 under the matched probe (same tau + layer sweep, same grader)
— superseding the un-matched, CRF-refined single-tau numbers in
`personal_scripts/plotting/plot_results_segments.ipynb`.
