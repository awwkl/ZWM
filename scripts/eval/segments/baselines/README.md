# Feature-NN segmentation — representation baselines (NOT ZWM)

Runs the **representation baselines** (DINOv3, V-JEPA2) through the *exact same*
matched feature-NN segmentation probe as the ZWM eval in `../supplementary/`, so
the numbers are directly comparable. These are **not** ZWM models — they're frozen
image/video backbones read with the identical readout.

The backbone-agnostic probe (cosine-NN segment rule, source-dataset loaders, lean
per-cell h5 IO) is shared from `zwm.eval.segments.feature_nn`; each script here only
adds its backbone's feature extractor (HF `transformers`). Output goes to the same
root as the ZWM probe (`viz/eval/segments/spelke_bench_featurenn/`) under a distinct
per-model slug, so the shared table grader compares everything in one place.

## Environment

Needs `transformers` (DINOv3) + `AutoVideoProcessor` (V-JEPA2) — run in the
**`ccwm`** (or `dinov3`) conda env, **not `zwm`** (which has no transformers).
Weights are already cached under `~/.cache/huggingface`. **No CRF** (so no
`pydensecrf`/`promerge`).

## Run

```bash
conda activate ccwm

# --- DINOv3 (patch-16, resolution-flexible: run BOTH 512->32x32 and 256->16x16) ---
GPU=N            bash scripts/eval/segments/baselines/eval_spelke_seg_dinov3.sh   # 512 (matched grid)
GPU=N FEAT_RES=256 bash scripts/eval/segments/baselines/eval_spelke_seg_dinov3.sh # 256 (matched input)
GPU=N VARIANT=dinov3_l16_babyview bash scripts/eval/segments/baselines/eval_spelke_seg_dinov3.sh

# --- V-JEPA2 (patch-16 video encoder; use NATIVE 256 -> 16x16; 512 is OOD) ---
GPU=N            bash scripts/eval/segments/baselines/eval_spelke_seg_vjepa2.sh   # 256 (native, default)
GPU=N VARIANT=vjepa2_vitl_babyview bash scripts/eval/segments/baselines/eval_spelke_seg_vjepa2.sh
```

DINOv3 is resolution-flexible, so it runs at both 512 (32×32, matched grid) and
256 (16×16, matched input). **V-JEPA2 is trained at crop 256** — feeding 512 runs
but is out-of-distribution, so use its native **256** (16×16). Masks/GT stay at
`--out_res 256` like the ZWM eval for all of them.

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
