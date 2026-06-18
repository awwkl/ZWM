# Feature-NN (cosine patch-feature matching) — segmentation

The **segmentation** counterpart to `scripts/eval/flow/supplementary/` (flow +
depth). It reads object segments out of **BabyZWM's own frozen patch features**
with the *same* rule used for the DINOv3 / V-JEPA2 point-segmentation baselines,
holding the readout fixed across all models to isolate representation quality
from readout mechanism — a matched, label-free comparison on SpelkeBench.

This is **not** BabyZWM's native perturb-compare-aggregate mechanism (that lives
in `scripts/eval/segments/eval_spelke_seg.sh`). The contrast localizes any
advantage to the mechanism rather than to the features or the probe.

## The probe

Encode the image as a **single frame** (`independent`) and take its patch-feature
grid at a chosen layer — the direct analogue of a frozen DINOv3 backbone. For
each GT segment's **seed centroid**, score every patch by **cosine similarity**
to the seed patch, threshold at `--bipartition_tau`, and keep the **connected
component containing the seed**; upsample that coarse grid mask (32×32 at 256px,
patch 8) to the image. This is the DINOv2 / V-JEPA2 `segment_at_point` rule, minus
the CRF refinement (`densecrf` is unavailable in ZWM).

**One forward per image** yields the feature grid, shared across all of that
image's segments **and across all taus** — tau only thresholds the cosine map, so
`--bipartition_tau` accepts a list and emits a segment set per tau from that single
pass, each into its own `..._tau{T}` dir.

## Sweep: feature layer × bipartition tau

`--feature_layer` (0-indexed block; `-1` = final block after the final LayerNorm)
and tau are both **recorded in every output path**
(`..._featurenn_layer{L}_{norm}_tau{T}`). The `.sh` wrapper sweeps `LAYERS` in
series (one forward pass per layer per image) and `TAUS` for free within each pass.
Default model is **bvd-1b** (48 layers); override with `CKPT=`.

ZWM is **pre-norm**, so a raw intermediate block output is an un-normalized
residual-stream activation. By default (`--feature_norm layernorm`) every feature
gets a parameter-free LayerNorm (center + scale) before matching — the same
`F.layer_norm` V-JEPA2 applies — making layers comparable; `--feature_norm none`
keeps raw outputs (ablation).

## Run

```bash
conda activate zwm

# Sweeps LAYERS x TAUS for bvd-1b on one GPU (GPU=N to pick the device).
bash scripts/eval/segments/supplementary/eval_spelke_seg_feature_nn.sh
```

Outputs go to
`viz/eval/segments/spelke_bench_featurenn/<slug>_featurenn_layer{L}_{norm}_tau{T}/<key>.h5`.
Each cell file stores **only `segment_pred`** — `image` and `segment_gt` are
identical across every cell, so the graders read them back from the source
`spelke_bench.h5` (via `zwm.eval.segments.feature_nn`) rather than duplicating them ~64× per
image. This makes each file ~2% of an all-in-one copy (the RGB image was ~96%).

## Grade

Both graders read predictions from the cell files and GT from `--dataset_path`.

**Whole layer × tau table** (mean AP / AR / IoU per cell, safe to run mid-sweep):

```bash
python scripts/eval/segments/supplementary/grade_spelke_seg_feature_nn_table.py \
    --model_name awwkl/zwm-bvd-1b/model.pt --metric IoU
```

**One cell** (set `LAYER` / `TAU` / `FEATURE_NORM`), also writes overlay PNGs
(image pulled from source):

```bash
LAYER=24 TAU=0.4 bash scripts/eval/segments/supplementary/grade_spelke_seg_feature_nn.sh
```

Both report mean AP / AR / IoU across the dataset (Hungarian-matched, IoU
0.50:0.05:0.95).
