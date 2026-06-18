# Feature-NN (cosine patch-feature matching) — matched, label-free probe

This folder reads optical flow and relative depth out of **BabyZWM's own frozen
patch features** with the *same* nearest-neighbor probe used for the
representation baselines (DINOv3, V-JEPA2, ResNet50). Holding the readout fixed
across all models isolates representation quality from readout mechanism — a
matched, label-free comparison on identical footing.

This is **not** BabyZWM's native perturb-compare-aggregate mechanism (that lives
in `scripts/eval/flow/eval_tapvid_flow.sh` and `scripts/eval/depth/`). The point
is the contrast: BabyZWM read with this matched feature-NN probe vs. BabyZWM
read with its native mechanism localizes any advantage to the mechanism rather
than to the features or the probe.

## The probe

Take a patch-feature grid for each frame at a chosen layer. For a query patch
in frame A, the predicted correspondence in frame B is the **argmax of cosine
similarity** over B's patch features; the displacement is the flow vector. As
with the baselines, **iterative zoom** (`--zoom_iters 4`) raises the effective
resolution above the coarse 32×32 patch grid (8px patches at 256px for the
170M model).

### Frame encoding (`--encoding`)

- **`in_context`** (default): both frames go through the model in a **single
  two-frame forward, both fully visible**, so they attend to each other — ZWM's
  native two-frame setting, where the model relates the frames. (Training masks
  frame 1; here both are visible so frame 1 carries real content to match.)
  This mirrors how the **V-JEPA2** baseline is read: V-JEPA2 is a video encoder,
  so its feature-NN probe also encodes both frames jointly (one clip, split into
  per-frame grids) with a parameter-free LayerNorm — see `feature_norm` below.
- **`independent`**: each frame is encoded alone — the direct analogue of the
  frozen **DINOv3** image backbone, which has no two-frame mode.

- **Flow**: TAP-Vid-DAVIS "first" two-frame pairs (38,881 points).
- **Depth**: the same correspondence on **binocular stereo** pairs (disparity ≈
  inverse depth); 164 pairs / 328 points. Same script, stereo dataset.

## Feature layer

The layer matched on is configurable (`--feature_layer`, 0-indexed block;
`-1` = final block after the final LayerNorm, i.e. the patch_head input) and is
**recorded in every output path** (`..._featurenn_{enc}_layer{L}_{norm}`). The `.sh`
wrappers sweep a set of layers **in series**, leading with the **middle layer
(12 of 24)** as the headline and then stepping every 4 layers across the stack
(plus `-1`). Edit `LAYERS=(...)` in the wrappers to change the sweep.

ZWM is a **pre-norm** transformer, so a raw intermediate block output is an
un-normalized residual-stream activation whose scale grows with depth and which
carries a large common-mode component that raw cosine cannot remove. So by
default (`--feature_norm layernorm`) every feature gets a parameter-free
LayerNorm (center + scale) before matching — the same `F.layer_norm` V-JEPA2
applies to its encoder output — making layers comparable. `--feature_norm none`
keeps raw outputs (ablation). For `--feature_layer -1` the base feature already
carries the model's final `ln_f`, and `layernorm` adds the parameter-free
LayerNorm on top (matching V-JEPA2 exactly); the path is tagged with the
`feature_norm` value (`layernorm`/`none`) like every other layer.

## Run

```bash
conda activate zwm

# Flow (full TAP-Vid-DAVIS), sweeps layers 12,0,4,8,16,20,-1 in series
bash scripts/eval/flow/supplementary/eval_tapvid_flow_feature_nn.sh

# Relative depth (stereo), same layer sweep
bash scripts/eval/flow/supplementary/eval_stereo_depth_feature_nn.sh
```

Outputs go to `viz/eval/flow/tapvid_davis_first_featurenn/zoom_4/<slug>_featurenn_{enc}_layer{L}_{norm}/`
(flow) and the analogous `stereo_depth_featurenn` path (depth), where `{enc}` is
`incontext`/`independent` and `{norm}` is `layernorm`/`none`. To shard across
GPUs, copy a wrapper and edit `START_IDX` / `NUM_POINTS`.

## Grade

Output is written in the **same format** as the native evals, so the existing
graders apply unchanged — just point `ROOT_DIR` at the per-layer dir. For each
layer `L`:

```bash
# Flow — set ROOT_DIR in the grader, or call directly:
python -m zwm.eval.flow.grade_tapvid_flow \
    --root_dir viz/eval/flow/tapvid_davis_first_featurenn/zoom_4/awwkl_zwm-bvd-170m_model_featurenn_incontext_layer12_layernorm \
    --pkl_path data/evals/flow/tapvid_davis_first/tapvid_davis.pkl \
    --occ_thresh 0.05

# Depth:
python -m zwm.eval.depth.grade_stereo_depth \
    --root_dir viz/eval/depth/stereo_depth_featurenn/zoom_4/awwkl_zwm-bvd-170m_model_featurenn_incontext_layer12_layernorm \
    --data_path data/evals/depth/stereo_depth/dataset.json
```

The headline flow metric is **Pct** (pixel-threshold tracking accuracy / position
accuracy); depth reports accuracy + bootstrap CI.

Note: `occ_metric` here is the **max cosine similarity** of the match (higher =
more confident, lower = likely occluded), and `pred_occ` uses `occ_metric < 0.05`
— the **same occlusion convention as the DINOv3 / V-JEPA2 baselines**. Grade with
`--occ_thresh 0.05` for the matched comparison (not the native ZWM `0.4`, which is
tuned for a different L1-norm metric). This only affects OA/OF1, not the Pct headline.

## Scope

This covers **flow** and **depth** (the two correspondence tasks). The
**segmentation** counterpart — group patches by thresholding cosine similarity to
a seed patch on SpelkeBench — uses a different matching rule and lives in
`scripts/eval/segments/supplementary/`.
