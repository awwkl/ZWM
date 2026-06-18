"""Feature-NN (cosine patch-feature matching) segmentation eval for ZWM on SpelkeBench.

Matched, label-free probe — the segmentation counterpart to
scripts/eval/flow/supplementary/eval_tapvid_flow_feature_nn.py (flow/depth). The
representation baselines (DINOv3, V-JEPA2) are read for point-prompted
segmentation by *cosine nearest-neighbor of frozen patch features*: take the
patch feature at a seed point, score every patch by cosine similarity to it,
threshold, and keep the connected component containing the seed. This script
applies that SAME rule to ZWM's own frozen patch features, so every model is
read on identical footing and representation quality is isolated from the readout
mechanism (ZWM's native perturb-compare-aggregate mechanism is NOT used here —
see zwm/eval/segments/eval_spelke_seg.py for that).

The mask rule mirrors the standard DINOv2 / V-JEPA2 point-segmentation baseline:
seed patch -> cosine affinity -> fixed-tau threshold -> connected component
containing the seed -> upsample. (The baseline's optional CRF refinement is
omitted; it would add an external dependency for marginal gain here.)

Each image is encoded as a single frame (independent), the direct analogue of a
frozen DINOv3 backbone. One forward per image yields a patch-feature grid shared
across all of that image's segments. The feature layer is configurable and
recorded in the output path (`..._featurenn_layer{L}_{norm}`); sweep layers with
the .sh wrapper.

Output is written in the SAME per-image h5 schema as the native seg eval
(image / segment_gt / segment_pred), so the existing grader applies unchanged:
    python -m zwm.eval.segments.grade_spelke_seg --input_dir <out>/<slug>_featurenn_layer{L}_{norm}

Run via scripts/eval/segments/supplementary/eval_spelke_seg_feature_nn.sh.
Minimal direct invocation (smoke test on 2 images):

    python scripts/eval/segments/supplementary/eval_spelke_seg_feature_nn.py \\
        --model_name awwkl/zwm-bvd-170m/model.pt \\
        --dataset_path data/evals/segments/spelke_bench.h5 \\
        --output_dir viz/eval/segments/spelke_bench_featurenn \\
        --feature_layer 12 --img_names image_3345 image_3346
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from zwm.data.image_processing import patchify_image
from zwm.data.sequence_construction import get_pos_idxs
from zwm.zwm_predictor import ZWMPredictor
from zwm.eval.segments.feature_nn import (
    load_image, load_segments, load_centroids,
    featurenn_segment, has_predictions, write_result_h5,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, required=True,
                   help='ZWM checkpoint relative to out/, e.g. awwkl/zwm-bvd-170m/model.pt')
    p.add_argument('--dataset_path', type=str, required=True,
                   help='SpelkeBench-style h5, e.g. data/evals/segments/spelke_bench.h5')
    p.add_argument('--output_dir', type=str, required=True,
                   help='Worker writes <model_slug>_featurenn_layer{L}_{norm}/<key>.h5 under here.')
    p.add_argument('--img_names', type=str, nargs='+', default=None,
                   help='h5 keys to process. Default: all keys (single-GPU full run). '
                        'Pass a subset for smoke tests or manual sharding.')

    # Feature-NN probe knobs (mirror the flow feature_nn eval).
    p.add_argument('--feature_layer', type=int, default=12,
                   help='Transformer block output to match on (0-indexed). -1 = final layer '
                        'after the final LayerNorm ln_f. The .sh wrapper sweeps layers in series.')
    p.add_argument('--feature_norm', type=str, default='layernorm', choices=['none', 'layernorm'],
                   help='Parameter-free LayerNorm over the feature dim before cosine matching '
                        '(centers + scales so layers are comparable; matches V-JEPA2). '
                        '"none" keeps raw block outputs (ablation).')
    p.add_argument('--bipartition_tau', type=float, nargs='+', default=[0.2],
                   help='Cosine-similarity threshold(s) for the segment mask (DINOv2 / V-JEPA2 '
                        'point-seg baseline default 0.2). Pass several to sweep tau — all are '
                        'emitted from ONE forward pass per image (tau only thresholds the cosine '
                        'map), each into its own output dir (..._tau{T}).')

    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


# ------------------------------------------------------------
# Frozen patch-feature extraction — copied from
# scripts/eval/flow/supplementary/eval_tapvid_flow_feature_nn.py (extract_patch_features,
# _frame_to_patches). Those live in a sibling script, not an importable module, so the
# patchify/encode logic is duplicated here and kept in sync by hand. See that file for the
# full rationale; segmentation uses only the single-frame (independent) path.
# ------------------------------------------------------------

@torch.no_grad()
def extract_patch_features(predictor, frame_np, layer, feature_norm='layernorm'):
    """Run a single frame through the frozen ZWM transformer and return its
    patch-feature grid as [num_patches, n_embd], float32 on the model device.

    Each frame is encoded independently with all patches visible (mask all
    zeros) — the direct analogue of a frozen DINOv3 / V-JEPA2 backbone. Attention
    is bidirectional, so a zeros mask is a no-op for attention.

    layer: 0-indexed transformer block whose output to return; -1 returns the
    final block output after the model's ln_f (the patch_head input).

    feature_norm: 'layernorm' applies a parameter-free LayerNorm over the feature
    dim before returning (matches V-JEPA2), so layers are comparable; 'none'
    returns the raw output. For layer -1 it is added on top of ln_f.
    """
    model = predictor.model
    num_patches = (model.config.resolution // model.config.patch_size) ** 2

    patches = _frame_to_patches(predictor, frame_np)
    seq = patches.reshape(1, num_patches, -1).to(predictor.device)
    pos = get_pos_idxs(patches, 0).reshape(1, -1).long().to(predictor.device)
    mask = torch.zeros((1, num_patches), device=predictor.device)

    n_blocks = len(model.transformer.h)
    target = layer if layer >= 0 else n_blocks - 1

    with predictor.ctx:
        # mask is all zeros, so the mask-token term vanishes; embed + position.
        h = model.transformer.token_embedding(seq) + model.transformer.positional_embedding(pos)
        h = model.transformer.drop(h)
        for i, block in enumerate(model.transformer.h):
            h = block(h, mask=mask)
            if i == target:
                break
        if layer < 0:
            h = model.transformer.ln_f(h)            # model's own final norm (final layer only)
        if feature_norm == 'layernorm':
            h = F.layer_norm(h, (h.shape[-1],))      # parameter-free centering — matches V-JEPA2

    return h.squeeze(0).float()  # [num_patches, n_embd]


def _frame_to_patches(predictor, frame_np):
    """PIL/np frame -> patchified tokens [1, num_patches, patch_size**2 * 3] on CPU."""
    resolution = predictor.model.config.resolution
    patch_size = predictor.model.config.patch_size
    pil = frame_np if isinstance(frame_np, Image.Image) else Image.fromarray(frame_np)
    pil = predictor.resize_crop_transform(pil, resolution)
    img = predictor.in_transform(pil).unsqueeze(0).permute(0, 2, 3, 1)
    return patchify_image(img, patch_size)


# The backbone-agnostic probe (featurenn_segment), source loaders, and lean h5 IO
# (has_predictions / write_result_h5) are shared with the baseline probes — they live
# in zwm.eval.segments.feature_nn. This file only adds ZWM's own feature extractor.


def main():
    args = parse_args()

    norm_tag = args.feature_norm
    layer_desc = 'final_lnf' if args.feature_layer < 0 else f'block{args.feature_layer}'
    model_slug = args.model_name.replace('/', '_').replace('.pt', '')
    taus = list(args.bipartition_tau)

    # One output dir per tau — all share the SAME single forward pass per image
    # (tau only thresholds the cosine map, so emitting every tau is essentially free).
    out_dirs = {}
    for tau in taus:
        d = os.path.join(args.output_dir,
                         f"{model_slug}_featurenn_layer{args.feature_layer}_{norm_tag}_tau{tau:g}")
        os.makedirs(d, exist_ok=True)
        out_dirs[tau] = d

    predictor = ZWMPredictor(model_name=args.model_name, device=args.device)
    assert 'ZWM2' not in predictor.model.config.model_class, \
        "This feature-NN seg eval targets the ZWM (non-ZWM2) architecture."

    res = predictor.model.config.resolution
    patch_size = predictor.model.config.patch_size
    grid = res // patch_size
    n_blocks = predictor.model.config.n_layer
    assert args.feature_layer < n_blocks, \
        f"--feature_layer {args.feature_layer} out of range for a {n_blocks}-block model."
    print(f"Feature-NN seg at {res}x{res}, {grid}x{grid} patch grid, layer {args.feature_layer} "
          f"({layer_desc}) of {n_blocks}, norm={norm_tag}, taus={[f'{t:g}' for t in taus]}")

    with h5py.File(args.dataset_path, 'r') as inp:
        img_names = args.img_names if args.img_names is not None else sorted(inp.keys())
        for img_key in img_names:
            # Which taus still need this image (resume-safe, per tau)?
            pending = [t for t in taus
                       if not has_predictions(os.path.join(out_dirs[t], f"{img_key}.h5"))]
            if not pending:
                print(f"[skip] {img_key} (all {len(taus)} taus done)")
                continue

            # image + GT segments come from the source dataset (not stored per cell).
            im = load_image(inp, img_key, res)
            segments = load_segments(inp, img_key, res)
            centroids = load_centroids(inp, img_key, segments)

            # ONE forward per image; the feature grid is shared across all segments AND all taus.
            feats = extract_patch_features(predictor, im, args.feature_layer, args.feature_norm)

            print(f"[{img_key}] {segments.shape[0]} segments, {len(pending)} tau(s)")
            for tau in pending:
                preds = np.stack([featurenn_segment(feats, c, grid, res, tau) for c in centroids])
                write_result_h5(os.path.join(out_dirs[tau], f"{img_key}.h5"), preds)
            print(f"[done] {img_key} -> tau {[f'{t:g}' for t in pending]}")


if __name__ == "__main__":
    main()
