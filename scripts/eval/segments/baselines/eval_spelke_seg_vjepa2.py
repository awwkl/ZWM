"""Feature-NN segmentation probe for V-JEPA 2 on SpelkeBench — a REPRESENTATION
BASELINE (not a ZWM model), run through the EXACT same probe as the ZWM feature-NN
eval so the two are directly comparable.

V-JEPA 2 is a video encoder, so a static image is encoded as a short clip (the
frame repeated to `num_frames`, the standard image-as-video trick), the tubelet
tokens are averaged over the (single) temporal step to a spatial [num_patches, D]
grid, then the shared cosine-NN rule runs (parameter-free LayerNorm, tau threshold,
connected component, upsample). Backbone-agnostic pieces are shared via
zwm.eval.segments.feature_nn; this file only adds V-JEPA 2's feature extractor.

Patch-16, so --feat_res 512 -> 32x32 grid (matching ZWM 256/8); --feat_res 256 ->
16x16 (matched input). Masks/GT at --out_res 256 like the ZWM eval. Slug carries the
resolution (vjepa2-vitl-r512 / -r256), gradable by grade_spelke_seg_feature_nn_table.py.

Needs `transformers` (AutoVideoProcessor) — run in `ccwm` / `dinov3`, NOT `zwm`. No CRF.
Run via eval_spelke_seg_vjepa2.sh.
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel

try:
    from transformers import AutoVideoProcessor as _ProcessorCls
except Exception:  # older transformers
    from transformers import AutoProcessor as _ProcessorCls

from zwm.eval.segments.feature_nn import (
    load_image, load_segments, load_centroids,
    featurenn_segment, has_predictions, write_result_h5,
)

CONFIGS = {
    'vjepa2_vitl':          {'hf_id': 'facebook/vjepa2-vitl-fpc64-256',                     'slug': 'vjepa2-vitl'},
    'vjepa2_vitg':          {'hf_id': 'facebook/vjepa2-vitg-fpc64-384',                     'slug': 'vjepa2-vitg'},
    'vjepa2_vitl_babyview': {'hf_id': 'awwkl/vjepa2-vitl-fpc16-256-babyview-bs3072-e140',   'slug': 'vjepa2-vitl-babyview'},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_variant', type=str, required=True, choices=list(CONFIGS.keys()))
    p.add_argument('--dataset_path', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--img_names', type=str, nargs='+', default=None)

    p.add_argument('--feature_layer', type=int, default=-1,
                   help='Encoder block output (0-indexed); -1 = final (last_hidden_state). '
                        'The .sh wrapper sweeps layers in series.')
    p.add_argument('--feature_norm', type=str, default='layernorm', choices=['none', 'layernorm'])
    p.add_argument('--bipartition_tau', type=float, nargs='+', default=[0.2])

    p.add_argument('--feat_res', type=int, default=512,
                   help='Resolution fed to V-JEPA 2 (patch 16 -> grid feat_res/16; 512 -> 32x32).')
    p.add_argument('--out_res', type=int, default=256)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def setup_model(hf_id, device):
    processor = _ProcessorCls.from_pretrained(hf_id)
    # We resize to feat_res ourselves; disable the processor's own resize/crop so feat_res
    # genuinely sets the grid (otherwise V-JEPA2's processor downsizes everything to native
    # 256 -> 16x16). NB: feat_res > native (e.g. 512) is then a true OOD run for V-JEPA2.
    for attr in ('do_resize', 'do_center_crop'):
        if hasattr(processor, attr):
            setattr(processor, attr, False)
    model = AutoModel.from_pretrained(hf_id).eval().to(device)
    cfg = model.config
    patch_size = int(getattr(cfg, 'patch_size', 16))
    tubelet_size = int(getattr(cfg, 'tubelet_size', 2))
    n_blocks = int(getattr(cfg, 'num_hidden_layers', 0)) or None
    return processor, model, patch_size, tubelet_size, n_blocks


@torch.no_grad()
def extract_features(model, processor, im_np, layer, feature_norm, patch_size, tubelet_size, device):
    """V-JEPA 2 patch grid for one image -> ([num_patches, D] row-major, grid).

    The image is encoded as a `num_frames`-frame clip (num_frames = tubelet_size, so the
    single temporal tubelet collapses to one spatial grid). layer: encoder block (-1 = final).
    """
    inputs = processor(Image.fromarray(im_np), return_tensors='pt')
    pv = inputs['pixel_values_videos'] if 'pixel_values_videos' in inputs else inputs['pixel_values']
    if pv.ndim == 4:
        pv = pv.unsqueeze(1)                      # [B,1,C,H,W]
    pv = pv.to(device)

    num_frames = tubelet_size                     # -> t_feat = 1 (purely spatial)
    B, T0, C, H, W = pv.shape
    if T0 != num_frames:
        reps = (num_frames + T0 - 1) // T0
        pv = pv.repeat(1, reps, 1, 1, 1)[:, :num_frames]

    t_feat = num_frames // tubelet_size
    h_feat, w_feat = H // patch_size, W // patch_size
    expected = t_feat * h_feat * w_feat

    out = model(pixel_values_videos=pv, output_hidden_states=True, skip_predictor=True, return_dict=True)
    toks = out.last_hidden_state if layer < 0 else out.hidden_states[layer + 1]   # [B, N, D]

    if toks.shape[1] != expected:                 # drop any prefix (CLS/register-like)
        prefix = toks.shape[1] - expected
        if 0 < prefix < 32:
            toks = toks[:, prefix:, :]
        else:
            raise RuntimeError(f"token mismatch: got {toks.shape[1]}, expected {expected} "
                               f"(H={H}, W={W}, patch={patch_size}, t={t_feat}).")

    toks = toks.reshape(B, t_feat, h_feat, w_feat, -1).mean(dim=1)[0]   # [h, w, D]
    feats = toks.reshape(h_feat * w_feat, -1).float()                  # [num_patches, D], row-major
    if feature_norm == 'layernorm':
        feats = F.layer_norm(feats, (feats.shape[-1],))
    return feats, h_feat


def main():
    args = parse_args()
    cfg = CONFIGS[args.model_variant]
    norm_tag = args.feature_norm
    slug = f"{cfg['slug']}-r{args.feat_res}"      # resolution tagged (matches the DINOv3 convention)

    out_dirs = {}
    for tau in args.bipartition_tau:
        d = os.path.join(args.output_dir, f"{slug}_featurenn_layer{args.feature_layer}_{norm_tag}_tau{tau:g}")
        os.makedirs(d, exist_ok=True)
        out_dirs[tau] = d

    processor, model, patch_size, tubelet_size, n_blocks = setup_model(cfg['hf_id'], args.device)
    if n_blocks is not None and args.feature_layer >= n_blocks:
        raise ValueError(f"--feature_layer {args.feature_layer} out of range for a {n_blocks}-block model.")
    grid_preview = args.feat_res // patch_size
    print(f"V-JEPA2 {args.model_variant} ({cfg['hf_id']}): feat_res {args.feat_res} -> {grid_preview}x{grid_preview} "
          f"grid (patch {patch_size}, tubelet {tubelet_size}), layer {args.feature_layer}, norm={norm_tag}, "
          f"taus={[f'{t:g}' for t in args.bipartition_tau]}; out_res {args.out_res}")

    with h5py.File(args.dataset_path, 'r') as inp:
        img_names = args.img_names if args.img_names is not None else sorted(inp.keys())
        for img_key in img_names:
            pending = [t for t in args.bipartition_tau
                       if not has_predictions(os.path.join(out_dirs[t], f"{img_key}.h5"))]
            if not pending:
                print(f"[skip] {img_key} (all {len(args.bipartition_tau)} taus done)")
                continue

            segments = load_segments(inp, img_key, args.out_res)
            centroids = load_centroids(inp, img_key, segments)
            im_feat = load_image(inp, img_key, args.feat_res)

            feats, grid = extract_features(model, processor, im_feat, args.feature_layer,
                                           args.feature_norm, patch_size, tubelet_size, args.device)

            print(f"[{img_key}] {segments.shape[0]} segments, {len(pending)} tau(s)")
            for tau in pending:
                preds = np.stack([featurenn_segment(feats, c, grid, args.out_res, tau) for c in centroids])
                write_result_h5(os.path.join(out_dirs[tau], f"{img_key}.h5"), preds)
            print(f"[done] {img_key} -> tau {[f'{t:g}' for t in pending]}")


if __name__ == "__main__":
    main()
