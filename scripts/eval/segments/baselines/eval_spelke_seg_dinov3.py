"""Feature-NN segmentation probe for DINOv3 on SpelkeBench — a REPRESENTATION
BASELINE (not a ZWM model), run through the EXACT same probe as the ZWM feature-NN
eval so the two are directly comparable.

DINOv3 is read with the same readout as scripts/eval/segments/supplementary/: encode
the image, take a patch-feature grid at a chosen layer (parameter-free LayerNorm),
cosine-affinity from each GT segment's seed centroid, threshold at bipartition tau,
keep the connected component containing the seed, upsample. The backbone-agnostic
pieces (probe, source loaders, lean IO) are shared via zwm.eval.segments.feature_nn;
this file only swaps in DINOv3's feature extractor (HF transformers).

DINOv3 is patch-16, so feeding it at --feat_res 512 gives a 32x32 grid — matching
ZWM's 256/8 grid — while masks/GT stay at --out_res 256 like the ZWM eval, so the
output is directly gradable by grade_spelke_seg_feature_nn_table.py.

Needs `transformers` (present in the `ccwm` / `dinov3` conda envs, NOT `zwm`); no CRF.
Run via eval_spelke_seg_dinov3.sh. Minimal direct invocation:

    python scripts/eval/segments/baselines/eval_spelke_seg_dinov3.py \\
        --model_variant dinov3_l16 --dataset_path data/evals/segments/spelke_bench.h5 \\
        --output_dir viz/eval/segments/spelke_bench_featurenn \\
        --feature_layer -1 --bipartition_tau 0.2 0.3 --img_names entityseg_1_image1007
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from zwm.eval.segments.feature_nn import (
    load_image, load_segments, load_centroids,
    featurenn_segment, has_predictions, write_result_h5,
)

# Slug = the output-dir model tag; grade with grade_spelke_seg_feature_nn_table.py
# --model_name <slug>. DINOv3 weights are cached under ~/.cache/huggingface.
CONFIGS = {
    'dinov3_s16':          {'hf_id': 'facebook/dinov3-vits16-pretrain-lvd1689m', 'slug': 'dinov3-vits16'},
    'dinov3_b16':          {'hf_id': 'facebook/dinov3-vitb16-pretrain-lvd1689m', 'slug': 'dinov3-vitb16'},
    'dinov3_l16':          {'hf_id': 'facebook/dinov3-vitl16-pretrain-lvd1689m', 'slug': 'dinov3-vitl16'},
    'dinov3_l16_babyview': {'hf_id': 'awwkl/dinov3-vitl-babyview-gradaccum1',    'slug': 'dinov3-vitl16-babyview'},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_variant', type=str, required=True, choices=list(CONFIGS.keys()))
    p.add_argument('--dataset_path', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--img_names', type=str, nargs='+', default=None,
                   help='h5 keys to process. Default: all keys.')

    p.add_argument('--feature_layer', type=int, default=-1,
                   help='Transformer block output (0-indexed); -1 = final post-LayerNorm '
                        '(last_hidden_state). The .sh wrapper sweeps layers in series.')
    p.add_argument('--feature_norm', type=str, default='layernorm', choices=['none', 'layernorm'],
                   help='Parameter-free LayerNorm over the feature dim before matching, to match '
                        'the ZWM probe; "none" = raw features.')
    p.add_argument('--bipartition_tau', type=float, nargs='+', default=[0.2],
                   help='Cosine threshold(s); all emitted from one forward per image (each into '
                        'its own ..._tau{T} dir).')

    p.add_argument('--feat_res', type=int, default=512,
                   help='Resolution fed to DINOv3 (patch 16 -> grid feat_res/16; 512 -> 32x32, '
                        'matching ZWM 256/8).')
    p.add_argument('--out_res', type=int, default=256,
                   help='Resolution of GT/centroids/output masks (matches the ZWM eval).')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def setup_model(hf_id, device):
    processor = AutoImageProcessor.from_pretrained(hf_id)
    # We resize to feat_res ourselves; disable processor resize/crop.
    for attr in ('do_resize', 'do_center_crop'):
        if hasattr(processor, attr):
            setattr(processor, attr, False)
    model = AutoModel.from_pretrained(hf_id).eval().to(device)
    patch_size = int(getattr(model.config, 'patch_size', 16))
    num_register = int(
        getattr(model.config, 'num_register_tokens', 0)
        or getattr(model.config, 'num_registers', 0)
        or 0
    )
    n_blocks = int(getattr(model.config, 'num_hidden_layers', 0)) or None
    return processor, model, patch_size, num_register, n_blocks


@torch.no_grad()
def extract_features(model, processor, im_np, layer, feature_norm, patch_size, num_register, device):
    """DINOv3 patch grid for one image -> ([num_patches, C] row-major, grid).

    im_np: [feat_res, feat_res, 3] uint8. layer: block index (-1 = final post-norm).
    """
    inputs = processor(images=Image.fromarray(im_np), return_tensors='pt')
    pv = inputs['pixel_values'].to(device)
    out = model(pixel_values=pv, output_hidden_states=True, return_dict=True)

    # hidden_states = (embeddings, block_0_out, ..., block_{N-1}_out); last_hidden_state = post-final-norm.
    if layer < 0:
        h = out.last_hidden_state
    else:
        h = out.hidden_states[layer + 1]

    prefix = 1 + num_register                       # drop CLS + register tokens
    toks = h[:, prefix:, :]                          # [1, num_patches, C], row-major
    H, W = int(pv.shape[-2]), int(pv.shape[-1])
    feat_h, feat_w = H // patch_size, W // patch_size
    if toks.shape[1] != feat_h * feat_w:
        raise RuntimeError(f"token mismatch: got {toks.shape[1]}, expected {feat_h * feat_w} "
                           f"(H={H}, W={W}, patch={patch_size}, prefix={prefix}).")

    feats = toks.squeeze(0).float()                  # [num_patches, C]
    if feature_norm == 'layernorm':
        feats = F.layer_norm(feats, (feats.shape[-1],))
    return feats, feat_h                             # square grid


def main():
    args = parse_args()
    cfg = CONFIGS[args.model_variant]
    norm_tag = args.feature_norm
    # DINOv3 is patch-16, so feat_res sets the grid (512 -> 32x32, matching ZWM's 256/8;
    # 256 -> 16x16, matching ZWM's input). feat_res is always tagged into the slug so the
    # two resolutions live in separate dirs (dinov3-vitl16-r512 / dinov3-vitl16-r256).
    slug = f"{cfg['slug']}-r{args.feat_res}"

    out_dirs = {}
    for tau in args.bipartition_tau:
        d = os.path.join(args.output_dir, f"{slug}_featurenn_layer{args.feature_layer}_{norm_tag}_tau{tau:g}")
        os.makedirs(d, exist_ok=True)
        out_dirs[tau] = d

    processor, model, patch_size, num_register, n_blocks = setup_model(cfg['hf_id'], args.device)
    if n_blocks is not None and args.feature_layer >= n_blocks:
        raise ValueError(f"--feature_layer {args.feature_layer} out of range for a {n_blocks}-block model.")
    grid_preview = args.feat_res // patch_size
    print(f"DINOv3 {args.model_variant} ({cfg['hf_id']}): feat_res {args.feat_res} -> {grid_preview}x{grid_preview} "
          f"grid (patch {patch_size}, {num_register} registers), layer {args.feature_layer}, norm={norm_tag}, "
          f"taus={[f'{t:g}' for t in args.bipartition_tau]}; out_res {args.out_res}")

    with h5py.File(args.dataset_path, 'r') as inp:
        img_names = args.img_names if args.img_names is not None else sorted(inp.keys())
        for img_key in img_names:
            pending = [t for t in args.bipartition_tau
                       if not has_predictions(os.path.join(out_dirs[t], f"{img_key}.h5"))]
            if not pending:
                print(f"[skip] {img_key} (all {len(args.bipartition_tau)} taus done)")
                continue

            segments = load_segments(inp, img_key, args.out_res)   # GT at out_res (for centroids only)
            centroids = load_centroids(inp, img_key, segments)
            im_feat = load_image(inp, img_key, args.feat_res)      # model input at feat_res

            feats, grid = extract_features(model, processor, im_feat, args.feature_layer,
                                           args.feature_norm, patch_size, num_register, args.device)

            print(f"[{img_key}] {segments.shape[0]} segments, {len(pending)} tau(s)")
            for tau in pending:
                preds = np.stack([featurenn_segment(feats, c, grid, args.out_res, tau) for c in centroids])
                write_result_h5(os.path.join(out_dirs[tau], f"{img_key}.h5"), preds)
            print(f"[done] {img_key} -> tau {[f'{t:g}' for t in pending]}")


if __name__ == "__main__":
    main()
