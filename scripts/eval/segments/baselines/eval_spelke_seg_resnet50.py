"""Feature-NN segmentation probe for ResNet-50 (ImageNet-supervised) on SpelkeBench —
a REPRESENTATION BASELINE, run through the EXACT same probe as the ZWM feature-NN eval.

ResNet is convolutional, so the "patch grid" is a conv-stage feature map. We sweep the
stages (the ResNet analog of a ViT layer sweep) AND bipartition tau, then take the best —
same matched readout (cosine-NN from a seed, threshold, connected component, upsample) as
the other baselines, via zwm.eval.segments.feature_nn. At --feat_res 256 the stages give
64/64/32/16/8 grids (strides 4/4/8/16/32); stage2 (32x32) matches ZWM, stage3 (16x16)
matches DINOv3-256.

microsoft/resnet-50 via HF transformers. Needs `transformers` (run in `ccwm`/`dinov3`, NOT
`zwm`). No CRF. Run via eval_spelke_seg_resnet50.sh.
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

CONFIGS = {
    'resnet50': {'hf_id': 'microsoft/resnet-50', 'slug': 'resnet50'},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_variant', type=str, default='resnet50', choices=list(CONFIGS.keys()))
    p.add_argument('--dataset_path', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--img_names', type=str, nargs='+', default=None)

    p.add_argument('--feature_layer', type=int, default=2,
                   help='Which 4D conv feature map (0-indexed; -1 = last/stage4). At feat_res 256: '
                        '0,1=64x64; 2=32x32; 3=16x16; 4=8x8. The .sh wrapper sweeps these stages.')
    p.add_argument('--feature_norm', type=str, default='layernorm', choices=['none', 'layernorm'])
    p.add_argument('--bipartition_tau', type=float, nargs='+', default=[0.2])

    p.add_argument('--feat_res', type=int, default=256,
                   help='Square resolution fed to ResNet (stage grids = feat_res / stride).')
    p.add_argument('--out_res', type=int, default=256)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def setup_model(hf_id, device):
    processor = AutoImageProcessor.from_pretrained(hf_id)
    # We resize to feat_res ourselves; disable the processor's resize/crop.
    for attr in ('do_resize', 'do_center_crop', 'do_crop'):
        if hasattr(processor, attr):
            setattr(processor, attr, False)
    model = AutoModel.from_pretrained(hf_id).eval().to(device)
    return processor, model


@torch.no_grad()
def extract_features(model, processor, im_np, layer, feature_norm, device):
    """ResNet conv-stage feature map for one image -> ([num_patches, C] row-major, grid).

    layer indexes the list of 4D feature maps (conv stages); -1 = last stage.
    """
    inputs = processor(images=Image.fromarray(im_np), return_tensors='pt')
    pv = inputs['pixel_values'].to(device)
    out = model(pixel_values=pv, output_hidden_states=True, return_dict=True)
    fmaps = [t for t in out.hidden_states if t.ndim == 4]          # [B, C, Hf, Wf] per stage
    if not (-len(fmaps) <= layer < len(fmaps)):
        raise ValueError(f"--feature_layer {layer} out of range for {len(fmaps)} ResNet stages.")
    fmap = fmaps[layer]
    _, C, Hf, Wf = fmap.shape
    if Hf != Wf:
        raise RuntimeError(f"non-square feature map {Hf}x{Wf}; feed a square image.")
    feats = fmap.squeeze(0).permute(1, 2, 0).reshape(Hf * Wf, C).float()   # [num_patches, C], row-major
    if feature_norm == 'layernorm':
        feats = F.layer_norm(feats, (feats.shape[-1],))
    return feats, Hf


def main():
    args = parse_args()
    cfg = CONFIGS[args.model_variant]
    norm_tag = args.feature_norm
    slug = f"{cfg['slug']}-r{args.feat_res}"

    out_dirs = {}
    for tau in args.bipartition_tau:
        d = os.path.join(args.output_dir, f"{slug}_featurenn_layer{args.feature_layer}_{norm_tag}_tau{tau:g}")
        os.makedirs(d, exist_ok=True)
        out_dirs[tau] = d

    processor, model = setup_model(cfg['hf_id'], args.device)
    print(f"ResNet-50 ({cfg['hf_id']}): feat_res {args.feat_res}, stage(layer) {args.feature_layer}, "
          f"norm={norm_tag}, taus={[f'{t:g}' for t in args.bipartition_tau]}; out_res {args.out_res}")

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
                                           args.feature_norm, args.device)

            print(f"[{img_key}] {segments.shape[0]} segments, grid {grid}x{grid}, {len(pending)} tau(s)")
            for tau in pending:
                preds = np.stack([featurenn_segment(feats, c, grid, args.out_res, tau) for c in centroids])
                write_result_h5(os.path.join(out_dirs[tau], f"{img_key}.h5"), preds)
            print(f"[done] {img_key} -> tau {[f'{t:g}' for t in pending]}")


if __name__ == "__main__":
    main()
