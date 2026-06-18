"""Grade ONE Feature-NN cell dir (a single layer x tau) on SpelkeBench.

Cell files store only `segment_pred`; the GT segments (and the RGB image for
`--num_viz` overlays) are read from the source dataset. Reuses the shared metric
and the overlay plot from the native grader, so numbers match exactly. For the
whole layer x tau table at once, use grade_spelke_seg_feature_nn_table.py.

Usage:
    python scripts/eval/segments/supplementary/grade_spelke_seg_feature_nn.py \\
        --input_dir viz/.../awwkl_zwm-bvd-1b_model_featurenn_layer24_layernorm_tau0.4 \\
        --num_viz 10
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import shutil

import h5py
import numpy as np

from zwm.eval.segments.segment import evaluate_AP_AR_single_image
from zwm.eval.segments.grade_spelke_seg import _dataset_name_from_key, plot_segments_with_gt_overlay

from source_loaders import load_image, load_segments  # sibling module (sys.path[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True,
                   help='A single ..._layer{L}_{norm}_tau{T} cell dir of per-image .h5 files.')
    p.add_argument('--dataset_path', default='data/evals/segments/spelke_bench.h5',
                   help='Source h5 — GT segments + RGB images are read from here.')
    p.add_argument('--num_viz', type=int, default=0, help='Random images to overlay (0 disables).')
    p.add_argument('--viz_dir', default=None, help='Defaults to <input_dir>/_viz/.')
    p.add_argument('--viz_seed', type=int, default=42)
    args = p.parse_args()

    h5_files = sorted(glob.glob(os.path.join(args.input_dir, '*.h5')))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files in {args.input_dir}")

    buckets = collections.defaultdict(lambda: {'AP': [], 'AR': [], 'IoU': []})
    with h5py.File(args.dataset_path, 'r') as src:
        for fn in h5_files:
            key = os.path.splitext(os.path.basename(fn))[0]
            with h5py.File(fn, 'r') as f:
                seg_pred = f['segment_pred'][:] > 0
            seg_gt = load_segments(src, key, seg_pred.shape[-1]) > 0
            r = evaluate_AP_AR_single_image(seg_pred, seg_gt)
            b = buckets[_dataset_name_from_key(key)]
            b['AP'].append(r['AP']); b['AR'].append(r['AR'])
            b['IoU'].append(float(np.mean(r['iou_mat'].max(-1))))

        header = f"{'dataset':<14}{'graded':>9}{'AP':>9}{'AR':>9}{'IoU':>9}"
        print(header); print('-' * len(header))
        allAP, allAR, allIoU = [], [], []
        for name in sorted(buckets):
            b = buckets[name]
            print(f"{name:<14}{len(b['AP']):>9}"
                  f"{np.mean(b['AP']):>9.4f}{np.mean(b['AR']):>9.4f}{np.mean(b['IoU']):>9.4f}")
            allAP += b['AP']; allAR += b['AR']; allIoU += b['IoU']
        print('-' * len(header))
        print(f"{'overall':<14}{len(allAP):>9}"
              f"{np.mean(allAP):>9.4f}{np.mean(allAR):>9.4f}{np.mean(allIoU):>9.4f}")

        if args.num_viz > 0:
            viz_dir = args.viz_dir or os.path.join(args.input_dir, '_viz')
            shutil.rmtree(viz_dir, ignore_errors=True)
            os.makedirs(viz_dir, exist_ok=True)
            rng = np.random.default_rng(args.viz_seed)
            sel = rng.choice(len(h5_files), min(args.num_viz, len(h5_files)), replace=False)
            for idx in sel:
                fn = h5_files[idx]
                key = os.path.splitext(os.path.basename(fn))[0]
                with h5py.File(fn, 'r') as f:
                    seg_pred = f['segment_pred'][:]
                img = load_image(src, key, seg_pred.shape[-1])
                seg_gt = load_segments(src, key, seg_pred.shape[-1])
                plot_segments_with_gt_overlay(img, seg_pred, seg_gt, os.path.join(viz_dir, f"{key}.png"))
            print(f"Wrote {len(sel)} viz PNGs to {viz_dir}")


if __name__ == '__main__':
    main()
