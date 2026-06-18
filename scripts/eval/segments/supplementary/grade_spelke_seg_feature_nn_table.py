"""Tabulate Feature-NN SpelkeBench results over a layer x bipartition-tau sweep.

Discovers every per-(layer, tau) output dir written by eval_spelke_seg_feature_nn.py
for a given model + feature_norm, grades each (mean AP / AR / IoU over its images),
and prints a layer x tau matrix for the chosen metric. Grading reuses the shared
metric (zwm.eval.segments.segment.evaluate_AP_AR_single_image), so the numbers match
grade_spelke_seg exactly. Safe to run mid-sweep — partial cells are flagged.

Usage:
    python scripts/eval/segments/supplementary/grade_spelke_seg_feature_nn_table.py \\
        --model_name awwkl/zwm-bvd-1b/model.pt --metric AP
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import h5py
import numpy as np

from zwm.eval.segments.segment import evaluate_AP_AR_single_image

from zwm.eval.segments.feature_nn import load_segments  # source-dataset GT loader


def grade_dir(d, src_h5, gt_cache):
    """Mean AP / AR / IoU over the *.h5 in one cell dir. Cell files store only
    segment_pred; GT segments are read from the source dataset (cached per image,
    since GT is shared across every cell). Returns (n, AP, AR, IoU)."""
    aps, ars, ious = [], [], []
    for fn in sorted(glob.glob(os.path.join(d, '*.h5'))):
        img_key = os.path.splitext(os.path.basename(fn))[0]
        try:
            with h5py.File(fn, 'r') as f:
                seg_pred = f['segment_pred'][:] > 0
        except (OSError, KeyError):
            continue
        if img_key not in gt_cache:
            gt_cache[img_key] = load_segments(src_h5, img_key, seg_pred.shape[-1]) > 0
        r = evaluate_AP_AR_single_image(seg_pred, gt_cache[img_key])
        aps.append(r['AP']); ars.append(r['AR']); ious.append(float(np.mean(r['iou_mat'].max(-1))))
    if not aps:
        return 0, float('nan'), float('nan'), float('nan')
    return len(aps), float(np.mean(aps)), float(np.mean(ars)), float(np.mean(ious))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_root', default='viz/eval/segments/spelke_bench_featurenn')
    p.add_argument('--model_name', default='awwkl/zwm-bvd-1b/model.pt',
                   help='Used to derive the dir slug (same rule as the eval worker).')
    p.add_argument('--feature_norm', default='layernorm', choices=['none', 'layernorm'])
    p.add_argument('--dataset_path', default='data/evals/segments/spelke_bench.h5',
                   help='Source h5 — GT segments are read from here (cells store only segment_pred).')
    p.add_argument('--metric', default='AP', choices=['AP', 'AR', 'IoU'])
    args = p.parse_args()

    slug = args.model_name.replace('/', '_').replace('.pt', '')
    pat = re.compile(rf'^{re.escape(slug)}_featurenn_layer(-?\d+)_{re.escape(args.feature_norm)}_tau([0-9.]+)$')

    cells = {}  # (layer, tau) -> (n, AP, AR, IoU)
    layers, taus = set(), set()
    gt_cache = {}  # img_key -> GT, loaded once from source and reused across all cells
    with h5py.File(args.dataset_path, 'r') as src:
        for d in sorted(glob.glob(os.path.join(args.output_root, '*'))):
            m = pat.match(os.path.basename(d))
            if not (m and os.path.isdir(d)):
                continue
            layer, tau = int(m.group(1)), float(m.group(2))
            cells[(layer, tau)] = grade_dir(d, src, gt_cache)
            layers.add(layer); taus.add(tau)

    if not cells:
        print(f"No dirs matching {slug}_featurenn_layer*_{args.feature_norm}_tau* under {args.output_root}")
        return

    layers = sorted(layers, key=lambda L: (L < 0, L))  # '-1' (final post-ln_f) sorts last
    taus = sorted(taus)
    midx = {'AP': 1, 'AR': 2, 'IoU': 3}[args.metric]

    print(f"\nmodel: {slug}   norm: {args.feature_norm}   metric: {args.metric}")
    print("(rows = feature layer, cols = bipartition tau; 'final' = after ln_f)\n")
    header = f"{'layer':>7}" + "".join(f"{t:>9g}" for t in taus)
    print(header)
    print('-' * len(header))
    for L in layers:
        label = 'final' if L < 0 else str(L)
        row = f"{label:>7}"
        for t in taus:
            cell = cells.get((L, t))
            row += f"{cell[midx]:>9.4f}" if (cell and cell[0]) else f"{'-':>9}"
        print(row)

    counts = {k: v[0] for k, v in cells.items()}
    n_max = max(counts.values()) if counts else 0
    partial = {k: v for k, v in counts.items() if v != n_max}
    if partial:
        print(f"\nNote: {len(partial)} cell(s) below {n_max} images graded (sweep still running?):")
        for (L, t), n in sorted(partial.items(), key=lambda kv: (kv[0][0] < 0, kv[0][0], kv[0][1])):
            print(f"  layer {'final' if L < 0 else L}, tau {t:g}: {n}/{n_max}")
    else:
        print(f"\nAll {len(cells)} cells graded on {n_max} images.")

    best = max((k for k in cells if cells[k][0]), key=lambda k: cells[k][midx], default=None)
    if best is not None:
        L, t = best
        print(f"Best {args.metric}: layer {'final' if L < 0 else L}, tau {t:g} = {cells[best][midx]:.4f}")


if __name__ == '__main__':
    main()
