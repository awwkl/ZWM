"""Grade SpelkeBench segmentation rollouts.

Globs `*.h5` under `--input_dir` (the output of `eval_spelke_seg`), runs the
per-image AP/AR/IoU metric on each, and prints the mean. Optionally writes
per-image visualizations for a random subset.
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import re
import shutil

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours

from zwm.eval.segments.segment import evaluate_AP_AR_single_image


def _dataset_name_from_key(key: str) -> str:
    """Bucket name from an image key, e.g. 'entityseg_1_image1007' -> 'entityseg'."""
    m = re.match(r'^[a-zA-Z]+', key)
    return m.group(0) if m else 'other'


def plot_segments_with_gt_overlay(rgb_image, pred_segments, gt_segments, save_path):
    """3 rows × N segments: image, GT mask, pred mask with red GT contour."""
    N = pred_segments.shape[0]
    fig, axs = plt.subplots(3, N, figsize=(4 * N, 10), squeeze=False)

    for i in range(N):
        axs[0, i].imshow(rgb_image)
        axs[0, i].axis('off')
        axs[0, i].set_title(f"image (seg {i})")

        axs[1, i].imshow(gt_segments[i], cmap='gray')
        axs[1, i].axis('off')
        axs[1, i].set_title("GT")

        axs[2, i].imshow(pred_segments[i], cmap='gray')
        for contour in find_contours(gt_segments[i].astype(float), 0.5):
            axs[2, i].plot(contour[:, 1], contour[:, 0], color='red', linewidth=1.5)
        axs[2, i].axis('off')
        axs[2, i].set_title("pred + GT contour")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def evaluate_directory(input_dir, dataset_path=None, num_viz=0, viz_dir=None, viz_seed=42):
    h5_files = sorted(glob.glob(os.path.join(input_dir, '*.h5')))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files in {input_dir}")

    # Per-bucket accumulators: {dataset_name: {'AP': [...], 'AR': [...], 'IoU': [...]}}
    buckets: dict = collections.defaultdict(lambda: {'AP': [], 'AR': [], 'IoU': []})

    for fn in h5_files:
        with h5py.File(fn, 'r') as f:
            seg_pred = f['segment_pred'][:] > 0
            seg_gt = f['segment_gt'][:] > 0

        result = evaluate_AP_AR_single_image(seg_pred, seg_gt)
        key = os.path.splitext(os.path.basename(fn))[0]
        b = buckets[_dataset_name_from_key(key)]
        b['AP'].append(result['AP'])
        b['AR'].append(result['AR'])
        b['IoU'].append(np.mean(result['iou_mat'].max(-1)))

    # Expected counts per bucket, if the source dataset is available.
    expected = {}
    if dataset_path and os.path.exists(dataset_path):
        with h5py.File(dataset_path, 'r') as f:
            expected = collections.Counter(_dataset_name_from_key(k) for k in f.keys())

    # Per-bucket report.
    header = f"{'dataset':<14}{'graded':>9}{'expected':>10}{'AP':>9}{'AR':>9}{'IoU':>9}"
    print(header)
    print('-' * len(header))
    total_graded = total_expected = 0
    for name in sorted(buckets):
        b = buckets[name]
        n = len(b['AP'])
        exp_str = str(expected.get(name, '?')) if expected else '-'
        print(f"{name:<14}{n:>9}{exp_str:>10}"
              f"{np.mean(b['AP']):>9.4f}{np.mean(b['AR']):>9.4f}{np.mean(b['IoU']):>9.4f}")
        total_graded += n
        if expected:
            total_expected += expected.get(name, 0)

    # Overall (micro-averaged across all graded images, ignoring bucket).
    all_AP = [v for b in buckets.values() for v in b['AP']]
    all_AR = [v for b in buckets.values() for v in b['AR']]
    all_IOU = [v for b in buckets.values() for v in b['IoU']]
    print('-' * len(header))
    overall_exp = str(total_expected) if expected else '-'
    print(f"{'overall':<14}{total_graded:>9}{overall_exp:>10}"
          f"{np.mean(all_AP):>9.4f}{np.mean(all_AR):>9.4f}{np.mean(all_IOU):>9.4f}")

    if num_viz <= 0:
        return

    viz_dir = viz_dir or os.path.join(input_dir, '_viz')
    # Wipe so the dir always reflects only the current run's selection (stale
    # PNGs from a different --num_viz or --viz_seed would otherwise linger).
    shutil.rmtree(viz_dir, ignore_errors=True)
    os.makedirs(viz_dir, exist_ok=True)

    rng = np.random.default_rng(viz_seed)
    n = min(num_viz, len(h5_files))
    selected = rng.choice(len(h5_files), n, replace=False)

    for idx in selected:
        fn = h5_files[idx]
        with h5py.File(fn, 'r') as f:
            img = f['image'][:]
            seg_pred = f['segment_pred'][:]
            seg_gt = f['segment_gt'][:]

        base = os.path.splitext(os.path.basename(fn))[0]
        out_path = os.path.join(viz_dir, f"{base}.png")
        plot_segments_with_gt_overlay(img, seg_pred, seg_gt, out_path)

    print(f"Wrote {n} viz PNGs to {viz_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', type=str, required=True,
                   help='Directory of per-image .h5 files from eval_spelke_seg.')
    p.add_argument('--dataset_path', type=str, default='data/evals/segments/spelke_bench.h5',
                   help='Source dataset h5; used to report expected counts per bucket. '
                        'Pass an empty string to skip the expected-count column.')
    p.add_argument('--num_viz', type=int, default=0,
                   help='Number of random images to visualize. 0 disables viz.')
    p.add_argument('--viz_dir', type=str, default=None,
                   help='Where to write viz PNGs. Defaults to <input_dir>/_viz/.')
    p.add_argument('--viz_seed', type=int, default=42,
                   help='Seed for the random subset selection.')
    args = p.parse_args()
    evaluate_directory(args.input_dir, dataset_path=args.dataset_path or None,
                       num_viz=args.num_viz,
                       viz_dir=args.viz_dir, viz_seed=args.viz_seed)


if __name__ == "__main__":
    main()
