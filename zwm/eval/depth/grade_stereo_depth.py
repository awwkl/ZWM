"""Grade stereo-depth eval from eval_tapvid_flow.py predictions.

The depth task reuses the flow EPE engine: each entry of dataset.json is one
(query, target) stereo half-pair, with two entries per image (`_point0`,
`_point1`). The engine writes a per-uid EPE to
`<root_dir>/results/epe_results_*.json`. This script joins those EPEs to the
dataset.json (for `dataset_name` and ground-truth `depth`), pairs `_point0`
with `_point1` by uid, and reports per-dataset accuracy of depth ordering.

A pair is "correct" iff EPE ordering matches GT depth ordering: the closer
point (smaller depth) should yield a larger counterfactual EPE.

Output: per-dataset N, accuracy, bootstrap 95% CI, binomial SEM, mean ratio.

Usage:
    python -m zwm.eval.depth.grade_stereo_depth \\
        --root_dir viz/eval/depth/stereo_depth/std_2_zoom_4/awwkl_zwm-babyview-170m_model_mask_ratio_0.9 \\
        --data_path data/evals/depth/stereo_depth/dataset.json
"""
import json
import math
import os
from argparse import ArgumentParser
from collections import defaultdict
from glob import glob

import numpy as np


def compute_bootstrap_95_ci(scores, n_boot, rng):
    scores = np.asarray(scores, dtype=np.float64)
    mean = float(scores.mean())
    boot_means = np.empty(n_boot, dtype=np.float64)
    n = len(scores)
    for i in range(n_boot):
        boot_means[i] = scores[rng.integers(0, n, size=n)].mean()
    lower = float(np.percentile(boot_means, 2.5))
    upper = float(np.percentile(boot_means, 97.5))
    return mean, lower, upper


def load_epe_shards(root_dir):
    pattern = os.path.join(root_dir, '**', 'epe_results_*.json')
    shard_paths = sorted(glob(pattern, recursive=True))
    assert shard_paths, f"No epe_results_*.json found under {root_dir}"
    print(f'Found {len(shard_paths)} epe shard file(s). Sample: {shard_paths[0]}')
    merged = {}
    for path in shard_paths:
        with open(path, 'r') as f:
            shard = json.load(f)
        overlap = set(merged) & set(shard)
        assert not overlap, f"Duplicate uids across shards: {sorted(overlap)[:5]}..."
        merged.update(shard)
    return merged


def build_uid_to_meta(data_path):
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    return {entry['uid']: {'dataset_name': entry['dataset_name'], 'depth': entry['depth']}
            for entry in dataset}


def pair_and_score(uid_to_epe, uid_to_meta):
    per_dataset = defaultdict(lambda: {'correct': [], 'ratio': []})
    missing_pairs = 0
    seen_pairs = 0

    for uid0 in sorted(uid_to_meta):
        if not uid0.endswith('_point0'):
            continue
        uid1 = uid0[: -len('_point0')] + '_point1'

        if uid1 not in uid_to_meta:
            missing_pairs += 1
            continue
        if uid0 not in uid_to_epe or uid1 not in uid_to_epe:
            missing_pairs += 1
            continue

        meta0 = uid_to_meta[uid0]
        meta1 = uid_to_meta[uid1]
        assert meta0['dataset_name'] == meta1['dataset_name'], (
            f"Pair {uid0!r}/{uid1!r} has mismatched dataset_name "
            f"({meta0['dataset_name']!r} vs {meta1['dataset_name']!r})")

        depth0, depth1 = meta0['depth'], meta1['depth']
        epe0 = float(uid_to_epe[uid0]['final'])
        epe1 = float(uid_to_epe[uid1]['final'])

        # Closer point (smaller depth) should produce larger EPE.
        correct = (depth0 < depth1) == (epe0 > epe1)

        if epe0 > 0:
            pred_depth1 = depth0 * (epe0 / epe1) if epe1 > 0 else 0.0
            if pred_depth1 > 0 and depth1 > 0:
                ratio = min(pred_depth1, depth1) / max(pred_depth1, depth1)
            else:
                ratio = 0.0
        else:
            ratio = 0.0

        per_dataset[meta0['dataset_name']]['correct'].append(bool(correct))
        per_dataset[meta0['dataset_name']]['ratio'].append(float(ratio))
        seen_pairs += 1

    if missing_pairs:
        print(f'WARNING: {missing_pairs} pair(s) skipped due to missing _point0/_point1 EPE.')
    return dict(per_dataset), seen_pairs, missing_pairs


def report(per_dataset_results, n_boot, rng):
    pretty = {'kitti_1000': 'KITTI 1000', 'kitti_500_flipud': 'KITTI 500 Flipud'}
    total = 0
    for ds_name in sorted(per_dataset_results):
        correct = per_dataset_results[ds_name]['correct']
        ratios = per_dataset_results[ds_name]['ratio']
        n = len(correct)
        total += n
        acc_mean, acc_lo, acc_hi = compute_bootstrap_95_ci(
            [100.0 * c for c in correct], n_boot=n_boot, rng=rng)
        p = sum(correct) / n
        sem = 100.0 * math.sqrt(p * (1 - p) / n) if n > 0 else float('nan')
        ratio_mean = float(np.mean(ratios)) if ratios else float('nan')
        label = pretty.get(ds_name, ds_name)
        print(f'{label} (N={n}) Accuracy 95% CI: ({acc_mean:.1f}, {acc_lo:.1f}, {acc_hi:.1f})  '
              f'SEM: {sem:.2f}  RatioMean: {ratio_mean:.3f}')
    return total


def main():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Eval out dir holding results/epe_results_*.json shards.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the depth dataset.json (with dataset_name + depth GT).')
    parser.add_argument('--n_boot', type=int, default=10000,
                        help='Bootstrap resamples for the 95%% CI.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for the bootstrap RNG (deterministic CIs).')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    uid_to_epe = load_epe_shards(args.root_dir)
    uid_to_meta = build_uid_to_meta(args.data_path)
    per_dataset, seen, missing = pair_and_score(uid_to_epe, uid_to_meta)

    total = report(per_dataset, n_boot=args.n_boot, rng=rng)
    expected = sum(1 for u in uid_to_meta if u.endswith('_point0'))
    print(f'\nTotal pairs evaluated: {total}/{expected}'
          + (f' ({missing} missing)' if missing else ''))


if __name__ == '__main__':
    main()
