"""Grade intuitive-physics rollouts.

For each (category, video_id, seed) in the eval output dir, loads the
predicted PNG and computes:

    mse_correct   = MSE(pred, frame3 | frame3_overall_mask) < MSE(pred, frame2 | frame2_overall_mask)
    lpips_correct = LPIPS(pred, frame3) < LPIPS(pred, frame2)

Aggregates per category and prints a table:

    category            n    mse_acc  lpips_acc
    1.cohesion          160   0.987   0.943
    ...
    overall             800   0.929   0.873
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import PIL.Image
import torch
import torchvision

import lpips


CATEGORY_ORDER = [
    '1.cohesion', '2.support_top', '3.support_bottom',
    '4.force_transfer', '5.force_separation',
]

# Matches notebook's `in_transform_without_normalize`.
IMG_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize(512),
    torchvision.transforms.CenterCrop(512),
    torchvision.transforms.ToTensor(),
])

PRED_RE = re.compile(r'^(?P<key>.+)_seed(?P<seed>\d+)\.png$')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_dir', type=str, required=True,
                   help='Directory of predicted PNGs (the worker\'s pred/ output).')
    p.add_argument('--dataset_dir', type=str, default='data/evals/intuitive_physics')
    p.add_argument('--out_csv', type=str, default=None,
                   help='Optional path to also write the per-category table as CSV.')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def load_image_tensor(path: str) -> torch.Tensor:
    return IMG_TRANSFORM(PIL.Image.open(path).convert('RGB'))


def masked_mse(pred: torch.Tensor, gt: torch.Tensor, mask: np.ndarray | None) -> float:
    """Notebook-compatible MSE: sum of squared differences / number of mask elements.
    With no mask, divides by total tensor numel (matches notebook)."""
    diff_sq = (pred - gt) ** 2
    if mask is None:
        denom = float(diff_sq.numel())
        return float(diff_sq.sum().item()) / denom
    mask_t = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)
    denom = float(mask_t.sum().item())
    if denom == 0:
        return float('nan')
    mask_t = mask_t.expand_as(diff_sq)
    return float((diff_sq * mask_t).sum().item()) / denom


def main():
    args = parse_args()

    annotations_df = pd.read_csv(os.path.join(args.dataset_dir, 'annotations.csv'), dtype=str)
    annotations_df['key'] = annotations_df['category'] + '_' + annotations_df['video_id']
    key_to_category = dict(zip(annotations_df['key'], annotations_df['category']))

    pred_paths = sorted(glob.glob(os.path.join(args.pred_dir, '*.png')))
    if not pred_paths:
        raise FileNotFoundError(f'No prediction PNGs under {args.pred_dir!r}')

    lpips_model = lpips.LPIPS(net='alex').to(args.device)
    keyframes_dir = os.path.join(args.dataset_dir, 'keyframes')
    masks_dir = os.path.join(args.dataset_dir, 'segment_masks')

    # Cache loaded GT tensors/masks per item key.
    gt_cache: dict[str, dict] = {}

    def get_gt(item_key: str) -> dict:
        if item_key in gt_cache:
            return gt_cache[item_key]
        category = key_to_category[item_key]
        video_id = item_key[len(category) + 1:]
        item_dir = os.path.join(keyframes_dir, category, video_id)
        mask_dir = os.path.join(masks_dir, category, video_id)
        gt = {
            'frame2': load_image_tensor(os.path.join(item_dir, 'frame_02.png')),
            'frame3': load_image_tensor(os.path.join(item_dir, 'frame_03.png')),
            'frame2_overall_mask': np.load(os.path.join(mask_dir, 'frame2_overall_mask.npy'))[0],
            'frame3_overall_mask': np.load(os.path.join(mask_dir, 'frame3_overall_mask.npy'))[0],
        }
        gt_cache[item_key] = gt
        return gt

    rows = []
    for pred_path in pred_paths:
        m = PRED_RE.match(os.path.basename(pred_path))
        if not m:
            print(f'[skip] unparseable filename: {pred_path}')
            continue
        item_key = m.group('key')
        seed = int(m.group('seed'))
        category = key_to_category.get(item_key)
        if category is None:
            print(f'[skip] {item_key} not in annotations.csv')
            continue

        gt = get_gt(item_key)
        pred = load_image_tensor(pred_path)

        mse_p_f3 = masked_mse(pred, gt['frame3'], gt['frame3_overall_mask'])
        mse_p_f2 = masked_mse(pred, gt['frame2'], gt['frame2_overall_mask'])

        # LPIPS expects tensors in [-1, 1].
        pred_d = pred.to(args.device).unsqueeze(0) * 2 - 1
        f3_d = gt['frame3'].to(args.device).unsqueeze(0) * 2 - 1
        f2_d = gt['frame2'].to(args.device).unsqueeze(0) * 2 - 1
        with torch.no_grad():
            lpips_p_f3 = float(lpips_model(pred_d, f3_d).item())
            lpips_p_f2 = float(lpips_model(pred_d, f2_d).item())

        rows.append({
            'category': category,
            'item_key': item_key,
            'seed': seed,
            'mse_correct': int(mse_p_f3 < mse_p_f2),
            'lpips_correct': int(lpips_p_f3 < lpips_p_f2),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('No prediction files were successfully graded.')

    # Per-category aggregation.
    print()
    header = f"{'category':<20}{'n':>6}{'mse_acc':>10}{'lpips_acc':>11}"
    print(header)
    print('-' * len(header))
    table_rows = []
    for cat in CATEGORY_ORDER:
        sub = df[df['category'] == cat]
        if sub.empty:
            continue
        n = len(sub)
        mse_acc = float(sub['mse_correct'].mean())
        lpips_acc = float(sub['lpips_correct'].mean())
        print(f"{cat:<20}{n:>6}{mse_acc:>10.4f}{lpips_acc:>11.4f}")
        table_rows.append({'category': cat, 'n': n, 'mse_acc': mse_acc, 'lpips_acc': lpips_acc})
    print('-' * len(header))
    n_all = len(df)
    mse_acc_all = float(df['mse_correct'].mean())
    lpips_acc_all = float(df['lpips_correct'].mean())
    print(f"{'overall':<20}{n_all:>6}{mse_acc_all:>10.4f}{lpips_acc_all:>11.4f}")
    table_rows.append({'category': 'overall', 'n': n_all, 'mse_acc': mse_acc_all, 'lpips_acc': lpips_acc_all})

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
        pd.DataFrame(table_rows).to_csv(args.out_csv, index=False, float_format='%.4f')
        print(f'\nWrote {args.out_csv}')


if __name__ == '__main__':
    main()
