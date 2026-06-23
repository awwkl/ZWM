"""Grade intuitive-physics rollouts (pixel MSE / LPIPS).

For each (category, video_id, seed) in the eval output dir, loads the predicted PNG
and, per object region (primary = manipulated object, secondary = other object,
overall = their union), computes against the two keyframes:

    d_ctx = MSE(pred, frame2 | frame2_{region}_mask)   # distance to the static "context"
    d_tgt = MSE(pred, frame3 | frame3_{region}_mask)    # distance to the correct outcome

    acc    = d_tgt < d_ctx                  binary: prediction closer to the correct outcome
    graded = d_ctx / (d_ctx + d_tgt)        continuous closeness in [0,1] (higher is better):
                                            1 -> matches the correct outcome (frame3),
                                            0 -> predicts no change (frame2).

A missing region mask (e.g. `secondary` for single-object categories) denotes an EMPTY
region, so its metrics come out NaN and are excluded. LPIPS is whole-image (no mask), so
only its binary accuracy is reported (a whole-image graded score is dominated by the static
background and is uninformative).

Aggregates per category and prints a table:

    category            n    primary_acc  primary_graded  ...  overall_graded  lpips_acc
    1.cohesion          160  ...
    overall             800  ...
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
REGIONS = ['primary', 'secondary', 'overall']

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
    p.add_argument('--rows_csv', type=str, default=None,
                   help='Optional path to write the per-prediction rows as CSV.')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def load_image_tensor(path: str) -> torch.Tensor:
    return IMG_TRANSFORM(PIL.Image.open(path).convert('RGB'))


def load_mask(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    m = np.load(path)
    if m.ndim == 3:  # stored as (1, H, W)
        m = m[0]
    return m.astype(np.float32)


def masked_mse(pred: torch.Tensor, gt: torch.Tensor, mask: np.ndarray | None) -> float:
    """Notebook-compatible MSE: sum of squared differences / number of mask elements.
    With no mask, divides by total tensor numel. An empty mask returns NaN."""
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


def region_scores(d_ctx: float, d_tgt: float) -> tuple[float, float]:
    """(binary acc, graded closeness) from context/target distances; NaN if either is NaN."""
    if not (np.isfinite(d_ctx) and np.isfinite(d_tgt)):
        return float('nan'), float('nan')
    denom = d_ctx + d_tgt
    graded = (d_ctx / denom) if denom > 1e-12 else float('nan')
    return float(d_tgt < d_ctx), graded


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

        f2_masks = {r: load_mask(os.path.join(mask_dir, f'frame2_{r}_mask.npy')) for r in REGIONS}
        f3_masks = {r: load_mask(os.path.join(mask_dir, f'frame3_{r}_mask.npy')) for r in REGIONS}
        # A missing region mask denotes an EMPTY region (not the whole image): substitute a
        # zero mask so its metrics come out NaN (and are excluded).
        for masks in (f2_masks, f3_masks):
            ref = next((m.shape for m in masks.values() if m is not None), None)
            if ref is not None:
                for r in REGIONS:
                    if masks[r] is None:
                        masks[r] = np.zeros(ref, dtype=np.float32)

        gt = {
            'frame2': load_image_tensor(os.path.join(item_dir, 'frame_02.png')),
            'frame3': load_image_tensor(os.path.join(item_dir, 'frame_03.png')),
            'f2_masks': f2_masks,
            'f3_masks': f3_masks,
        }
        gt_cache[item_key] = gt
        return gt

    rows = []
    for i, pred_path in enumerate(pred_paths):
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

        # LPIPS expects tensors in [-1, 1]; whole-image (no mask).
        pred_d = pred.to(args.device).unsqueeze(0) * 2 - 1
        f3_d = gt['frame3'].to(args.device).unsqueeze(0) * 2 - 1
        f2_d = gt['frame2'].to(args.device).unsqueeze(0) * 2 - 1
        with torch.no_grad():
            lpips_p_f3 = float(lpips_model(pred_d, f3_d).item())
            lpips_p_f2 = float(lpips_model(pred_d, f2_d).item())

        # Whole-image MSE (no mask) — the fairest like-for-like region vs the
        # V-JEPA2 baseline's whole-image cosine, since it privileges no segment.
        # Background-inflated, so use for the binary closer-to-frame3 metric;
        # the object-region (primary) graded stays the informative score.
        d_tgt_whole = masked_mse(pred, gt['frame3'], None)
        d_ctx_whole = masked_mse(pred, gt['frame2'], None)
        whole_acc, whole_graded = region_scores(d_ctx_whole, d_tgt_whole)

        row = {
            'category': category, 'item_key': item_key, 'seed': seed,
            'lpips_correct': int(lpips_p_f3 < lpips_p_f2),
            'whole_acc': whole_acc, 'whole_graded': whole_graded,
        }
        for r in REGIONS:
            d_tgt = masked_mse(pred, gt['frame3'], gt['f3_masks'][r])
            d_ctx = masked_mse(pred, gt['frame2'], gt['f2_masks'][r])
            acc, graded = region_scores(d_ctx, d_tgt)
            row[f'{r}_acc'] = acc
            row[f'{r}_graded'] = graded
        rows.append(row)

        if (i + 1) % 100 == 0:
            print(f'  graded {i + 1}/{len(pred_paths)} predictions...')

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('No prediction files were successfully graded.')

    # Per-category aggregation (pandas .mean() skips NaN -> empty regions excluded).
    cols = [f'{r}_{m}' for r in REGIONS for m in ('acc', 'graded')]
    def agg(sub):
        out = {'n': len(sub), 'lpips_acc': float(sub['lpips_correct'].mean()),
               'whole_acc': float(sub['whole_acc'].mean()),
               'whole_graded': float(sub['whole_graded'].mean())}
        for c in cols:
            out[c] = float(sub[c].mean())
        return out

    table_rows = []
    for cat in CATEGORY_ORDER:
        sub = df[df['category'] == cat]
        if sub.empty:
            continue
        table_rows.append({'category': cat, **agg(sub)})
    table_rows.append({'category': 'overall', **agg(df)})
    table = pd.DataFrame(table_rows)

    show = ['category', 'n', 'primary_acc', 'primary_graded',
            'secondary_acc', 'secondary_graded', 'overall_acc', 'overall_graded',
            'whole_acc', 'whole_graded', 'lpips_acc']
    pd.set_option('display.width', 200)
    print('\n' + table[show].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    if args.rows_csv:
        os.makedirs(os.path.dirname(args.rows_csv) or '.', exist_ok=True)
        df.to_csv(args.rows_csv, index=False, float_format='%.6f')
        print(f'\nWrote {args.rows_csv}')
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
        table.to_csv(args.out_csv, index=False, float_format='%.4f')
        print(f'Wrote {args.out_csv}')


if __name__ == '__main__':
    main()
