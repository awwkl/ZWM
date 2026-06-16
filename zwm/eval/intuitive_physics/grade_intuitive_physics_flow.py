"""Grade intuitive-physics rollouts with a FLOW-based motion-fidelity metric.

Companion to `grade_intuitive_physics.py` (which scores pixel MSE / LPIPS
closer-to-target-than-context). This script measures whether the predicted
*object motion* matches the real motion, per object region.

For each (category, video_id):
    gt_flow   = optical_flow(frame2, frame3)          # the real motion
For each predicted PNG (category, video_id, seed):
    pred_flow = optical_flow(frame2, prediction)      # the predicted motion

Within each region mask (primary object / secondary object / overall / no-mask,
all on the frame2 grid) we report:
    epe       = mean || pred_flow - gt_flow ||                       (pixels)
    percent   = 100 * epe / mean||gt_flow||         (normalized EPE; clipped 100)
    graded    = d_ctx / (d_ctx + d_tgt)             continuous closeness in [0,1]
                where d_tgt = epe (distance to real motion) and
                      d_ctx = mean||pred_flow|| (distance to no-motion/context).
                graded = 1 -> predicted motion equals the real motion;
                graded = 0 -> predicted motion equals "no motion" (the context).
                (Meaningful where there IS motion; degenerate static regions -> NaN.)

Accuracy per region = fraction of examples with percent <= --threshold (default 50).

Optical flow uses the same estimator as the SpelkeBench segments eval
(`zwm.eval.segments.segment_zoom.compute_flow`, ptlflow DPFlow), so no new
dependency or precomputed ground-truth flow is required: both gt and predicted
flow are computed on the fly from the existing keyframes.

Example:
    python -m zwm.eval.intuitive_physics.grade_intuitive_physics_flow \
        --pred_dir viz/eval/intuitive_physics/seeds8_gap10/awwkl_zwm-babyview-170m_model/pred \
        --dataset_dir data/evals/intuitive_physics \
        --out_csv viz/eval/intuitive_physics/seeds8_gap10/awwkl_zwm-babyview-170m_model/flow_accuracy.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import PIL.Image

from zwm.eval.segments.segment_zoom import compute_flow


CATEGORY_ORDER = [
    '1.cohesion', '2.support_top', '3.support_bottom',
    '4.force_transfer', '5.force_separation',
]

REGIONS = ['primary', 'secondary', 'overall']

PRED_RE = re.compile(r'^(?P<key>.+)_seed(?P<seed>\d+)\.png$')

RES = 512  # masks are 512x512; flow is computed at this resolution to match.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_dir', type=str, required=True,
                   help="Directory of predicted PNGs (the worker's pred/ output).")
    p.add_argument('--dataset_dir', type=str, default='data/evals/intuitive_physics')
    p.add_argument('--out_csv', type=str, default=None,
                   help='Optional path to write the per-category table as CSV.')
    p.add_argument('--rows_csv', type=str, default=None,
                   help='Optional path to write per-(item, seed, region) raw rows.')
    p.add_argument('--threshold', type=float, default=50.0,
                   help='Normalized-EPE percent threshold for accuracy (default 50).')
    p.add_argument('--device', type=str,
                   default='cuda' if _cuda_available() else 'cpu')
    return p.parse_args()


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def load_rgb_512(path: str) -> PIL.Image.Image:
    return PIL.Image.open(path).convert('RGB').resize((RES, RES), PIL.Image.BILINEAR)


def load_mask(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    m = np.load(path)
    if m.ndim == 3:  # stored as (1, H, W)
        m = m[0]
    return m.astype(np.float32)


def flow_np(img1, img2, device: str) -> np.ndarray:
    """(H, W, 2) numpy optical flow between two PIL images."""
    return compute_flow(img1, img2, device=device).detach().cpu().numpy()


def region_metrics(pred_flow: np.ndarray, gt_flow: np.ndarray,
                   mask: np.ndarray | None) -> dict:
    """Per-region EPE, normalized percent error, and graded closeness."""
    epe_map = np.linalg.norm(pred_flow - gt_flow, axis=-1)   # (H, W)
    gtmag_map = np.linalg.norm(gt_flow, axis=-1)
    predmag_map = np.linalg.norm(pred_flow, axis=-1)

    sel = np.ones(epe_map.shape, dtype=bool) if mask is None else mask.astype(bool)
    if sel.sum() == 0:
        return {'epe': np.nan, 'percent': np.nan, 'graded': np.nan}

    epe = float(epe_map[sel].mean())
    gtmag = float(gtmag_map[sel].mean())
    predmag = float(predmag_map[sel].mean())

    percent = (epe / gtmag * 100.0) if gtmag > 1e-6 else np.nan
    denom = predmag + epe
    graded = (predmag / denom) if denom > 1e-6 else np.nan
    return {'epe': epe, 'percent': percent, 'graded': graded}


def main():
    args = parse_args()

    annotations_df = pd.read_csv(os.path.join(args.dataset_dir, 'annotations.csv'), dtype=str)
    annotations_df['key'] = annotations_df['category'] + '_' + annotations_df['video_id']
    key_to_category = dict(zip(annotations_df['key'], annotations_df['category']))

    pred_paths = sorted(glob.glob(os.path.join(args.pred_dir, '*.png')))
    if not pred_paths:
        raise FileNotFoundError(f'No prediction PNGs under {args.pred_dir!r}')

    keyframes_dir = os.path.join(args.dataset_dir, 'keyframes')
    masks_dir = os.path.join(args.dataset_dir, 'segment_masks')

    # Cache per item key: gt flow + frame2 region masks + the frame2 PIL.
    gt_cache: dict[str, dict] = {}

    def get_gt(item_key: str) -> dict:
        if item_key in gt_cache:
            return gt_cache[item_key]
        category = key_to_category[item_key]
        video_id = item_key[len(category) + 1:]
        item_dir = os.path.join(keyframes_dir, category, video_id)
        mask_dir = os.path.join(masks_dir, category, video_id)

        frame2 = load_rgb_512(os.path.join(item_dir, 'frame_02.png'))
        frame3 = load_rgb_512(os.path.join(item_dir, 'frame_03.png'))

        masks = {
            r: load_mask(os.path.join(mask_dir, f'frame2_{r}_mask.npy'))
            for r in REGIONS
        }
        # A missing region mask (e.g. `secondary` for single-object categories)
        # denotes an EMPTY region, not the whole image: substitute a zero mask so
        # its metrics come out NaN (and are excluded), matching the reference
        # notebook's `np.zeros_like(primary_mask)` behavior.
        ref_shape = next((m.shape for m in masks.values() if m is not None), None)
        if ref_shape is not None:
            for r in REGIONS:
                if masks[r] is None:
                    masks[r] = np.zeros(ref_shape, dtype=np.float32)

        gt = {
            'frame2': frame2,
            'gt_flow': flow_np(frame2, frame3, args.device),
            'masks': masks,
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
        pred = load_rgb_512(pred_path)
        pred_flow = flow_np(gt['frame2'], pred, args.device)

        row = {'category': category, 'item_key': item_key, 'seed': seed}
        for r in REGIONS:
            mr = region_metrics(pred_flow, gt['gt_flow'], gt['masks'][r])
            row[f'{r}_epe'] = mr['epe']
            row[f'{r}_percent'] = mr['percent']
            row[f'{r}_graded'] = mr['graded']
        rows.append(row)

        if (i + 1) % 100 == 0:
            print(f'  graded {i + 1}/{len(pred_paths)} predictions...')

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('No prediction files were successfully graded.')

    def agg(sub: pd.DataFrame, region: str) -> dict:
        epe = sub[f'{region}_epe']
        pct = sub[f'{region}_percent'].clip(upper=100)
        graded = sub[f'{region}_graded']
        valid = pct.notna()
        acc = float((pct[valid] <= args.threshold).mean()) if valid.any() else float('nan')
        # pandas .mean() skips NaN and returns NaN for an all-NaN column without
        # the numpy "Mean of empty slice" warning.
        return {
            f'{region}_epe': float(epe.mean()),
            f'{region}_acc': acc,
            f'{region}_graded': float(graded.mean()),
        }

    # Per-category + overall table.
    print()
    header = (f"{'category':<20}{'n':>6}"
              f"{'prim_EPE':>10}{'sec_EPE':>10}{'ovr_EPE':>10}"
              f"{'prim_acc':>10}{'sec_acc':>10}{'ovr_acc':>10}"
              f"{'prim_grd':>10}{'sec_grd':>10}{'ovr_grd':>10}")
    print(header)
    print('-' * len(header))

    table_rows = []

    def print_row(label: str, sub: pd.DataFrame):
        a = {}
        for r in REGIONS:
            a.update(agg(sub, r))
        print(f"{label:<20}{len(sub):>6}"
              f"{a['primary_epe']:>10.3f}{a['secondary_epe']:>10.3f}{a['overall_epe']:>10.3f}"
              f"{a['primary_acc']:>10.4f}{a['secondary_acc']:>10.4f}{a['overall_acc']:>10.4f}"
              f"{a['primary_graded']:>10.4f}{a['secondary_graded']:>10.4f}{a['overall_graded']:>10.4f}")
        table_rows.append({'category': label, 'n': len(sub), **a})

    for cat in CATEGORY_ORDER:
        sub = df[df['category'] == cat]
        if not sub.empty:
            print_row(cat, sub)
    print('-' * len(header))
    print_row('overall', df)

    if args.rows_csv:
        os.makedirs(os.path.dirname(args.rows_csv) or '.', exist_ok=True)
        df.to_csv(args.rows_csv, index=False, float_format='%.4f')
        print(f'\nWrote per-row metrics to {args.rows_csv}')

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
        pd.DataFrame(table_rows).to_csv(args.out_csv, index=False, float_format='%.4f')
        print(f'Wrote per-category table to {args.out_csv}')


if __name__ == '__main__':
    main()
