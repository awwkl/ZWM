"""Eval ZWM on TapVID-DAVIS via counterfactual point-tracking (optical flow).

For each query/target point pair from the TapVID-DAVIS annotations, perturb
the query frame around the query point, predict frame1 with and without the
perturbation, then locate where the perturbation "moved" via the L2 norm of
the RGB-prediction difference. Iterative zoom refines the prediction.
End-Point Error (EPE) against the ground-truth target point is the metric.

Ported from the internal `inv_flow_final.py` script — the CWM (== ZWM) code
path only; CCWM, MaskPSI, and PSI branches removed.

Typical invocation lives in scripts/eval/flow/eval_tapvid_flow.sh; minimal
direct invocation:

    python -m zwm.eval.flow.eval_tapvid_flow \\
        --model_name awwkl/zwm-babyview-170m/model.pt \\
        --data_path data/evals/flow/tapvid_davis_first/dataset.json \\
        --num_flat_points_to_process 10 --viz_all
"""
import argparse
import json
import logging
import os
import socket
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image, ImageOps

from zwm.zwm_predictor import ZWMPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    # Core inputs
    parser.add_argument('--model_name', type=str, required=True,
                        help='ZWM model checkpoint relative to out/, e.g. awwkl/zwm-babyview-170m/model.pt')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to TapVID-style dataset.json with query/target point pairs.')
    parser.add_argument('--frames_root', type=str, default=None,
                        help='Directory holding the extracted PNG frames referenced by dataset.json. '
                             'Defaults to <dirname(data_path)>/frames/.')
    parser.add_argument('--out_dir', type=str, default='viz/eval/flow/tapvid_davis_first',
                        help='Where results, viz, and progress flags are written.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on.')
    parser.add_argument('--seed', type=int, default=1110, help='Base random seed.')

    # Counterfactual flow knobs
    parser.add_argument('--num_rollouts', type=int, default=5,
                        help='Number of seeded mask rollouts per zoom step.')
    parser.add_argument('--mask_ratio', type=float, default=0.9,
                        help='Mask ratio for frame1 patches (default 0.9 matches training).')
    parser.add_argument('--frame_gap', type=int, default=-1,
                        help='Frame gap to pass to ZWMPredictor. -1 enables dynamic policy: '
                             'use the actual gap clamped to [5, 15].')
    parser.add_argument('--perturb_std', type=float, default=2.0,
                        help='Gaussian std for the query perturbation (in pixels at model resolution).')
    parser.add_argument('--perturb_magnifier', type=float, default=1.0,
                        help='Multiply perturb_std by this every zoom iteration.')
    parser.add_argument('--amplitude', type=int, default=255,
                        help='Pixel amplitude of the perturbation, 0-255.')
    parser.add_argument('--zoom_iters', type=int, default=4,
                        help='Number of refinement zoom iterations after the initial pass.')
    parser.add_argument('--zoom_stride', type=int, default=1,
                        help='Equivalent to N consecutive 25%% crops in one zoom step. '
                             'Requires --zoom_iters 1 if >1.')
    parser.add_argument('--no_blur', action='store_true',
                        help='Skip the 8x8 box blur over the averaged diff map. The shell wrapper sets this.')
    parser.add_argument('--squish', action='store_true',
                        help='Squish-resize frames to img_size instead of square center-cropping. '
                             'Required for full_tapvid datasets; the shell wrapper sets this.')

    # Sharding / iteration
    parser.add_argument('--flat_points_start_idx', type=int, default=0,
                        help='First entry of dataset.json to evaluate.')
    parser.add_argument('--num_flat_points_to_process', type=int, default=200,
                        help='How many entries from start_idx to evaluate.')

    # Viz / logging
    parser.add_argument('--viz_all', action='store_true', help='Visualize every datum.')
    parser.add_argument('--viz_interval', type=int, default=1000)
    parser.add_argument('--no_viz', action='store_true', help='Disable all viz.')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Flush JSON results every N data points.')

    parser.add_argument('--compile', action='store_true', help='torch.compile the model.')

    return parser.parse_args()


# ------------------------------------------------------------
# Helpers (inlined from inv_flow_final.py; lift to eval/utils/
# only when a second eval port actually reuses one).
# ------------------------------------------------------------

def perturb_image(image, x, y, std=3, color=(255, 255, 255)):
    """Add a 2D Gaussian bump to an RGB image at (x, y). Clamps to [0, 255]."""
    height, width, _ = image.shape
    xv, yv = np.meshgrid(np.arange(width), np.arange(height))
    gaussian = np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * std ** 2))
    perturb = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        perturb[..., c] = gaussian * color[c]
    new_image = image.astype(np.float32) + perturb
    return np.clip(new_image, 0, 255).astype(np.uint8)


def zoom_into_frame(frame, center_x, center_y, reduce_pct=0.25, zoom_stride=1, rect=False, img_size=256):
    """Crop a (1-reduce_pct)**zoom_stride window around (center_x, center_y), upsample to img_size."""
    h, w, _ = frame.shape
    keep = (1 - reduce_pct) ** zoom_stride
    rh, rw = int(h * keep), int(w * keep)
    if not rect:
        assert h == w, f"square crop required (got {frame.shape}); pass rect=True for first-iter rectangles"
    else:
        rh = rw = min(h, w)
    left = int(max(center_x - rw // 2, 0))
    right = min(left + rw, w); left = right - rw
    top = int(max(center_y - rh // 2, 0))
    bottom = min(top + rh, h); top = bottom - rh
    frame = frame[top:bottom, left:right]
    assert frame.shape[:-1] == (rh, rw), f"expected ({rh},{rw}) after crop, got {frame.shape[:-1]}"
    frame = np.array(Image.fromarray(frame).resize((img_size, img_size)))
    return frame, left, top, img_size / rw, img_size / rh


def recover_og_coordinates(gt_query_x, gt_query_y, gt_target_x, gt_target_y,
                           pred_x, pred_y,
                           f0_x_scales, f0_x_offsets, f0_y_scales, f0_y_offsets,
                           f1_x_scales, f1_x_offsets, f1_y_scales, f1_y_offsets):
    """Walk the zoom-transform stack backwards to map points back to the pre-zoom frame."""
    for i in range(len(f0_x_scales) - 1, -1, -1):
        gt_query_x = gt_query_x / f0_x_scales[i] + f0_x_offsets[i]
        gt_query_y = gt_query_y / f0_y_scales[i] + f0_y_offsets[i]
        gt_target_x = gt_target_x / f1_x_scales[i] + f1_x_offsets[i]
        gt_target_y = gt_target_y / f1_y_scales[i] + f1_y_offsets[i]
        pred_x = pred_x / f1_x_scales[i] + f1_x_offsets[i]
        pred_y = pred_y / f1_y_scales[i] + f1_y_offsets[i]
    return gt_query_x, gt_query_y, gt_target_x, gt_target_y, pred_x, pred_y


def resize(np_img, patch_size=8, fixed_size=None, smart=False):
    """Resize to a square; if smart, center-crop-fit instead of squishing aspect ratio."""
    h, w, _ = np_img.shape
    rh, rw = (h // patch_size) * patch_size, (w // patch_size) * patch_size
    size = min(rh, rw)
    if fixed_size is not None:
        size = fixed_size
        if h == w == size:
            return np_img
    if smart:
        img = ImageOps.fit(Image.fromarray(np_img), (size, size))
    else:
        img = Image.fromarray(np_img).resize((size, size))
    return np.array(img)


def crop_and_rescale_points(points_xy_raster, og_size):
    """Center-crop the raster to a square, rescale points back into [0, 1]."""
    og_h, og_w = og_size
    points_xy = points_xy_raster * [[[og_w, og_h]]]
    points_xy[..., 0] = points_xy[..., 0] - (og_w - og_h) // 2
    return points_xy / og_h


def get_pred_and_epe(heatmap, gt_x, gt_y, img_size=256):
    """Argmax of heatmap, scaled to img_size, plus EPE against (gt_x, gt_y)."""
    y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
    img_size_y, img_size_x = (img_size, img_size) if not isinstance(img_size, tuple) else img_size
    y = (img_size_y / heatmap.shape[0]) * y
    x = (img_size_x / heatmap.shape[1]) * x
    return x, y, np.sqrt((x - gt_x) ** 2 + (y - gt_y) ** 2)


# ------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------

def viz_rollout(out_dir, out_name, ground_truths, frame_curr, frame_curr_perturbed,
                frame_next, perturbed_dict, unperturbed_dict, diff_map, rgb_predictions):
    gt_qx, gt_qy, gt_tx, gt_ty = ground_truths
    px, py = rgb_predictions
    epe = np.sqrt((px - gt_tx) ** 2 + (py - gt_ty) ** 2)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes = axes.flatten()
    axes[0].imshow(frame_curr); axes[0].set_title("Frame 0 (r=pred, g=GT)")
    axes[0].arrow(gt_qx, gt_qy, gt_tx - gt_qx, gt_ty - gt_qy, head_width=3, head_length=3, fc='green', ec='green')
    axes[0].arrow(gt_qx, gt_qy, px - gt_qx, py - gt_qy, head_width=3, head_length=3, fc='red', ec='red')
    axes[1].imshow(frame_curr_perturbed); axes[1].set_title("Perturbed Frame 0")
    axes[1].add_patch(plt.Circle((gt_qx, gt_qy), radius=15, edgecolor='red', facecolor='none', linewidth=2))
    axes[2].imshow(frame_next); axes[2].set_title("Frame 1")
    axes[2].scatter(gt_tx, gt_ty, c='g', s=5); axes[2].scatter(px, py, c='r', s=5)
    axes[3].imshow(perturbed_dict["frame1_pred_pil"]); axes[3].set_title("Recon Pert Frame 1")
    axes[4].imshow(unperturbed_dict["frame1_pred_pil"]); axes[4].set_title("Recon Clean Frame 1")
    axes[5].imshow(diff_map, cmap='viridis'); axes[5].set_title("Diff Map")
    fig.suptitle(f"EPE: {epe:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, out_name))
    plt.close(fig)


def viz_multimask(out_dir, ground_truths, frame_curr, frame_next, avg_diff_map, rgb_predictions):
    gt_qx, gt_qy, gt_tx, gt_ty = ground_truths
    px, py = rgb_predictions
    epe = np.sqrt((px - gt_tx) ** 2 + (py - gt_ty) ** 2)
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(frame_curr); axs[0].set_title("Frame 0 (r=pred, g=GT)")
    axs[0].arrow(gt_qx, gt_qy, gt_tx - gt_qx, gt_ty - gt_qy, head_width=3, head_length=3, fc='green', ec='green')
    axs[0].arrow(gt_qx, gt_qy, px - gt_qx, py - gt_qy, head_width=3, head_length=3, fc='red', ec='red')
    axs[1].imshow(frame_next); axs[1].set_title("Frame 1")
    axs[1].scatter(gt_tx, gt_ty, c='g', s=5); axs[1].scatter(px, py, c='r', s=5)
    axs[2].imshow(avg_diff_map, cmap='hot'); axs[2].set_title("Avg Diff Map")
    axs[2].add_patch(plt.Circle((gt_tx, gt_ty), radius=15, edgecolor='green', facecolor='none', linewidth=2, linestyle='--'))
    axs[2].add_patch(plt.Circle((px, py), radius=15, edgecolor='red', facecolor='none', linewidth=2, linestyle='--'))
    fig.suptitle(f"EPE: {epe:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'averaged_preds.png'))
    plt.close(fig)


def viz_basic(out_dir, ground_truths, frame_curr, frame_next, rgb_predictions):
    gt_qx, gt_qy, gt_tx, gt_ty = ground_truths
    px, py = rgb_predictions
    epe = np.sqrt((px - gt_tx) ** 2 + (py - gt_ty) ** 2)
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(frame_curr); axs[0].set_title("Frame 0 (r=pred, g=GT)")
    axs[1].imshow(frame_next); axs[1].set_title("Frame 1")
    axs[0].arrow(gt_qx, gt_qy, gt_tx - gt_qx, gt_ty - gt_qy, head_width=3, head_length=3, fc='green', ec='green')
    axs[0].arrow(gt_qx, gt_qy, px - gt_qx, py - gt_qy, head_width=3, head_length=3, fc='red', ec='red')
    axs[1].scatter(gt_tx, gt_ty, c='g', s=5); axs[1].scatter(px, py, c='r', s=5)
    fig.suptitle(f"EPE: {epe:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'final.png'))
    plt.close(fig)


# ------------------------------------------------------------
# Main eval loop
# ------------------------------------------------------------

def main(args):
    if args.zoom_stride > 1:
        assert args.zoom_iters == 1, "zoom_stride > 1 requires zoom_iters == 1 (single big zoom)."

    if args.frames_root is None:
        args.frames_root = os.path.join(os.path.dirname(args.data_path), 'frames')

    with open(args.data_path, 'r') as f:
        dataset = json.load(f)

    predictor = ZWMPredictor(model_name=args.model_name, device=args.device)
    if args.compile:
        print("Compiling the model")
        predictor.model = torch.compile(predictor.model)

    img_size = predictor.model.config.resolution
    logger.info(f"Running at resolution {img_size}x{img_size}")

    if "ZWM2" in predictor.model.config.model_class:
        assert args.frame_gap is not None, "ZWM2 requires explicit --frame_gap (cannot infer from RoPE)."
    if 'full_tapvid' in args.data_path:
        assert args.squish, '--squish is required for full_tapvid datasets.'

    def predict(frame0, frame1, seed, frame_gap):
        return predictor.factual_prediction(
            frame0=frame0, frame1=frame1,
            frame_gap=frame_gap, mask_ratio=args.mask_ratio, seed=seed,
        )

    def should_viz(c):
        return False if args.no_viz else (args.viz_all or c % args.viz_interval == 0)

    def should_log(c):
        return c % args.log_interval == 0

    out_dir = os.path.join(args.out_dir, args.model_name.replace('/', '_').replace('.pt', ''))
    out_dir = f"{out_dir}_mask_ratio_{args.mask_ratio}"
    os.makedirs(out_dir, exist_ok=True)
    viz_dir = os.path.join(out_dir, 'viz'); os.makedirs(viz_dir, exist_ok=True)
    flags_dir = os.path.join(out_dir, 'flags'); os.makedirs(flags_dir, exist_ok=True)
    args_dir = os.path.join(out_dir, 'args'); os.makedirs(args_dir, exist_ok=True)
    results_dir = os.path.join(out_dir, 'results'); os.makedirs(results_dir, exist_ok=True)

    data_range_str = f'[{args.flat_points_start_idx},{args.flat_points_start_idx + args.num_flat_points_to_process})'
    hostname = socket.gethostname()
    with open(os.path.join(args_dir, f'args_{data_range_str}.json'), 'w') as f:
        d = vars(args).copy(); d['hostname'] = hostname
        json.dump(d, f, indent=4)

    dataset = dataset[args.flat_points_start_idx:
                      args.flat_points_start_idx + args.num_flat_points_to_process]
    print(f'Evaluating on {len(dataset)} points.')

    epe_logs, pred_logs, tapvid_formatted_results = {}, {}, []
    rgb_batch_average = 0; counts = 0; start = time.time()

    for data in tqdm.tqdm(dataset, total=len(dataset)):
        query_frame = np.array(Image.open(os.path.join(args.frames_root, data['query_frame_file'])))
        target_frame = np.array(Image.open(os.path.join(args.frames_root, data['target_frame_file'])))

        query_x_raster, query_y_raster = data['query_x_raster'], data['query_y_raster']
        target_x_raster, target_y_raster = data['target_x_raster'], data['target_y_raster']
        data_uid = data['uid']

        # Frame-gap policy: -1 → dynamic clamp to [5, 15], else fixed.
        start_uid, end_uid, *_ = data_uid.split(',')
        st = int(start_uid.split('_')[-1]); et = int(end_uid.split('_')[-1])
        if args.frame_gap < 0:
            frame_gap = min(max(et - st, 5), 15); policy = "dynamic_with_cap"
        else:
            frame_gap = args.frame_gap; policy = "fixed"
        if counts == 0:
            print(f"Data Frame Gap: {et - st} => {frame_gap} (policy={policy})")

        epe_logs[data_uid] = {}; pred_logs[data_uid] = {}

        # Resize frames + rescale points to img_size.
        points = np.array([[[query_x_raster, query_y_raster], [target_x_raster, target_y_raster]]])
        query_frame_og, target_frame_og = query_frame, target_frame
        og_h, og_w = query_frame_og.shape[:2]
        points_og = (query_x_raster * og_w, query_y_raster * og_h,
                     target_x_raster * og_w, target_y_raster * og_h)

        if not args.squish:
            points = crop_and_rescale_points(points, query_frame.shape[:2]) * img_size
        else:
            points = points * img_size
        query_x, query_y = points[0, 0]
        target_x, target_y = points[0, 1]

        query_frame = resize(query_frame, fixed_size=img_size, smart=not args.squish)
        target_frame = resize(target_frame, fixed_size=img_size, smart=not args.squish)

        if not args.squish:
            query_frame_og, target_frame_og = query_frame, target_frame
            og_h, og_w = query_frame_og.shape[:2]
            points_og = (float(query_x), float(query_y), float(target_x), float(target_y))

        f0_x_off, f0_y_off, f0_x_sc, f0_y_sc = [], [], [], []
        f1_x_off, f1_y_off, f1_x_sc, f1_y_sc = [], [], [], []
        occ_pred = False; occ_metric = 0.0
        rgb_pred_x = rgb_pred_y = None
        zoom_dir = None  # set inside the rollout loop when viz fires

        for zoom_itr in range(args.zoom_iters + 1):
            epe_logs[data_uid][zoom_itr] = {'iters': []}
            pred_logs[data_uid][zoom_itr] = {'iters': []}
            seed_diff_maps, prenorm_diff_maps = [], []

            for rollout_idx in range(args.num_rollouts):
                current_seed = args.seed + rollout_idx
                query_frame_perturbed = perturb_image(
                    query_frame, x=query_x, y=query_y,
                    std=args.perturb_std * (args.perturb_magnifier ** zoom_itr),
                    color=(args.amplitude,) * 3,
                )
                perturbed_dict = predict(query_frame_perturbed, target_frame, current_seed, frame_gap)
                unperturbed_dict = predict(query_frame, target_frame, current_seed, frame_gap)

                perturbed_pred = perturbed_dict['frame1_pred_rgb'][0].cpu().numpy()
                unperturbed_pred = unperturbed_dict['frame1_pred_rgb'][0].cpu().numpy()

                prenorm_diff_maps.append(perturbed_pred - unperturbed_pred)
                diff_map = np.linalg.norm(perturbed_pred - unperturbed_pred, axis=0)

                itr_x, itr_y, itr_epe = get_pred_and_epe(diff_map, target_x, target_y, img_size=img_size)
                epe_logs[data_uid][zoom_itr]['iters'].append(itr_epe)
                pred_logs[data_uid][zoom_itr]['iters'].append((itr_x, itr_y))
                seed_diff_maps.append(diff_map)

                if should_viz(counts):
                    data_viz_dir = os.path.join(viz_dir, data_uid)
                    zoom_dir = os.path.join(data_viz_dir, f'zoom={zoom_itr}')
                    os.makedirs(zoom_dir, exist_ok=True)
                    viz_rollout(zoom_dir, f'rollout_{rollout_idx:03d}.png',
                                (query_x, query_y, target_x, target_y),
                                query_frame, query_frame_perturbed, target_frame,
                                perturbed_dict, unperturbed_dict, diff_map, (itr_x, itr_y))

            # Occlusion signal: per-rollout L1-norm peak averaged across seeds.
            prenorm_arr = np.stack(prenorm_diff_maps, 0)  # (m, c, h, w)
            l1_reduced = np.linalg.norm(prenorm_arr, ord=1, axis=1)
            occ_metric = float(np.amax(l1_reduced, axis=(1, 2)).mean())
            occ_pred = bool(occ_metric < 0.05)

            avg_diff_map = np.mean(seed_diff_maps, axis=0)
            final_diff_map = avg_diff_map
            if not args.no_blur:
                t = torch.tensor(avg_diff_map).unsqueeze(0).unsqueeze(0).float()
                t = F.pad(t, (3, 4, 3, 4), mode='reflect')
                final_diff_map = F.avg_pool2d(t, kernel_size=8, stride=1).squeeze().numpy()

            rgb_pred_x, rgb_pred_y, rgb_epe = get_pred_and_epe(
                final_diff_map, target_x, target_y, img_size=img_size,
            )
            epe_logs[data_uid][zoom_itr]['multi_mask'] = rgb_epe
            pred_logs[data_uid][zoom_itr]['multi_mask'] = (rgb_pred_x, rgb_pred_y)

            if should_viz(counts) and zoom_dir is not None:
                viz_multimask(zoom_dir, (query_x, query_y, target_x, target_y),
                              query_frame, target_frame, final_diff_map, (rgb_pred_x, rgb_pred_y))

            if zoom_itr == args.zoom_iters:
                break

            # Set up the next zoom step.
            if zoom_itr == 0:
                # Move from img_size space back into the original-resolution frame.
                h_scale = query_frame_og.shape[0] / query_frame.shape[0]
                w_scale = query_frame_og.shape[1] / query_frame.shape[1]
                query_frame, target_frame = query_frame_og, target_frame_og
                query_x, query_y = query_x * w_scale, query_y * h_scale
                rgb_pred_x, rgb_pred_y = rgb_pred_x * w_scale, rgb_pred_y * h_scale
                target_x, target_y = target_x * w_scale, target_y * h_scale

            is_rect = query_frame.shape[0] != query_frame.shape[1]
            if is_rect:
                assert zoom_itr == 0, "Zooming should only produce squares after the first step."

            query_frame, cl, ct, cws, chs = zoom_into_frame(
                query_frame, query_x, query_y, zoom_stride=args.zoom_stride, rect=is_rect, img_size=img_size,
            )
            target_frame, nl, nt, nws, nhs = zoom_into_frame(
                target_frame, rgb_pred_x, rgb_pred_y, zoom_stride=args.zoom_stride, rect=is_rect, img_size=img_size,
            )
            query_x = (query_x - cl) * cws; query_y = (query_y - ct) * chs
            target_x = (target_x - nl) * nws; target_y = (target_y - nt) * nhs
            f0_x_off.append(cl); f0_x_sc.append(cws); f0_y_off.append(ct); f0_y_sc.append(chs)
            f1_x_off.append(nl); f1_x_sc.append(nws); f1_y_off.append(nt); f1_y_sc.append(nhs)

        # Recover predictions in the pre-zoom (original) coordinate space.
        rgb_pred_x, rgb_pred_y = recover_og_coordinates(
            query_x, query_y, target_x, target_y, rgb_pred_x, rgb_pred_y,
            f0_x_sc, f0_x_off, f0_y_sc, f0_y_off,
            f1_x_sc, f1_x_off, f1_y_sc, f1_y_off,
        )[-2:]

        query_x, query_y, target_x, target_y = points_og
        rgb_epe = np.sqrt((target_x - rgb_pred_x) ** 2 + (target_y - rgb_pred_y) ** 2)
        rgb_batch_average += rgb_epe
        epe_logs[data_uid]['final'] = rgb_epe
        pred_logs[data_uid]['final'] = (rgb_pred_x, rgb_pred_y)

        tapvid_formatted_results.append({
            'uid': data['uid'],
            'gt_query_x': data['query_x_raster'] * 256,
            'gt_query_y': data['query_y_raster'] * 256,
            'gt_target_x': data['target_x_raster'] * 256,
            'gt_target_y': data['target_y_raster'] * 256,
            'gt_occ': data.get('occluded', False),
            'pred_target_x': rgb_pred_x * (256 / og_w),
            'pred_target_y': rgb_pred_y * (256 / og_h),
            'pred_occ': occ_pred,
            'occ_metric': occ_metric,
        })

        if should_log(counts):
            with open(os.path.join(results_dir, f'epe_results_{data_range_str}.json'), 'w') as f:
                json.dump(epe_logs, f, indent=4)
            with open(os.path.join(results_dir, f'pred_results_{data_range_str}.json'), 'w') as f:
                json.dump(pred_logs, f, indent=4)
            with open(os.path.join(results_dir, f'tapvid_formatted_results_256res_{data_range_str}.json'), 'w') as f:
                json.dump(tapvid_formatted_results, f, indent=4)

        if should_viz(counts):
            viz_basic(os.path.join(viz_dir, data_uid),
                      (query_x, query_y, target_x, target_y),
                      query_frame_og, target_frame_og, (rgb_pred_x, rgb_pred_y))

        counts += 1
        elapsed = time.time() - start
        avg_per_data = elapsed / counts
        remaining = (len(dataset) - counts) * avg_per_data
        now = datetime.now()
        eta = (now + timedelta(seconds=remaining)).strftime('%Y-%m-%d %H:%M:%S')
        with open(os.path.join(flags_dir, f'PROGRESS_{data_range_str}.txt'), 'w') as f:
            f.write(f"Logging @ {now.strftime('%Y-%m-%d %H:%M:%S')} ({hostname})\n"
                    f"- Progress     : {counts}/{len(dataset)} done\n"
                    f"- Total elapsed: {elapsed:.3f}s\n"
                    f"- Avg sec / itr: {avg_per_data:.3f}s\n"
                    f"- Expected done: {eta}\n")

    # Final flush
    with open(os.path.join(results_dir, f'epe_results_{data_range_str}.json'), 'w') as f:
        json.dump(epe_logs, f, indent=4)
    with open(os.path.join(results_dir, f'pred_results_{data_range_str}.json'), 'w') as f:
        json.dump(pred_logs, f, indent=4)
    with open(os.path.join(results_dir, f'tapvid_formatted_results_256res_{data_range_str}.json'), 'w') as f:
        json.dump(tapvid_formatted_results, f, indent=4)
    with open(os.path.join(flags_dir, f'FLAG_{data_range_str}.txt'), 'w') as f:
        f.write("Done!\n")

    print(f"Mean RGB EPE: {rgb_batch_average / max(counts, 1):.3f}")


if __name__ == "__main__":
    main(get_args())
