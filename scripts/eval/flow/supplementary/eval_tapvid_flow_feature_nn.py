"""Feature-NN (cosine patch-feature matching) flow/depth eval for BabyZWM.

Matched, label-free probe. The representation baselines (DINOv3, V-JEPA2,
ResNet50) are read zero-shot by *nearest-neighbor cosine matching of frozen
patch features*: pass both frames through the frozen backbone, take a query
patch's feature in frame A, and find the argmax-cosine-similarity patch in
frame B; the query->match displacement is the predicted flow. This script
applies that SAME probe to BabyZWM's own frozen patch features, so every model
is read on identical footing and representation quality is isolated from
readout mechanism (the native perturb-compare-aggregate mechanism is NOT used
here).

This is the matched, label-free counterpart to zwm/eval/flow/eval_tapvid_flow.py
(BabyZWM's native mechanism). It reuses that file's data handling, iterative
zoom refinement, coordinate recovery, and output format verbatim, so the
existing graders apply unchanged:
  - flow:  python -m zwm.eval.flow.grade_tapvid_flow ...
  - depth: python -m zwm.eval.depth.grade_stereo_depth ...

Only the inner per-frame step differs: instead of perturb -> predict -> RGB
diff -> argmax, we extract a frozen patch-feature grid for each frame and take
the cosine-NN match. As with the baselines, iterative zoom raises the effective
resolution above the coarse patch grid (32x32 for BabyZWM-170M).

Two frame encodings (--encoding):
  - in_context (default): both frames are encoded in one two-frame forward (both
    fully visible), so they attend to each other — ZWM's native two-frame
    setting, where the model relates the frames.
  - independent: each frame is encoded alone, the direct analogue of a frozen
    DINOv3 / V-JEPA2 backbone (which has no two-frame mode).

The feature layer is configurable and recorded in the output path
(`..._featurenn_{enc}_layer{L}_{norm}`); the default is the middle layer (12 of
24). Sweep several layers in series with the .sh wrappers.

Run via scripts/eval/flow/supplementary/eval_tapvid_flow_feature_nn.sh (flow) or
eval_stereo_depth_feature_nn.sh (depth). Minimal direct invocation:

    python scripts/eval/flow/supplementary/eval_tapvid_flow_feature_nn.py \\
        --model_name awwkl/zwm-bvd-170m/model.pt \\
        --data_path data/evals/flow/tapvid_davis_first/dataset.json \\
        --feature_layer 12 --num_flat_points_to_process 10 --squish
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
from PIL import Image

from zwm.zwm_predictor import ZWMPredictor
from zwm.data.image_processing import patchify_image
from zwm.data.sequence_construction import get_pos_idxs
# Reuse the native flow eval's data + zoom machinery verbatim so the two evals
# stay on identical footing (same resize/crop, same zoom, same coord recovery).
from zwm.eval.flow.eval_tapvid_flow import (
    zoom_into_frame,
    recover_og_coordinates,
    resize,
    crop_and_rescale_points,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    # Core inputs
    parser.add_argument('--model_name', type=str, required=True,
                        help='ZWM model checkpoint relative to out/, e.g. awwkl/zwm-bvd-170m/model.pt')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to TapVID-style dataset.json with query/target point pairs '
                             '(flow: tapvid_davis_first; depth: stereo_depth).')
    parser.add_argument('--frames_root', type=str, default=None,
                        help='Directory holding the PNG frames referenced by dataset.json. '
                             'Defaults to <dirname(data_path)>/frames/.')
    parser.add_argument('--out_dir', type=str, default='viz/eval/flow/tapvid_davis_first_featurenn',
                        help='Where results, viz, and progress flags are written.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on.')

    # Feature-NN probe knobs
    parser.add_argument('--encoding', type=str, default='in_context', choices=['in_context', 'independent'],
                        help='How the two frames are encoded before matching. '
                             '"in_context" (default): both frames are passed through the model in a '
                             'single two-frame forward (both fully visible), so they attend to each '
                             'other — ZWM''s native two-frame setting, where the model relates the '
                             'frames. "independent": each frame is encoded alone (the direct analogue '
                             'of a frozen DINOv3 / V-JEPA2 backbone, which has no two-frame mode).')
    parser.add_argument('--feature_layer', type=int, default=12,
                        help='Which transformer block output to match on (0-indexed). '
                             'Default 12 = middle of the 24-layer BabyZWM-170M. '
                             '-1 = final layer after the final LayerNorm ln_f (the patch_head input).')
    parser.add_argument('--feature_norm', type=str, default='layernorm', choices=['none', 'layernorm'],
                        help='Normalization applied to an intermediate (layer>=0) feature before '
                             'cosine matching. ZWM is pre-norm, so raw block outputs are '
                             'un-normalized residual-stream activations; "layernorm" applies a '
                             'parameter-free LayerNorm over the feature dim (centers + scales) so '
                             'layers are comparable, matching V-JEPA2. Applied uniformly; for '
                             '--feature_layer -1 it is added on top of the final ln_f.')
    parser.add_argument('--occ_thresh', type=float, default=0.05,
                        help='Predict occluded when the best cosine similarity < occ_thresh. '
                             'Stored as occ_metric=max cosine; the grader can override.')

    # Zoom (resolution-boosting), matched to the baselines + native eval.
    parser.add_argument('--zoom_iters', type=int, default=4,
                        help='Number of refinement zoom iterations after the initial pass.')
    parser.add_argument('--zoom_stride', type=int, default=1,
                        help='Equivalent to N consecutive 25%% crops in one zoom step. '
                             'Requires --zoom_iters 1 if >1.')
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
# Frozen patch-feature extraction (the only model-specific piece).
# ------------------------------------------------------------

@torch.no_grad()
def extract_patch_features(predictor, frame_np, layer, feature_norm='layernorm'):
    """Run a single frame through the frozen ZWM transformer and return its
    patch-feature grid as [num_patches, n_embd], float32 on the model device.

    Each frame is encoded independently with all patches visible (mask all
    zeros) — the direct analogue of passing a frame through a frozen DINOv3 /
    V-JEPA2 backbone. Attention is bidirectional (causal_attention=False), so a
    zeros mask is a no-op for attention.

    layer: 0-indexed transformer block whose output to return; -1 returns the
    final block output after the model's ln_f (the representation that feeds
    patch_head).

    feature_norm: ZWM is pre-norm, so an intermediate block output is an
    un-normalized residual-stream activation. 'layernorm' applies a
    parameter-free LayerNorm over the feature dim before returning (matching
    V-JEPA2's readout), so layers are comparable; 'none' returns the raw output.
    Applied uniformly across layers; for layer -1 it is added on top of ln_f.
    """
    model = predictor.model
    num_patches = (model.config.resolution // model.config.patch_size) ** 2

    patches = _frame_to_patches(predictor, frame_np)
    seq = patches.reshape(1, num_patches, -1).to(predictor.device)
    pos = get_pos_idxs(patches, 0).reshape(1, -1).long().to(predictor.device)
    mask = torch.zeros((1, num_patches), device=predictor.device)

    n_blocks = len(model.transformer.h)
    target = layer if layer >= 0 else n_blocks - 1

    with predictor.ctx:
        # mask is all zeros, so the mask-token term vanishes; embed + position.
        h = model.transformer.token_embedding(seq) + model.transformer.positional_embedding(pos)
        h = model.transformer.drop(h)
        for i, block in enumerate(model.transformer.h):
            h = block(h, mask=mask)
            if i == target:
                break
        if layer < 0:
            h = model.transformer.ln_f(h)            # model's own final norm (final layer only)
        if feature_norm == 'layernorm':
            h = F.layer_norm(h, (h.shape[-1],))      # parameter-free centering — matches V-JEPA2

    return h.squeeze(0).float()  # [num_patches, n_embd]


def _frame_to_patches(predictor, frame_np):
    """PIL/np frame -> patchified tokens [1, num_patches, patch_size**2 * 3] on CPU."""
    resolution = predictor.model.config.resolution
    patch_size = predictor.model.config.patch_size
    pil = frame_np if isinstance(frame_np, Image.Image) else Image.fromarray(frame_np)
    pil = predictor.resize_crop_transform(pil, resolution)
    img = predictor.in_transform(pil).unsqueeze(0).permute(0, 2, 3, 1)
    return patchify_image(img, patch_size)


@torch.no_grad()
def extract_patch_features_in_context(predictor, frame0_np, frame1_np, layer, feature_norm='layernorm'):
    """Encode BOTH frames in one two-frame forward (both fully visible) and
    return their patch-feature grids (f0, f1), each [num_patches, n_embd].

    This is ZWM's native two-frame setting: the concatenated [frame0, frame1]
    sequence is processed with bidirectional attention, so the two frames attend
    to each other and each patch feature is informed by cross-frame
    correspondence. Both frames are fully visible (mask all zeros) so frame1
    carries real content to match against — unlike training, where frame1 is
    masked. (frame_gap is irrelevant for ZWM: positions are plain indices; only
    ZWM2 would consume it.)

    layer / feature_norm: as in extract_patch_features.
    """
    model = predictor.model
    resolution = model.config.resolution
    patch_size = model.config.patch_size
    num_patches = (resolution // patch_size) ** 2

    patches0 = _frame_to_patches(predictor, frame0_np)
    patches1 = _frame_to_patches(predictor, frame1_np)
    seq = torch.cat([patches0.reshape(-1, patches0.shape[-1]),
                     patches1.reshape(-1, patches1.shape[-1])]).unsqueeze(0).to(predictor.device)
    pos = torch.cat([get_pos_idxs(patches0, 0).reshape(-1),
                     get_pos_idxs(patches1, num_patches).reshape(-1)]).long().unsqueeze(0).to(predictor.device)
    mask = torch.zeros((1, 2 * num_patches), device=predictor.device)  # both frames visible

    n_blocks = len(model.transformer.h)
    target = layer if layer >= 0 else n_blocks - 1

    with predictor.ctx:
        h = model.transformer.token_embedding(seq) + model.transformer.positional_embedding(pos)
        h = model.transformer.drop(h)
        for i, block in enumerate(model.transformer.h):
            h = block(h, mask=mask)
            if i == target:
                break
        if layer < 0:
            h = model.transformer.ln_f(h)            # model's own final norm (final layer only)
        if feature_norm == 'layernorm':
            h = F.layer_norm(h, (h.shape[-1],))      # parameter-free centering — matches V-JEPA2

    h = h.squeeze(0).float()                          # [2 * num_patches, n_embd]
    return h[:num_patches], h[num_patches:]


def featurenn_match(query_feat_grid, target_feat_grid, query_x, query_y, grid, img_size):
    """Cosine-NN match a query point against the target feature grid.

    Returns (heatmap [grid, grid], pred_x, pred_y, max_cosine). The query and
    predictions are in img_size pixel space; predictions land at patch centers.
    """
    patch_px = img_size / grid
    qcol = min(max(int(query_x // patch_px), 0), grid - 1)
    qrow = min(max(int(query_y // patch_px), 0), grid - 1)
    query_feat = query_feat_grid[qrow * grid + qcol]  # [n_embd]

    cos = F.cosine_similarity(query_feat.unsqueeze(0), target_feat_grid, dim=-1)  # [num_patches]
    heatmap = cos.reshape(grid, grid).detach().cpu().numpy()

    idx = int(cos.argmax())
    prow, pcol = idx // grid, idx % grid
    pred_x = pcol * patch_px + patch_px / 2.0
    pred_y = prow * patch_px + patch_px / 2.0
    return heatmap, pred_x, pred_y, float(cos.max())


def viz_basic(out_dir, ground_truths, frame_curr, frame_next, heatmap, rgb_predictions):
    gt_qx, gt_qy, gt_tx, gt_ty = ground_truths
    px, py = rgb_predictions
    epe = np.sqrt((px - gt_tx) ** 2 + (py - gt_ty) ** 2)
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(frame_curr); axs[0].set_title("Frame 0 (r=pred, g=GT)")
    axs[0].arrow(gt_qx, gt_qy, gt_tx - gt_qx, gt_ty - gt_qy, head_width=3, head_length=3, fc='green', ec='green')
    axs[0].arrow(gt_qx, gt_qy, px - gt_qx, py - gt_qy, head_width=3, head_length=3, fc='red', ec='red')
    axs[1].imshow(frame_next); axs[1].set_title("Frame 1")
    axs[1].scatter(gt_tx, gt_ty, c='g', s=5); axs[1].scatter(px, py, c='r', s=5)
    axs[2].imshow(heatmap, cmap='viridis'); axs[2].set_title("Cosine-NN heatmap")
    fig.suptitle(f"EPE: {epe:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'final.png'))
    plt.close(fig)


# ------------------------------------------------------------
# Main eval loop (mirrors eval_tapvid_flow.main; perturb-predict swapped for
# feature-NN, and the per-step rollout loop dropped — feature-NN is deterministic).
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

    assert 'ZWM2' not in predictor.model.config.model_class, \
        "This feature-NN eval targets the ZWM (non-ZWM2) architecture."

    img_size = predictor.model.config.resolution
    patch_size = predictor.model.config.patch_size
    grid = img_size // patch_size
    n_blocks = predictor.model.config.n_layer
    layer_desc = 'final_lnf' if args.feature_layer < 0 else f'block{args.feature_layer}'
    # layer -1's base feature already carries the model's ln_f; feature_norm then
    # adds (or not) the parameter-free centering LayerNorm, uniformly across layers.
    norm_tag = args.feature_norm
    enc_tag = args.encoding.replace('_', '')  # incontext | independent
    assert args.feature_layer < n_blocks, \
        f"--feature_layer {args.feature_layer} out of range for a {n_blocks}-block model."
    logger.info(f"Feature-NN flow at {img_size}x{img_size}, {grid}x{grid} patch grid, "
                f"encoding={args.encoding}, matching on layer {args.feature_layer} "
                f"({layer_desc}) of {n_blocks}, norm={norm_tag}.")
    if 'full_tapvid' in args.data_path:
        assert args.squish, '--squish is required for full_tapvid datasets.'

    def should_viz(c):
        return False if args.no_viz else (args.viz_all or c % args.viz_interval == 0)

    def should_log(c):
        return c % args.log_interval == 0

    # Record the layer + norm in the output path so each run is self-describing
    # and runs with different settings never collide.
    model_slug = args.model_name.replace('/', '_').replace('.pt', '')
    out_dir = os.path.join(args.out_dir, f"{model_slug}_featurenn_{enc_tag}_layer{args.feature_layer}_{norm_tag}")
    os.makedirs(out_dir, exist_ok=True)
    viz_dir = os.path.join(out_dir, 'viz'); os.makedirs(viz_dir, exist_ok=True)
    flags_dir = os.path.join(out_dir, 'flags'); os.makedirs(flags_dir, exist_ok=True)
    args_dir = os.path.join(out_dir, 'args'); os.makedirs(args_dir, exist_ok=True)
    results_dir = os.path.join(out_dir, 'results'); os.makedirs(results_dir, exist_ok=True)

    data_range_str = f'[{args.flat_points_start_idx},{args.flat_points_start_idx + args.num_flat_points_to_process})'
    hostname = socket.gethostname()
    with open(os.path.join(args_dir, f'args_{data_range_str}.json'), 'w') as f:
        d = vars(args).copy(); d['hostname'] = hostname
        d['feature_layer_desc'] = layer_desc; d['norm_tag'] = norm_tag; d['enc_tag'] = enc_tag
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

        epe_logs[data_uid] = {}; pred_logs[data_uid] = {}

        # Resize frames + rescale points to img_size (mirrors eval_tapvid_flow).
        points = np.array([[[query_x_raster, query_y_raster], [target_x_raster, target_y_raster]]])
        query_frame_og, target_frame_og = query_frame, target_frame
        og_h, og_w = query_frame_og.shape[:2]

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
        points_og = (query_x_raster * og_w, query_y_raster * og_h,
                     target_x_raster * og_w, target_y_raster * og_h)
        if not args.squish:
            points_og = (float(query_x), float(query_y), float(target_x), float(target_y))

        f0_x_off, f0_y_off, f0_x_sc, f0_y_sc = [], [], [], []
        f1_x_off, f1_y_off, f1_x_sc, f1_y_sc = [], [], [], []
        occ_pred = False; occ_metric = 0.0
        rgb_pred_x = rgb_pred_y = None

        for zoom_itr in range(args.zoom_iters + 1):
            if args.encoding == 'in_context':
                f0_feat, f1_feat = extract_patch_features_in_context(
                    predictor, query_frame, target_frame, args.feature_layer, args.feature_norm)
            else:
                f0_feat = extract_patch_features(predictor, query_frame, args.feature_layer, args.feature_norm)
                f1_feat = extract_patch_features(predictor, target_frame, args.feature_layer, args.feature_norm)

            heatmap, rgb_pred_x, rgb_pred_y, occ_metric = featurenn_match(
                f0_feat, f1_feat, query_x, query_y, grid, img_size,
            )
            occ_pred = bool(occ_metric < args.occ_thresh)

            rgb_epe = np.sqrt((rgb_pred_x - target_x) ** 2 + (rgb_pred_y - target_y) ** 2)
            epe_logs[data_uid][zoom_itr] = {'multi_mask': rgb_epe}
            pred_logs[data_uid][zoom_itr] = {'multi_mask': (rgb_pred_x, rgb_pred_y)}

            if zoom_itr == args.zoom_iters:
                break

            # Set up the next zoom step (identical to eval_tapvid_flow).
            if zoom_itr == 0:
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
            data_viz_dir = os.path.join(viz_dir, data_uid); os.makedirs(data_viz_dir, exist_ok=True)
            viz_basic(data_viz_dir, (query_x, query_y, target_x, target_y),
                      query_frame_og, target_frame_og, heatmap, (rgb_pred_x, rgb_pred_y))

        counts += 1
        elapsed = time.time() - start
        avg_per_data = elapsed / counts
        remaining = (len(dataset) - counts) * avg_per_data
        now = datetime.now()
        eta = (now + timedelta(seconds=remaining)).strftime('%Y-%m-%d %H:%M:%S')
        with open(os.path.join(flags_dir, f'PROGRESS_{data_range_str}.txt'), 'w') as f:
            f.write(f"Logging @ {now.strftime('%Y-%m-%d %H:%M:%S')} ({hostname})\n"
                    f"- Encoding     : {args.encoding}\n"
                    f"- Feature layer: {args.feature_layer} ({layer_desc}), norm={norm_tag}\n"
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

    print(f"Mean feature-NN EPE (layer {args.feature_layer}): {rgb_batch_average / max(counts, 1):.3f}")


if __name__ == "__main__":
    main(get_args())
