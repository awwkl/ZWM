"""Zoom-based RGB-diff segmentation, CWM path only.

Ported from ccwm/utils/segment_zoom.py. Compared to the source:
  - psi_rgb / psi_flow / psi_flow_100M branches removed
  - `counterfactual_prediction` calls renamed to `hypothetical_prediction`
    (the ZWM predictor method name)
  - flow-quantizer / token_to_flow_dict plumbing dropped (CWM doesn't need it)
  - dead-code helpers from the source kept off (get_segment, get_segment_gen2,
    get_segment_uncond, resize_flow_to_original, get_random_coords_outside_segment)
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from zwm.eval.segments.segment import offset_multiple_centroids, threshold_heatmap

# Fail fast at import time rather than producing a cryptic segfault inside
# compute_flow on the first inference (partial installs of ptlflow's native
# deps have been observed to crash at the C level instead of raising).
try:
    import ptlflow
    from ptlflow.utils.io_adapter import IOAdapter
except ImportError as e:
    raise ImportError(
        "ptlflow is required for the SpelkeBench segments eval. "
        "Install with `pip install ptlflow`."
    ) from e

# One SEA-RAFT model per device per process — `compute_flow` is called once
# per (seed, dir, segment) and reconstructing it each time is wasteful and
# can exhaust GPU memory under multi-worker load.
_PTLFLOW_MODEL_CACHE: dict = {}


def _get_flow_model(device: str):
    if device not in _PTLFLOW_MODEL_CACHE:
        model = ptlflow.get_model("dpflow", ckpt_path="sintel").to(device).eval()
        _PTLFLOW_MODEL_CACHE[device] = model
    return _PTLFLOW_MODEL_CACHE[device]


def paste_segment_to_full_image(segment_crop, x_start, y_start, full_height, full_width):
    """Paste a cropped binary segment back into its position in the full image."""
    full_segment = np.zeros((full_height, full_width), dtype=segment_crop.dtype)
    h, w = segment_crop.shape
    x_end = min(x_start + w, full_height)
    y_end = min(y_start + h, full_width)
    crop_x_end = x_end - x_start
    crop_y_end = y_end - y_start
    full_segment[y_start:y_end, x_start:x_end] = segment_crop[:crop_y_end, :crop_x_end]
    return full_segment


def get_dot_product_map(avg_flow, flow_cond_with_obj):
    dx, dy = np.array(flow_cond_with_obj[-1][2:]) - np.array(flow_cond_with_obj[-1][0:2])
    direction = np.array([dx, dy])
    dot_prod = torch.sum(
        avg_flow * torch.tensor(direction, dtype=avg_flow.dtype, device=avg_flow.device)[None, None, :],
        dim=-1,
    )
    return dot_prod


def square_crop_with_padding(image, seg_mask, probe_point, padding=25, out_size=256):
    """Crop a square region tight around `seg_mask` with `padding` extra pixels."""
    x, y = probe_point

    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy() if image.ndim == 3 else image.cpu().numpy()
        seg_mask_np = seg_mask.cpu().numpy() if isinstance(seg_mask, torch.Tensor) else seg_mask
    else:
        image_np = image
        seg_mask_np = seg_mask

    H, W = image_np.shape[:2]

    y_indices, x_indices = np.where(seg_mask_np > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("Segmentation mask is empty.")

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    half_size = max(x_max - x_min, y_max - y_min) // 2 + padding

    x_start = center_x - half_size
    x_end = center_x + half_size + 1
    y_start = center_y - half_size
    y_end = center_y + half_size + 1

    def adjust_bounds(start, end, max_val):
        length = end - start
        if start < 0:
            end = min(end - start, max_val)
            start = 0
        if end > max_val:
            start = max(0, start - (end - max_val))
            end = max_val
        return start, start + length

    x_start, x_end = adjust_bounds(x_start, x_end, W)
    y_start, y_end = adjust_bounds(y_start, y_end, H)

    crop_width = x_end - x_start
    crop_height = y_end - y_start
    diff = abs(crop_height - crop_width)

    if crop_width < crop_height and x_end + diff <= W:
        x_end += diff
    elif crop_width < crop_height:
        x_start = max(0, x_start - diff)
    elif crop_height < crop_width and y_end + diff <= H:
        y_end += diff
    elif crop_height < crop_width:
        y_start = max(0, y_start - diff)

    cropped_img = image_np[y_start:y_end, x_start:x_end]
    cropped_mask = seg_mask_np[y_start:y_end, x_start:x_end]

    ratio = out_size / cropped_img.shape[0]

    cropped_img = cv2.resize(cropped_img, (out_size, out_size), interpolation=cv2.INTER_AREA)

    x = (x - x_start) * ratio
    y = (y - y_start) * ratio

    if isinstance(image, torch.Tensor):
        cropped_img = torch.from_numpy(cropped_img).permute(2, 0, 1) if image.ndim == 3 else torch.from_numpy(cropped_img)
        cropped_mask = torch.from_numpy(cropped_mask)

    return cropped_img, cropped_mask, int(x), int(y), x_start, y_start, x_end, y_end, ratio


def convert_iterative_bboxes_to_absolute(bboxes, ratios):
    """Chain a list of nested relative bboxes into absolute original-image coords."""
    abs_bboxes = []
    offset = np.array([0, 0], dtype=np.float32)
    ratio = 1

    for ct, bbox in enumerate(bboxes):
        bbox = bbox.astype(np.float32)
        if ct > 0:
            ratio *= 1 / ratios[ct - 1]
        abs_bbox = bbox.copy() * ratio
        abs_bbox[0] += offset[0]
        abs_bbox[2] += offset[0]
        abs_bbox[1] += offset[1]
        abs_bbox[3] += offset[1]
        abs_bboxes.append(abs_bbox)
        offset = abs_bbox[:2]

    return [np.round(b).astype(np.int32) for b in abs_bboxes]


def resize_segment_to_original(segment, final_bbox_in_orig, original_image_shape):
    """Resize a cropped segment to its bbox in the original image and paste in place."""
    H, W = original_image_shape
    x_start, y_start, x_end, y_end = final_bbox_in_orig

    crop_width = min(x_end, W) - x_start
    crop_height = min(y_end, H) - y_start

    resized_segment = cv2.resize(segment, (crop_width, crop_height), interpolation=cv2.INTER_NEAREST)
    full_segment = np.zeros((H, W), dtype=resized_segment.dtype)
    full_segment[y_start:y_start + crop_height, x_start:x_start + crop_width] = resized_segment
    return full_segment


def sample_distant_point_on_segment(segment_map, point, min_dist=8, max_dist=20, max_tries=100):
    """Sample a point on the segment within [min_dist, max_dist] L2 of `point`."""
    H, W = segment_map.shape
    x0, y0 = point

    for _ in range(max_tries):
        angle = torch.rand(1).item() * 2 * np.pi
        radius = torch.empty(1).uniform_(min_dist, max_dist).item()
        dy = int(round(radius * np.sin(angle)))
        dx = int(round(radius * np.cos(angle)))
        y_new, x_new = y0 + dy, x0 + dx
        if 0 <= y_new < H and 0 <= x_new < W and segment_map[y_new, x_new] == 1:
            return torch.tensor([x_new, y_new])
    return False


def compute_flow(img1, img2, device: str = "cuda"):
    """Compute dense optical flow between two RGB images via SEA-RAFT (ptlflow).

    Returns a (H, W, 2) torch tensor on `device`. The underlying ptlflow
    model is loaded lazily on first call and cached per device.
    """
    def to_bgr_uint8(x):
        if isinstance(x, Image.Image):
            x = np.array(x)
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            if x.ndim == 3 and x.shape[0] in {1, 3}:
                x = np.transpose(x, (1, 2, 0))
        if x.dtype != np.uint8:
            x = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    cv1, cv2_img = map(to_bgr_uint8, (img1, img2))

    model = _get_flow_model(device)

    io_adapter = IOAdapter(model, cv1.shape[:2])
    inputs = io_adapter.prepare_inputs([cv1, cv2_img])
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        preds = model(inputs)

    flows = preds["flows"][0, 0].permute(1, 2, 0)
    return flows


def get_segment_gen2_rgb(im, probe_point, flow_predictor, flow_cond,
                         num_seq=0, num_seeds=1, num_dirs=5,
                         min_mag=10.0, max_mag=25.0, topk=None, topp=None,
                         uncond=False, initial_segment=None):
    """RGB-diff segmentation for one probe point on a CWM/ZWM predictor.

    Perturbs the probe patch in `num_dirs` directions × `num_seeds` seeds,
    rolls out the predicted next frame, computes flow against the unperturbed
    frame, and aggregates dot products with the perturbation direction into a
    heatmap that is Otsu-thresholded into a segment.
    """
    all_dot_prods = []
    all_flows = []
    all_probe_points = []

    probe_point_orig = probe_point.clone()
    patch_size = 8  # CWM patch size

    for seed in range(num_seeds):
        dx, dy = offset_multiple_centroids(probe_point, num_dirs, min_mag=min_mag, max_mag=max_mag)

        for ct in range(num_dirs):
            if initial_segment is not None:
                second_point = sample_distant_point_on_segment(initial_segment, probe_point_orig, min_dist=9, max_dist=50)
                if second_point is False:
                    print("No second point found in 100 tries")
                    probe_point = probe_point_orig
                else:
                    probe_point = torch.stack([probe_point_orig, second_point], 0)
            else:
                probe_point = probe_point_orig[None]

            torch.manual_seed(seed + ct)
            x_expand = probe_point[:, 0]
            y_expand = probe_point[:, 1]
            x_new = x_expand + dx[ct]
            y_new = y_expand + dy[ct]

            pt = torch.stack([x_expand, y_expand, x_new, y_new], 1).tolist()
            flow_cond_with_obj = flow_cond if uncond else flow_cond + pt

            # --- source patch block (2x2 on the 32x32 patch grid) ---
            x_expand_patch = max(0, min(30, int(x_expand[0].item() // patch_size)))
            y_expand_patch = max(0, min(30, int(y_expand[0].item() // patch_size)))
            index = y_expand_patch * 32 + x_expand_patch
            src_idxs = [index, index + 1, index + 32, index + 33]

            # --- destination block ---
            x_new_patch = max(0, min(30, int(x_new[0].item() // patch_size)))
            y_new_patch = max(0, min(30, int(y_new[0].item() // patch_size)))
            index_new = y_new_patch * 32 + x_new_patch
            dst_idxs = [index_new, index_new + 1, index_new + 32, index_new + 33]

            def idx_to_xy(idx):
                return idx % 32, idx // 32

            src_or_dst_xy = [idx_to_xy(i) for i in dst_idxs] + [idx_to_xy(i) for i in src_idxs]

            def far_enough(idx, margin=1):
                x, y = idx_to_xy(idx)
                return all(abs(x - dx_dst) > margin or abs(y - dy_dst) > margin
                           for dx_dst, dy_dst in src_or_dst_xy)

            hold_idxs = [
                0, 1, 32, 33,
                30, 31, 62, 63,
                960, 961, 992, 993,
                990, 991, 1022, 1023,
            ]
            hold_idxs = [i for i in hold_idxs if far_enough(i)]

            if src_idxs == dst_idxs:
                continue

            predictions = flow_predictor.hypothetical_prediction(
                im,
                move_points=None,
                patch_size_move_mult=None,
                src_idxs=src_idxs,
                dst_idxs=dst_idxs,
                hold_idxs=hold_idxs,
            )
            frame0 = predictions['frame0_pil']
            pred_frame = predictions['frame1_pred_pil']

            avg_flow = compute_flow(frame0, pred_frame, device=flow_predictor.device)

            dot_product_map = get_dot_product_map(avg_flow, flow_cond_with_obj)
            all_dot_prods.append(dot_product_map)
            all_flows.append(avg_flow.cpu().numpy())

            if len(pt) == 1:
                pt = torch.cat([torch.tensor(pt)] * 2, 0)
            else:
                pt = torch.tensor(pt)
            all_probe_points.append(pt)

    all_dot_prods = torch.stack(all_dot_prods, 0)
    mean_dot_prod = all_dot_prods.mean(0).cpu().numpy()
    all_probe_points = torch.stack(all_probe_points, 0).tolist()

    segment = threshold_heatmap(mean_dot_prod)
    segment_resized = cv2.resize(segment.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)

    all_flows = np.stack(all_flows, 0)
    return segment_resized, all_flows, all_probe_points


def zoom_into_object(im, probe_point, flow_predictor, flow_cond,
                     num_iters=2, num_seq=0, num_seeds=1, num_dirs=5,
                     min_mag=10.0, max_mag=25.0, topk=None, topp=None):
    """Iteratively zoom toward an object, returning the final crop + segment.

    Each iteration: get a segment with `get_segment_gen2_rgb`, square-crop
    around it with padding, then repeat. After `num_iters` zooms, one more
    segment is computed on the final crop.
    """
    all_bbox = [np.array([0, 0, im.shape[1], im.shape[0]])]
    crop = im.copy()
    all_crops = [crop]
    all_probe_points = []
    all_segments = []
    all_flows = []
    all_ratios = [1.0]

    out_size = 256
    padding = 25

    for _ in range(num_iters):
        segment, flows, probe_points_doubled = get_segment_gen2_rgb(
            crop, probe_point, flow_predictor, flow_cond,
            num_seq=num_seq, num_seeds=num_seeds, num_dirs=num_dirs,
            min_mag=min_mag, max_mag=max_mag, topk=topk, topp=topp,
        )
        all_segments.append(segment)
        all_probe_points.append(probe_points_doubled)
        all_flows.append(flows)

        crop, _, x_new, y_new, x_start, y_start, x_end, y_end, ratio = square_crop_with_padding(
            crop, segment, probe_point, padding=padding, out_size=out_size,
        )
        all_ratios.append(ratio)
        probe_point = torch.tensor([x_new, y_new])
        all_crops.append(crop)
        all_bbox.append(np.array([x_start, y_start, x_end, y_end]))

    final_probe_point = probe_point
    final_crop = crop
    final_segment, flows, probe_points_doubled = get_segment_gen2_rgb(
        final_crop, probe_point, flow_predictor, flow_cond,
        num_seq=num_seq, num_seeds=num_seeds, num_dirs=num_dirs,
        min_mag=min_mag, max_mag=max_mag,
    )
    all_probe_points.append(probe_points_doubled)
    all_segments.append(final_segment)
    all_flows.append(flows)

    all_bbox = np.stack(all_bbox, 0)
    all_bbox_in_orig = convert_iterative_bboxes_to_absolute(all_bbox, all_ratios)
    final_bbox_in_orig = all_bbox_in_orig[-1].astype(np.int32)
    all_probe_points = np.array(all_probe_points)

    full_segment_in_orig = resize_segment_to_original(final_segment, final_bbox_in_orig, im.shape[:2])

    return {
        "all_flows": all_flows,
        "all_bboxes": all_bbox,
        "all_crops": all_crops,
        "all_probe_points": all_probe_points,
        "all_segments": all_segments,
        "all_ratios": all_ratios,
        "full_segment_in_orig": full_segment_in_orig,
        "final_probe_point": final_probe_point,
        "final_crop": final_crop,
        "final_segment": final_segment,
        "final_bbox_in_orig": final_bbox_in_orig,
    }
