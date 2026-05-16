"""Segmentation helpers used by the SpelkeBench eval.

Ported from ccwm/utils/segment.py — only the functions reached by the CWM /
RGB-diff segmentation path are kept (no flow-quantizer, no logits utilities).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment


def batched_iou(x, y=None):
    """IoU between (B, H, W) boolean tensors."""
    if y is None:
        y = x

    xp = x[:, None]
    yp = y[None]

    intersection = (xp & yp).sum(axis=(-1, -2))
    union = (xp | yp).sum(axis=(-1, -2))

    return intersection / union


def evaluate_AP_AR_single_image(pred_segments, gt_segments):
    """AP / AR for a single image at IoU thresholds 0.50:0.05:0.95.

    Procedure: Hungarian-match predictions to GT to maximize total IoU, then
    count true positives at each IoU threshold; precision and recall are
    averaged across thresholds.
    """
    iou_mat = batched_iou(gt_segments, pred_segments)
    gt_inds, pred_inds = linear_sum_assignment(1. - iou_mat)

    ious = iou_mat[gt_inds, pred_inds]

    num_gt_segments = gt_segments.shape[0]
    num_pred_segments = pred_segments.shape[0]

    precisions = []
    recalls = []
    thresholds = np.arange(start=0.50, stop=0.95, step=0.05)

    for iou_thresh in thresholds:
        tp = np.count_nonzero(ious >= iou_thresh)
        precisions.append(tp / num_pred_segments if num_pred_segments else 0)
        recalls.append(tp / num_gt_segments if num_gt_segments else 0)

    return {
        'AP': np.mean(precisions),
        'AR': np.mean(recalls),
        'assignments': [gt_inds, pred_inds],
        'iou_mat': iou_mat,
        'thresholds': thresholds,
    }


def safe_central_point(mask, erosion_kernel=5):
    """Pick a central point inside a binary mask. Returns [x, y] or None."""
    mask_np = mask.cpu().numpy().astype(np.uint8)

    coords = torch.nonzero(mask, as_tuple=False).float()
    if coords.shape[0] == 0:
        return None

    centroid = coords.mean(dim=0).round().long()
    y_c, x_c = centroid.tolist()

    H, W = mask.shape
    if 0 <= y_c < H and 0 <= x_c < W and mask[y_c, x_c] > 0:
        return [x_c, y_c]

    mask_t = mask.unsqueeze(0).unsqueeze(0).float()
    eroded = -F.max_pool2d(-mask_t, kernel_size=erosion_kernel, stride=1, padding=erosion_kernel // 2)
    eroded_bin = (eroded > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    used_mask = eroded_bin if eroded_bin.sum() > 0 else mask_np
    dist = distance_transform_edt(used_mask)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return [x, y]


def compute_segment_centroids(segments):
    pts = [safe_central_point(seg) for seg in segments]
    return torch.tensor(np.array(pts)).to(segments.device)


def offset_multiple_centroids(centroids, N, min_mag=10.0, max_mag=25.0):
    """Sample N equi-angular offsets (one magnitude per direction, shared across centroids)."""
    device = centroids.device
    angles = torch.arange(N, device=device) * (2 * math.pi / N)
    magnitudes = torch.rand(N, device=device) * (max_mag - min_mag) + min_mag
    dx = magnitudes * torch.cos(angles)
    dy = magnitudes * torch.sin(angles)
    return dx, dy


def threshold_heatmap(heatmap):
    """Min-max normalize then Otsu-threshold a 2D heatmap to {0,1}."""
    heatmap_min = np.min(heatmap)
    heatmap_max = np.max(heatmap)
    heatmap_norm = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
    heatmap_scaled = (heatmap_norm * 255).astype(np.uint8)
    _, thresh = cv2.threshold(heatmap_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh // 255
