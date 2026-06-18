"""Shared building blocks for the matched feature-NN segmentation probe.

Used by BOTH the ZWM probe (scripts/eval/segments/supplementary/) and the
representation-baseline probes (scripts/eval/segments/baselines/), so the readout
is byte-identical across models. Holds the backbone-agnostic pieces:

  - source-dataset loaders (image / GT segments / seed centroids),
  - the cosine-NN segment rule (seed patch -> affinity -> tau -> connected
    component -> upsample),
  - the lean per-cell h5 IO.

Each backbone only supplies its own feature extractor returning a
[num_patches, n_embd] patch grid (row-major); everything downstream is shared.
Cell files store ONLY segment_pred — image/segment_gt are identical across every
(layer, tau) cell, so they are read back from the source SpelkeBench h5 here.
"""
from __future__ import annotations

import os
import shutil

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from zwm.eval.segments.segment import compute_segment_centroids


# ------------------------------------------------------------
# Source-dataset loaders (image / GT segments / centroids are not stored per cell).
# ------------------------------------------------------------

def load_image(src_h5, img_key, res):
    """RGB image [res, res, 3] uint8 from the source h5 (resized to match the eval)."""
    im = src_h5[img_key]['rgb'][:][:, :, :3]
    if im.shape[0] != res or im.shape[1] != res:
        im = cv2.resize(im, (res, res), interpolation=cv2.INTER_AREA)
    return im


def load_segments(src_h5, img_key, res):
    """GT segments [N, res, res] uint8 from the source h5 (nearest-resized to match)."""
    seg = src_h5[img_key]['segment'][:]
    if seg.shape[1] != res:
        seg = np.stack(
            [cv2.resize(s.astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
             for s in seg],
            axis=0,
        )
    return seg


def load_centroids(src_h5, img_key, segments):
    """Seed centroids [N, 2] ([x, y]) — stored if present, else computed from segments."""
    if 'centroid' in src_h5[img_key].keys():
        centroids = torch.tensor(src_h5[img_key]['centroid'][:])
    else:
        centroids = compute_segment_centroids(torch.tensor(segments))
    return centroids[:segments.shape[0]]


# ------------------------------------------------------------
# Cosine-NN segment rule (the standard DINOv2 / V-JEPA2 point-seg rule, sans CRF).
# ------------------------------------------------------------

def featurenn_segment(feats, centroid, grid, res, tau):
    """Cosine-NN segment for one seed centroid.

    feats: [num_patches, n_embd] patch features for the image (num_patches = grid**2,
           row-major). centroid: [x, y] seed point in `res`-pixel space.
    Returns a binary mask [res, res] (uint8).
    """
    patch_px = res / grid
    cx, cy = float(centroid[0]), float(centroid[1])
    qcol = min(max(int(cx // patch_px), 0), grid - 1)
    qrow = min(max(int(cy // patch_px), 0), grid - 1)

    seed_feat = feats[qrow * grid + qcol]
    cos = F.cosine_similarity(seed_feat.unsqueeze(0), feats, dim=-1)  # [num_patches]
    mask_grid = (cos > tau).reshape(grid, grid).detach().cpu().numpy().astype(np.uint8)

    # Connected component containing the seed (fall back to the largest CC).
    labeled, n_components = ndimage.label(mask_grid)
    seed_component = labeled[qrow, qcol]
    if seed_component == 0:
        if n_components == 0:
            return np.zeros((res, res), dtype=np.uint8)
        sizes = [int((labeled == i).sum()) for i in range(1, n_components + 1)]
        seed_component = int(np.argmax(sizes)) + 1
    mask = (labeled == seed_component).astype(np.uint8)

    return cv2.resize(mask, (res, res), interpolation=cv2.INTER_NEAREST)


# ------------------------------------------------------------
# Lean per-cell h5 IO (segment_pred only).
# ------------------------------------------------------------

def has_predictions(out_path):
    """True if out_path exists and already holds predictions (resume-safe skip)."""
    if not os.path.exists(out_path):
        return False
    try:
        with h5py.File(out_path, 'r') as f:
            return "segment_pred" in f.keys()
    except OSError:
        return False


def write_result_h5(out_path, preds, tmp_root="/tmp/spelke_seg_featurenn"):
    """Write one per-image result h5 (segment_pred only) via /tmp then move."""
    tmp_path = os.path.join(tmp_root, f"{os.path.basename(out_path)}.{np.random.randint(0, int(1e10))}.tmp")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("segment_pred", data=preds, compression="gzip")
    shutil.move(tmp_path, out_path)
