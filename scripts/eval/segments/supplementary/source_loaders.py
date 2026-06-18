"""Source-dataset loaders shared by the Feature-NN SpelkeBench eval + its graders.

Scope: only the segmentation feature-NN scripts in this directory
(eval_spelke_seg_feature_nn.py and the two graders). NOT used by the flow eval.

The per-cell result files store ONLY `segment_pred` — `image` and `segment_gt`
are identical across every (layer, tau) cell and every model, so rather than
duplicating them ~64x per image per model we read them back from the canonical
source h5 (data/evals/segments/spelke_bench.h5) here. Keeping the loaders in one
module means the worker and the graders share identical resize logic.

These helpers sit in the same dir as the by-path scripts that import them, so
`import source_loaders` resolves via sys.path[0]; `zwm` resolves via PYTHONPATH
(set by the .sh wrappers).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from zwm.eval.segments.segment import compute_segment_centroids


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
