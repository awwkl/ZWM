"""SpelkeBench segmentation rollout — per-image worker (CWM/ZWM path).

Reads one or more image keys from a SpelkeBench-style h5, runs RGB-diff
segmentation for each GT segment's centroid, and writes one
`<image_key>.h5` per image with datasets `image / segment_gt / segment_pred`.

Usage:
    python -m zwm.eval.segments.eval_spelke_seg \\
        --model_name awwkl/zwm-babyview-170m/model.pt \\
        --dataset_path data/evals/segments/spelke_bench.h5 \\
        --img_names image_3345 image_3346 \\
        --output_dir viz/eval/segments/spelke_bench/seq16_seeds3_dirs8_zoom0 \\
        --num_seq_patches 16 --num_seeds 3 --num_dirs 8 --num_zoom_iters 0 --num_zoom_dirs 5 \\
        --min_mag_zoom 10.0 --max_mag_zoom 25.0 --min_mag 25.0 --max_mag 35.0

Driven by `eval_spelke_seg_parallel.py` for multi-GPU sharding; not normally
invoked directly.
"""
from __future__ import annotations

import argparse
import os
import shutil

import cv2
import h5py
import numpy as np
import torch

from zwm.eval.segments.segment import compute_segment_centroids
from zwm.eval.segments.segment_zoom import (
    get_segment_gen2_rgb,
    resize_segment_to_original,
    zoom_into_object,
)
from zwm.zwm_predictor import ZWMPredictor


# Four-corner "hold" probes — keep image boundaries stationary.
FLOW_COND_BASE = [
    [512 - 16, 0,        512 - 16, 0],
    [0,        0,        0,        0],
    [0,        512 - 16, 0,        512 - 16],
    [512 - 16, 512 - 16, 512 - 16, 512 - 16],
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, required=True,
                   help='HF repo path or local checkpoint, e.g. awwkl/zwm-babyview-170m/model.pt')
    p.add_argument('--dataset_path', type=str, required=True)
    p.add_argument('--img_names', type=str, nargs='+', required=True,
                   help='h5 keys to process on this worker.')
    p.add_argument('--output_dir', type=str, required=True,
                   help='Worker writes <model_slug>/<key>.h5 under here.')

    p.add_argument('--num_seq_patches', type=int, default=16)
    p.add_argument('--num_seeds', type=int, default=3)
    p.add_argument('--num_dirs', type=int, default=8)

    p.add_argument('--num_zoom_iters', type=int, default=0)
    p.add_argument('--num_zoom_dirs', type=int, default=5)
    p.add_argument('--min_mag_zoom', type=float, default=10.0)
    p.add_argument('--max_mag_zoom', type=float, default=25.0)

    p.add_argument('--min_mag', type=float, default=25.0)
    p.add_argument('--max_mag', type=float, default=35.0)

    p.add_argument('--topk', type=int, default=None)
    p.add_argument('--topp', type=float, default=None)

    p.add_argument('--device', type=str, default='cuda:0')
    return p.parse_args()


def main():
    args = parse_args()

    model_slug = args.model_name.replace('/', '_').replace('.pt', '')
    out_dir = os.path.join(args.output_dir, model_slug)
    os.makedirs(out_dir, exist_ok=True)

    predictor = ZWMPredictor(model_name=args.model_name, device=args.device)

    with h5py.File(args.dataset_path, 'r') as inp:
        for img_key in args.img_names:
            out_path = os.path.join(out_dir, f"{img_key}.h5")
            if os.path.exists(out_path):
                with h5py.File(out_path, 'r') as f:
                    if "segment_pred" in f.keys():
                        print(f"[skip] {out_path} already has predictions")
                        continue
                    print(f"[overwrite] {out_path} exists but no predictions")

            im = inp[img_key]['rgb'][:][:, :, :3]
            if im.shape[0] != 512 or im.shape[1] != 512:
                im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
            # CWM/ZWM runs at 256x256.
            im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)

            segments = inp[img_key]['segment'][:]
            if segments.shape[1] != 256:
                segments = np.stack(
                    [cv2.resize(s.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
                     for s in segments],
                    axis=0,
                )

            if 'centroid' in inp[img_key].keys():
                centroids = torch.tensor(inp[img_key]['centroid'][:])
            else:
                centroids = compute_segment_centroids(torch.tensor(segments))
            centroids = centroids[:segments.shape[0]]

            print(f"[{img_key}] image {im.shape}, {segments.shape[0]} segments")

            tmp_out_path = os.path.join("/tmp/spelke_seg", f"{img_key}_{np.random.randint(0, int(1e10))}.h5")
            os.makedirs(os.path.dirname(tmp_out_path), exist_ok=True)

            all_pred_segs = []
            with h5py.File(tmp_out_path, "w") as f:
                for seg_idx, centroid in enumerate(centroids):
                    probe_point = torch.tensor(centroid)

                    if args.num_zoom_iters > 0:
                        zoom_result = zoom_into_object(
                            im, probe_point, predictor, FLOW_COND_BASE,
                            num_iters=args.num_zoom_iters, num_seq=64,
                            num_seeds=1, num_dirs=args.num_zoom_dirs,
                            min_mag=args.min_mag_zoom, max_mag=args.max_mag_zoom,
                        )
                        initial_segment = zoom_result["final_segment"]
                        final_crop = zoom_result["final_crop"]
                        final_probe_point = zoom_result["final_probe_point"]
                    else:
                        initial_segment = None
                        final_crop = im
                        final_probe_point = probe_point

                    final_segment, _flows, _probe_points = get_segment_gen2_rgb(
                        final_crop, final_probe_point, predictor, FLOW_COND_BASE,
                        num_seq=args.num_seq_patches,
                        num_seeds=args.num_seeds, num_dirs=args.num_dirs,
                        min_mag=args.min_mag_zoom, max_mag=args.max_mag_zoom,
                        topk=args.topk, topp=args.topp,
                        uncond=False, initial_segment=initial_segment,
                    )

                    if args.num_zoom_iters > 0:
                        final_segment = resize_segment_to_original(
                            final_segment, zoom_result["final_bbox_in_orig"], im.shape[:2],
                        )

                    final_segment = cv2.resize(
                        final_segment.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST,
                    )
                    all_pred_segs.append(final_segment)

                all_pred_segs = np.stack(all_pred_segs)
                f.create_dataset("image", data=im, compression="gzip")
                f.create_dataset("segment_gt", data=segments, compression="gzip")
                f.create_dataset("segment_pred", data=all_pred_segs, compression="gzip")

            # shutil.move (not os.replace) so the /tmp -> project-dir hop works
            # across filesystems; collapses to atomic rename when same-FS.
            shutil.move(tmp_out_path, out_path)
            print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
