"""Multi-GPU launcher for the SpelkeBench segmentation eval.

Shards the dataset's image keys across `--gpus` and spawns one
`eval_spelke_seg` worker per GPU via `CUDA_VISIBLE_DEVICES=<id>`. All
recipe knobs are forwarded verbatim to each worker.

Usage:
    python -m zwm.eval.segments.eval_spelke_seg_parallel \\
        --gpus 0 1 2 3 \\
        --model_name awwkl/zwm-babyview-170m/model.pt \\
        --dataset_path data/evals/segments/spelke_bench.h5 \\
        --output_dir viz/eval/segments/spelke_bench/seq16_seeds3_dirs8_zoom0 \\
        --num_seq_patches 16 --num_seeds 3 --num_dirs 8 --num_zoom_iters 0 --num_zoom_dirs 5 \\
        --min_mag_zoom 10.0 --max_mag_zoom 25.0 --min_mag 25.0 --max_mag 35.0

The two-level split (`--num_splits` / `--split_num`) is preserved for users
who want to spread a single run across multiple machines.
"""
from __future__ import annotations

import argparse
import os
import subprocess

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpus', nargs='+', type=int, required=True)
    p.add_argument('--model_name', type=str, required=True)
    p.add_argument('--dataset_path', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)

    p.add_argument('--num_seq_patches', type=int, default=16)
    p.add_argument('--num_seeds', type=int, default=3)
    p.add_argument('--num_dirs', type=int, default=8)
    p.add_argument('--num_zoom_iters', type=int, default=0)
    p.add_argument('--num_zoom_dirs', type=int, default=5)
    p.add_argument('--min_mag_zoom', type=float, default=10.0)
    p.add_argument('--max_mag_zoom', type=float, default=25.0)
    p.add_argument('--min_mag', type=float, default=25.0)
    p.add_argument('--max_mag', type=float, default=35.0)

    p.add_argument('--topk', type=str, default='None')
    p.add_argument('--topp', type=str, default='None')

    # Cross-machine sharding (optional). One machine: num_splits=1, split_num=0.
    p.add_argument('--num_splits', type=int, default=1)
    p.add_argument('--split_num', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    with h5py.File(args.dataset_path, 'r') as f:
        np.random.seed(42)
        keys = sorted(f.keys())
        np.random.shuffle(keys)
        keys_this_machine = np.array_split(keys, args.num_splits)[args.split_num]

    print(f"Total images for this machine: {len(keys_this_machine)} / {len(keys)}")
    if len(keys_this_machine) == 0:
        return

    gpu_chunks = np.array_split(keys_this_machine, len(args.gpus))

    processes = []
    for gpu_id, img_list in zip(args.gpus, gpu_chunks):
        if len(img_list) == 0:
            continue

        cmd = [
            f"CUDA_VISIBLE_DEVICES={gpu_id}",
            "python", "-m", "zwm.eval.segments.eval_spelke_seg",
            "--model_name", args.model_name,
            "--dataset_path", args.dataset_path,
            "--img_names", *list(img_list),
            "--output_dir", args.output_dir,
            "--num_seq_patches", str(args.num_seq_patches),
            "--num_seeds", str(args.num_seeds),
            "--num_dirs", str(args.num_dirs),
            "--num_zoom_iters", str(args.num_zoom_iters),
            "--num_zoom_dirs", str(args.num_zoom_dirs),
            "--min_mag_zoom", str(args.min_mag_zoom),
            "--max_mag_zoom", str(args.max_mag_zoom),
            "--min_mag", str(args.min_mag),
            "--max_mag", str(args.max_mag),
            "--device", "cuda:0",
        ]
        if args.topk != 'None':
            cmd += ["--topk", args.topk]
        if args.topp != 'None':
            cmd += ["--topp", args.topp]

        full_cmd = " ".join(cmd)
        print(f"[GPU {gpu_id}] {len(img_list)} images")
        processes.append(subprocess.Popen(full_cmd, shell=True))

    for p in processes:
        p.wait()


if __name__ == "__main__":
    main()
