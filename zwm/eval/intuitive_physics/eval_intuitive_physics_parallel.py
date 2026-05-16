"""Multi-GPU launcher for the intuitive-physics eval.

Reads `annotations.csv`, shards the 100 item keys across `--gpus`, and
spawns one `eval_intuitive_physics` worker per GPU via
`CUDA_VISIBLE_DEVICES=<id>`. All seeds run inside each worker (so the model
loads once per GPU, not once per seed).

Usage:
    python -m zwm.eval.intuitive_physics.eval_intuitive_physics_parallel \
        --gpus 0 1 2 3 4 5 6 7 \
        --model_name awwkl/zwm-babyview-170m/model.pt \
        --dataset_dir data/evals/intuitive_physics \
        --output_dir viz/eval/intuitive_physics/seeds8_gap10 \
        --seeds 0 1 2 3 4 5 6 7 --frame_gap 10
"""
from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpus', nargs='+', type=int, required=True)
    p.add_argument('--model_name', type=str, required=True)
    p.add_argument('--dataset_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)

    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument('--frame_gap', type=int, default=10)
    p.add_argument('--square_length_in_patches', type=int, default=4)

    p.add_argument('--items_limit', type=int, default=0,
                   help='If >0, only process the first N items (for smoke tests).')

    # Cross-machine sharding (optional).
    p.add_argument('--num_splits', type=int, default=1)
    p.add_argument('--split_num', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(os.path.join(args.dataset_dir, 'annotations.csv'), dtype=str)
    items = [f"{c}_{v}" for c, v in zip(df['category'], df['video_id'])]
    # Stable shuffle so multi-machine splits are deterministic.
    rng = np.random.default_rng(42)
    rng.shuffle(items)
    items_this_machine = np.array_split(items, args.num_splits)[args.split_num].tolist()

    if args.items_limit > 0:
        items_this_machine = items_this_machine[:args.items_limit]

    print(f"Total items for this machine: {len(items_this_machine)} / {len(items)}")
    if not items_this_machine:
        return

    gpu_chunks = np.array_split(items_this_machine, len(args.gpus))

    processes = []
    for gpu_id, item_list in zip(args.gpus, gpu_chunks):
        if len(item_list) == 0:
            continue

        cmd = [
            f"CUDA_VISIBLE_DEVICES={gpu_id}",
            "python", "-m", "zwm.eval.intuitive_physics.eval_intuitive_physics",
            "--model_name", args.model_name,
            "--dataset_dir", args.dataset_dir,
            "--output_dir", args.output_dir,
            "--items", *list(item_list),
            "--seeds", *[str(s) for s in args.seeds],
            "--frame_gap", str(args.frame_gap),
            "--square_length_in_patches", str(args.square_length_in_patches),
            "--device", "cuda:0",
        ]
        full_cmd = " ".join(cmd)
        print(f"[GPU {gpu_id}] {len(item_list)} items x {len(args.seeds)} seeds")
        processes.append(subprocess.Popen(full_cmd, shell=True))

    for p in processes:
        p.wait()


if __name__ == "__main__":
    main()
