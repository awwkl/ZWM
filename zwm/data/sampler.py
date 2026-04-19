"""
python zwm/data/sampler.py \
    --videos_dir /path/to/videos_dir \
    --output_path /path/to/output.csv \
    --shuffle_within_each_day \
    --n_days 1 \
    --n_minutes 30 \
"""

import os
import random
import pandas as pd
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

import math
from torch.utils.data import Sampler, SequentialSampler, BatchSampler
import torch.distributed as dist
from itertools import islice

class ReplicatedSampler(Sampler[int]):
    """Every rank yields the SAME sequence of length N."""
    def __init__(self, order):
        self.order = list(order)
        self.N = len(self.order)

    def __len__(self):
        return self.N

    def __iter__(self):
        return islice(self.order, 0, self.N)
                

def _build_indices_interleave_within_day(df: pd.DataFrame, per_rank_needed: int, seed: int = 0, n_days: int = 1, n_minutes=None):
    """
    Keep day blocks in CSV order; within each N-day block:
    - expand ids by n_repeats
    - shuffle that expanded block
    Then concat blocks and truncate to per_rank_needed.
    """
    rng = random.Random(seed)

    # preserve day order exactly as they appear in the CSV
    day_values_in_order = pd.unique(df['day'].to_numpy())

    base_len = len(df)
    assert base_len > 0, "CSV has no ids"
    n_repeats = math.ceil(per_rank_needed / base_len)

    # normalize n_days
    if n_days is None or n_days < 1:
        n_days = 1

    indices = []
    # process consecutive N-day groups in order
    for i in range(0, len(day_values_in_order), n_days):
        days_block = set(day_values_in_order[i:i + n_days])
        # select rows for this block in CSV order
        ids = df.loc[df['day'].isin(days_block), 'id'].tolist()
        if not ids:
            continue
        if n_minutes is not None:
            # group them by n_minutes * 6 (assuming 10s clips) and shuffle within each
            bucket = n_minutes * 6
            minute_groups = {}
            for local_idx, vid_id in enumerate(ids):   # <- use local index
                minute = local_idx // bucket
                minute_groups.setdefault(minute, []).append(vid_id)
            for minute_block, group in minute_groups.items():
                expanded = [x for x in group for _ in range(n_repeats)]
                rng.shuffle(expanded)
                indices.extend(expanded)
                print(f'day_block {i} minute {minute_block}: {len(group)} ids shuffled')
        else:
            expanded = [x for x in ids for _ in range(n_repeats)]
            rng.shuffle(expanded)
            indices.extend(expanded)
            print(f'day_block {i}: {len(ids)} ids expanded to {len(expanded)} shuffled')

    return indices, n_repeats

def make_sampler_from_csv(csv_path, world_size, rank, total_needed, start_idx=0, seed=0):
    df = pd.read_csv(csv_path)
    if 'day' in df.columns:
        # total_needed is GLOBAL; each rank will iterate ceil(total_needed / world_size)
        per_rank_needed = math.ceil(total_needed / world_size)
        
        n_days = 1
        if 'n_days' in df.columns:
            n_days = int(df['n_days'].iloc[0])
            print('n_days from csv:', n_days)
        n_minutes = None
        if 'n_minutes' in df.columns:
            n_minutes = int(df['n_minutes'].iloc[0])
            print('n_minutes from csv:', n_minutes)

        # build indices with within-day interleaving (post-expansion)
        indices, n_repeats = _build_indices_interleave_within_day(df, per_rank_needed, seed=seed, n_days=n_days, n_minutes=n_minutes)

    else:
        base = df['id'].tolist()
        # repeat each element (in place!) by (total_needed // len(indices) + 1
        per_rank_needed = math.ceil(total_needed / world_size)
        n_repeats = math.ceil(per_rank_needed / len(base))
        indices = [x for x in base for _ in range(n_repeats)]
        
    # start from start_idx, if specified
    print('start_idx: ', start_idx)
    print('len)(indices) before slicing:', len(indices))
    indices = indices[start_idx:] 
    
    sampler = ReplicatedSampler(indices)
    print('per_rank_needed:', per_rank_needed,
          'n_repeats:', n_repeats,
          'len(sampler):', len(sampler))
    return sampler


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', type=str, default=None, help='Path to directory containing video files')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the csv file')
    parser.add_argument('--shuffle_within_each_day', action='store_true', help='If set, shuffle videos within each day')
    parser.add_argument('--n_days', type=int, default=1, help='Number of days to group together when shuffling within each day')
    parser.add_argument('--n_minutes', type=int, default=None, help='If set, shuffle videos within each n_minutes segment')
    return parser.parse_args()

def create_csv():
    args = get_args()
    videos = []
    for root, _, files in os.walk(args.videos_dir):
        for video in files:
            video = os.path.join(root, video)
            if video.endswith(".mp4"):
                videos.append(video)

    videos = sorted(videos)
    id = list(range(len(videos)))
    
    df = pd.DataFrame({"id": id, "video_path": videos})
    
    df["day"] = df["video_path"].str.extract(r"_(\d{4}-\d{2}-\d{2})_")
    df["n_days"] = args.n_days
    if args.n_minutes is not None:
        df['n_minutes'] = args.n_minutes
    df.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    create_csv()
    