"""Intuitive-physics benchmark rollout — per-item worker.

For each (category, video_id, seed) in `--items` x `--seeds`, conditions on
the context frame (`frame_02.png`) plus a small set of revealed patches
around the factual grounding point, runs `ZWMPredictor.factual_prediction`,
and writes the predicted target frame as a PNG. Also writes a "ctf_drawn"
PNG (prediction with the conditioning boxes drawn on top) and, for seed 0,
a frame2 -> ctf_drawn GIF.

Driven by `eval_intuitive_physics_parallel.py` for multi-GPU sharding; not
normally invoked directly.

Usage (typically via the launcher, not directly):
    python -m zwm.eval.intuitive_physics.eval_intuitive_physics \
        --model_name awwkl/zwm-babyview-170m/model.pt \
        --dataset_dir data/evals/intuitive_physics \
        --items 1.cohesion_000 1.cohesion_001 \
        --seeds 0 1 2 3 4 5 6 7 \
        --output_dir viz/eval/intuitive_physics/seeds8_gap10
"""
from __future__ import annotations

import argparse
import os
import random

import pandas as pd
import torchvision
from PIL import Image, ImageDraw

from zwm.zwm_predictor import ZWMPredictor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, required=True,
                   help='HF repo path or local checkpoint, e.g. awwkl/zwm-babyview-170m/model.pt')
    p.add_argument('--dataset_dir', type=str, required=True,
                   help='Root with annotations.csv + keyframes/.')
    p.add_argument('--items', type=str, nargs='+', required=True,
                   help='Item keys to process: "<category>_<video_id>".')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument('--output_dir', type=str, required=True,
                   help='Worker writes <model_slug>/{pred,ctf_drawn,gif}/<key>_seed<NN>.<ext> under here.')

    p.add_argument('--frame_gap', type=int, default=10)
    p.add_argument('--square_length_in_patches', type=int, default=4,
                   help='Side length (in patches) of each conditioning box around an unmask point.')
    p.add_argument('--device', type=str, default='cuda:0')
    return p.parse_args()


def get_unmask_points(factual_x: int, factual_y: int, rng: random.Random):
    """1 grounding point + 3-5 random top edge + 3-5 random bottom edge points (all in 512x512 space)."""
    unmask_points = [(factual_x, factual_y)]

    fixed_top_points = [(i, 32) for i in range(32, 481, 64)]
    fixed_bottom_points = [(i, 480) for i in range(32, 481, 64)]

    n_top = rng.randint(3, 5)
    unmask_points.extend(rng.sample(fixed_top_points, n_top))
    n_bottom = rng.randint(3, 5)
    unmask_points.extend(rng.sample(fixed_bottom_points, n_bottom))
    return unmask_points


def convert_unmask_points_to_unmask_indices(
    unmask_points, patch_size: int, square_length_in_patches: int, resolution: int,
):
    """Map (x, y) -> square_length^2 patch indices in a `resolution`-sized image."""
    num_patches_per_side = resolution // patch_size
    side = square_length_in_patches
    offsets = [(i, j) for i in range(side) for j in range(side)]

    indices = []
    for ux, uy in unmask_points:
        top_left_x = ux - patch_size * side // 2
        top_left_y = uy - patch_size * side // 2
        top_left_idx = (top_left_y // patch_size) * num_patches_per_side + (top_left_x // patch_size)
        for i, j in offsets:
            indices.append(top_left_idx + j * num_patches_per_side + i)
    return indices


def add_factual_drawing(image: Image.Image, unmask_points_512: list[tuple[int, int]]) -> Image.Image:
    """Draw 64x64 outlined boxes at each unmask point on a 512x512 copy of the image."""
    image = torchvision.transforms.Resize((512, 512))(image)
    draw = ImageDraw.Draw(image)
    for i, (x, y) in enumerate(unmask_points_512):
        color = (0, 255, 0) if i == 0 else (255, 0, 0)
        x = round(x / 16) * 16
        y = round(y / 16) * 16
        draw.rectangle([x - 32, y - 32, x + 32, y + 32], outline=color, width=6, fill=None)
    return image


def main():
    args = parse_args()

    model_slug = args.model_name.replace('/', '_').replace('.pt', '')
    base = os.path.join(args.output_dir, model_slug)
    pred_dir = os.path.join(base, 'pred')
    ctf_dir = os.path.join(base, 'ctf_drawn')
    gif_dir = os.path.join(base, 'gif')
    for d in [pred_dir, ctf_dir, gif_dir]:
        os.makedirs(d, exist_ok=True)

    annotations_df = pd.read_csv(os.path.join(args.dataset_dir, 'annotations.csv'), dtype=str)
    annotations_df['key'] = annotations_df['category'] + '_' + annotations_df['video_id']
    items_set = set(args.items)
    rows = annotations_df[annotations_df['key'].isin(items_set)].reset_index(drop=True)
    missing = items_set - set(rows['key'])
    if missing:
        raise ValueError(f'Items not found in annotations.csv: {sorted(missing)}')

    predictor = ZWMPredictor(model_name=args.model_name, device=args.device)
    resolution = predictor.model.config.resolution
    patch_size = predictor.model.config.patch_size
    # Conditioning-box math assumes 32x32 patches at 256x256 resolution; fail loudly otherwise.
    assert resolution // patch_size == 32, (
        f'eval_intuitive_physics expects resolution/patch_size == 32; '
        f'got resolution={resolution}, patch_size={patch_size}'
    )

    keyframes_dir = os.path.join(args.dataset_dir, 'keyframes')

    for _, row in rows.iterrows():
        category = row['category']
        video_id = row['video_id']
        item_key = f'{category}_{video_id}'
        factual_x = int(row['factual_x'])
        factual_y = int(row['factual_y'])

        frame2_path = os.path.join(keyframes_dir, category, video_id, 'frame_02.png')
        frame3_path = os.path.join(keyframes_dir, category, video_id, 'frame_03.png')
        frame2_pil = Image.open(frame2_path).convert('RGB')
        frame3_pil = Image.open(frame3_path).convert('RGB')

        for seed in args.seeds:
            seed_tag = f'seed{seed:02d}'
            pred_path = os.path.join(pred_dir, f'{item_key}_{seed_tag}.png')
            ctf_path = os.path.join(ctf_dir, f'{item_key}_{seed_tag}.png')
            gif_path = os.path.join(gif_dir, f'{item_key}_{seed_tag}.gif')

            if os.path.exists(pred_path):
                print(f'[skip] {pred_path} exists')
                continue

            # Private RNG for unmask sampling so model state and sampler are decoupled
            # (matches CWM's `fixed_point_sampler_rng`). Reseed per (item, seed) so
            # identical seeds yield identical unmask configurations across runs.
            unmask_rng = random.Random(seed)
            unmask_points_512 = get_unmask_points(factual_x, factual_y, unmask_rng)
            # Annotations are in 512x512; predictor runs at 256x256.
            unmask_points_256 = [(x // 2, y // 2) for x, y in unmask_points_512]
            unmask_indices = convert_unmask_points_to_unmask_indices(
                unmask_points_256, patch_size=patch_size,
                square_length_in_patches=args.square_length_in_patches,
                resolution=resolution,
            )

            result = predictor.factual_prediction(
                frame0=frame2_pil,
                frame1=frame3_pil,
                frame_gap=args.frame_gap,
                unmask_indices=unmask_indices,
                seed=seed,
            )
            pred_img = result['frame1_pred_pil_unmasked']
            pred_img.save(pred_path)

            ctf_img = add_factual_drawing(pred_img.copy(), unmask_points_512)
            ctf_img.save(ctf_path)

            if seed == 0:
                frame2_512 = frame2_pil.resize((512, 512))
                # PIL GIF duration is in milliseconds (2000ms = 2s per frame).
                frame2_512.save(
                    gif_path, save_all=True, append_images=[ctf_img],
                    duration=2000, loop=0,
                )

            print(f'[done] {item_key} {seed_tag}')


if __name__ == '__main__':
    main()
