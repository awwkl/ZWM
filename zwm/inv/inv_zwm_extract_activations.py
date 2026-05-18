"""
CUDA_VISIBLE_DEVICES=5 \
python zwm/inv/inv_zwm_extract_activations.py \
    --image_dir /path/to/image_dir \
    --model_name awwkl/zwm-bvd-170m/model.pt \
    --forward_mode two_image \
    --debug \
"""

import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
from zwm.zwm_predictor import ZWMPredictor
import torch
from zwm.utils.viz import fig_to_img
from decord import VideoReader
from tqdm import tqdm
import shutil
from collections import defaultdict
from contextlib import contextmanager

random.seed(42)
np.random.seed(42)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='awwkl/zwm-bvd-170m/model.pt', help='ZWM Model Name (from gcloud)')
    parser.add_argument('--image_dir', type=str, help='path to images')
    parser.add_argument('--out_viz_dir', type=str, default='viz/extract_activations', help='path to output directory')
    parser.add_argument('--forward_mode', type=str, default='one_image', choices=['one_image', 'one_image_plus_10perc', 'two_image'], help='which forward mode to use')
    parser.add_argument('--n_samples_to_eval', type=int, default=1000, help='number of samples to evaluate')
    parser.add_argument('--num_viz', type=int, default=10, help='iteration interval for visualization')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--frame0_mask_ratio', type=float, default=0.0, help='mask ratio for frame 1')
    return parser.parse_args()

# --- minimal, always-clone-and-cpu hook helper ---
@contextmanager
def capture_activations_cpu(model: torch.nn.Module, layer_names):
    acts = defaultdict(list)
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            acts[name].append(t[0][0].detach().cpu().numpy())
        return hook

    for name in layer_names:
        try:
            m = model.get_submodule(name)
            handles.append(m.register_forward_hook(make_hook(name)))
        except Exception:
            pass

    try:
        yield acts
    finally:
        for h in handles:
            h.remove()

def call_single_image_forward(args):
    zwm_predictor = ZWMPredictor(model_name=args.model_name)
    backbone = zwm_predictor.model  # per your note

    image_files = glob.glob(os.path.join(args.image_dir, '**/*.jpg'), recursive=True)
    random.shuffle(image_files)

    final_layer_num = len(zwm_predictor.model.transformer.h) - 1
    block_ids = list(range(0, final_layer_num + 1, 2)) + [final_layer_num]
    layer_names = [f"transformer.h.{n}" for n in block_ids]  # names relative to backbone

    for i, image_file in enumerate(tqdm(image_files, total=args.n_samples_to_eval)):
        if i >= args.n_samples_to_eval:
            break
        try:
            image_name = os.path.basename(image_file).split('.')[0]
            image = Image.open(image_file).convert("RGB")

            with torch.inference_mode():
                with capture_activations_cpu(backbone, layer_names) as activations:
                    if args.forward_mode == 'one_image':
                        results = zwm_predictor.single_image_forward(image, frame_gap=-1)
                    elif args.forward_mode == 'one_image_plus_10perc':
                        results = zwm_predictor.factual_prediction(image, image, mask_ratio=0.9, frame_gap=-1)
                    elif args.forward_mode == 'two_image':
                        results = zwm_predictor.factual_prediction(image, image, mask_ratio=0.0, frame_gap=-1)

            if activations:
                for k, v in activations.items():
                    act_save_dir = os.path.join(args.output_activations_dir, f"zwm_{args.forward_mode}--{args.model_name.replace('/', '_')}--{k}")
                    act_save_path = os.path.join(act_save_dir, f'{image_name}.npy')
                    os.makedirs(act_save_dir, exist_ok=True)
                    np.save(act_save_path, v)

            if i < args.num_viz:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                for axis in ax:
                    axis.title.set_fontsize(35)
                    axis.axis('off')
                ax[0].imshow(results['frame0_pil']);         ax[0].set_title("Frame 0")
                ax[1].imshow(results['frame0_pred_raw_PIL']); ax[1].set_title("Frame 0 Pred Raw")
                img = fig_to_img(fig)
                img_viz_path = os.path.join(args.out_viz_dir, f'iter_{i}_{image_name}.png')
                img.save(img_viz_path)

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            continue

if __name__ == "__main__":
    args = get_args()
    args.output_activations_dir = os.path.join(args.out_viz_dir, 'activations')
    args.out_viz_dir = os.path.join(args.out_viz_dir, args.forward_mode, args.model_name)
    if os.path.exists(args.out_viz_dir):
        shutil.rmtree(args.out_viz_dir)
    os.makedirs(args.out_viz_dir, exist_ok=True)
    
    if args.debug:
        args.n_samples_to_eval = args.num_viz
    
    print(args)
    call_single_image_forward(args)
