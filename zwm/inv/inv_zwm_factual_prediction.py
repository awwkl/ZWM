"""
CUDA_VISIBLE_DEVICES=0 \
python zwm/inv/inv_zwm_factual_prediction.py \
    --videos_dir /ccn2/dataset/kinetics400/Kinetics400/k400/val/ \
    --n_samples_to_eval 100 \
    --model_name awwkl/zwm-bvd-170m/model.pt \
"""

import os
import argparse
import importlib
import random
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
from PIL import Image
from zwm.zwm_predictor import ZWMPredictor
from zwm.data.patch_sequence_dataset import PatchSequenceDataset
import torch
from zwm.utils.model_wrapper import cfg_to_dict, ModelFactory
from zwm.utils.viz import fig_to_img
from zwm.data.image_processing import create_images_from_patches
import torchvision
from decord import VideoReader
from tqdm import tqdm
import shutil

in_transform_without_normalize = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor()
])

random.seed(42)
np.random.seed(42)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='awwkl/zwm-bvd-170m/model.pt', help='ZWM Model Name (from huggingface)')
    parser.add_argument('--videos_dir', type=str, help='path to videos')
    parser.add_argument('--out_viz_dir', type=str, default='viz/zwm_factual_predictions', help='path to output directory')
    parser.add_argument('--n_samples_to_eval', type=int, default=1000, help='number of samples to evaluate')
    parser.add_argument('--num_viz', type=int, default=20, help='iteration interval for visualization')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--frame1_mask_ratio', type=float, default=0.9, help='mask ratio for frame 1')
    return parser.parse_args()

def generate_factual_predictions(args):
    zwm_predictor = ZWMPredictor(model_name=args.model_name)
    
    video_files = glob.glob(os.path.join(args.videos_dir, '**/*.mp4'), recursive=True)
    assert len(video_files) > 0, f"No .mp4 files found in {args.videos_dir}"
    random.shuffle(video_files)

    losses = []
    for i in tqdm(range(args.n_samples_to_eval)):
        video_file = video_files[i % len(video_files)]

        try:
            video_name = os.path.basename(video_file)
            
            vr = VideoReader(video_file)
            num_frames = len(vr)
            ridx0 = np.random.randint(0, num_frames - 16)
            frame_gap = np.random.randint(5, 16)
            ridx1 = ridx0 + frame_gap
            if ridx1 >= num_frames:
                ridx1 = num_frames - 1  # just clamp to the last frame if needed
                
            frames_batch = vr.get_batch([ridx0, ridx1]).asnumpy()  # shape will be [2, H, W, 3]
            frame0_np, frame1_np = frames_batch[0], frames_batch[1]
            
            frame0, frame1 = Image.fromarray(frame0_np), Image.fromarray(frame1_np)
            results = zwm_predictor.factual_prediction(frame0, frame1, frame_gap=frame_gap,
                                                       mask_ratio=args.frame1_mask_ratio,
                                                       seed=i)
            
            if i < args.num_viz:
                fig, ax = plt.subplots(1, 7, figsize=(20, 5))
                for axis in ax:
                    axis.title.set_fontsize(35)
                ax[0].imshow(results['frame0_pil'])
                ax[0].set_title("Frame 0")
                ax[1].imshow(results['frame1_pil'])
                ax[1].set_title("Frame 1 GT")
                ax[2].imshow(results['frame1_pred_pil'])
                ax[2].set_title("Prediction")
                ax[3].imshow(results['frame1_pred_pil_unmasked'])
                ax[3].set_title("Prediction Unmasked")
                ax[4].imshow(results['frame1_with_mask_pil'])
                ax[4].set_title("Frame 1 GT with Mask")
                ax[5].imshow(results['frame0_pred_raw_PIL'])
                ax[5].set_title("Frame 0 Pred Raw")
                ax[6].imshow(results['frame1_pred_raw_PIL'])
                ax[6].set_title("Frame 1 Pred Raw")

                img = fig_to_img(fig)
                img_viz_path = os.path.join(args.out_viz_dir, f'iter_{i}_{video_name}_idx_{ridx0}_{ridx1}.png')
                img.save(img_viz_path)
            
            loss_func = torch.nn.MSELoss(reduction='mean')
            frame1_tensor = in_transform_without_normalize(frame1).unsqueeze(0)  # add batch dimension
            frame1_pred_tensor = in_transform_without_normalize(results['frame1_pred_pil']).unsqueeze(0)
            mse_loss_value = loss_func(frame1_tensor, frame1_pred_tensor)
            losses.append(mse_loss_value.item())
            
            if i % 100 == 0:
                print(f"Mean loss: {np.mean(losses)}")
            
        except Exception as e:
            print(f"Error processing video {video_file}: {e}")
            continue
        
    print(f"Mean loss: {np.mean(losses)}")
       
    # create text file with the loss value 
    loss_file_path = os.path.join(args.out_viz_dir, "loss_value.txt")
    with open(loss_file_path, "w") as f:
        f.write(f"Mean loss: {np.mean(losses)}\n")
    

if __name__ == "__main__":
    args = get_args()
    
    args.out_viz_dir = os.path.join(args.out_viz_dir, args.model_name)
    if os.path.exists(args.out_viz_dir):
        shutil.rmtree(args.out_viz_dir)
    os.makedirs(args.out_viz_dir, exist_ok=True)
    
    print(args)
    generate_factual_predictions(args)