import os
import json
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
import glob
import h5py
from typing import Tuple
from decord import VideoReader

from zwm.utils.sequence_construction import (
    get_frame, add_patch_indexes, get_pos_idxs, shuffle_and_trim_values_and_positions, supress_targets,

)
from zwm.utils.image_processing import patchify

in_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
inv_in_transform = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    torchvision.transforms.Lambda(lambda x: torch.clamp(x, 0, 1)), 
    torchvision.transforms.ToPILImage()
])
resize_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
])
resize_transform_256 = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
])
resize_transform_512 = torchvision.transforms.Compose([
    torchvision.transforms.Resize(512),
])

def get_random_square_crop_transform(image_height, image_width, resolution):
    if image_height == image_width and image_height == resolution:
        crop_transform = torchvision.transforms.Lambda(lambda img: img)
    elif image_height >= image_width:
        starting_pixel_value = np.random.randint(0, image_height - resolution)
        crop_transform = torchvision.transforms.Lambda(lambda img: img.crop((0, starting_pixel_value, img.width, starting_pixel_value + resolution)))
    else:
        starting_pixel_value = np.random.randint(0, image_width - resolution)
        crop_transform = torchvision.transforms.Lambda(lambda img: img.crop((starting_pixel_value, 0, starting_pixel_value + resolution, img.height)))
    return crop_transform

def get_resize_to_target_area_transform(image_height, image_width, target_area, patch_size):
    # it should return a transform that resizes the image such that the number of patches is equal to target_num_patches
    # at the same time maintaining the aspect ratio
    # and also the new height and width should be multiples of patch_size, and the total number of patches should be as close as possible to target_num_patches
    current_area = image_height * image_width
    scale_factor = (target_area / current_area) ** 0.5
    new_height = round(image_height * scale_factor)
    new_width = round(image_width * scale_factor)
    
    # make new_height and new_width multiples of patch_size
    new_height = round(new_height / patch_size) * patch_size
    new_width = round(new_width / patch_size) * patch_size
    resize_transform = torchvision.transforms.Resize((new_height, new_width))
    return resize_transform

def patchify_image(imgs: torch.Tensor, patch_size: int = 4, n_height_tokens=None, n_width_tokens=None) -> torch.Tensor:
    """
    Convert images with a channel dimension into patches.

    Parameters:
        - imgs: Tensor of shape (B, H, W, C) 
            where B is the batch size, H is the height, and W is the width.
        - patch_size: 
            The size of each patch.

    Returns:
        Tensor of shape (B, L, patch_size**2 * C)
            where L is the number of patches (H//patch_size * W//patch_size).
    """

    if n_height_tokens is None and n_width_tokens is None:
        assert imgs.shape[1] == imgs.shape[2] and imgs.shape[1] % patch_size == 0, \
            "Image dimensions must be square and divisible by the patch size."
        h = w = imgs.shape[1] // patch_size
    else:
        h = n_height_tokens
        w = n_width_tokens
    c = imgs.shape[3]
    x = imgs.reshape(shape=(imgs.shape[0], h, patch_size, w, patch_size, c))
    x = torch.einsum('bhpwqc->bhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2 * c))
    return x

def unpatchify_image(x: torch.Tensor, patch_size: int = 4, n_height_tokens=None, n_width_tokens=None) -> torch.Tensor:
    """
    Reconstruct images from their patchified representation.

    Parameters:
        - x: Tensor of shape (B, L, patch_size**2 * C)
            where B is the batch size, L is the number of patches (should be a perfect square),
            and C is the number of channels.
        - patch_size:
            The size of each patch used in patchification.

    Returns:
        Tensor of shape (B, H, W, C)
            where H and W are the original image height and width.
    """
    B, L, patch_dim = x.shape
    if n_height_tokens is None and n_width_tokens is None:
        h = w = int(L ** 0.5)
        assert h * w == L, "The number of patches is not a perfect square."
    else:
        h = n_height_tokens
        w = n_width_tokens
    
    # Determine the number of channels.
    c = patch_dim // (patch_size ** 2)
    
    # Reshape x to separate patches into grid structure.
    # New shape: (B, h, w, patch_size, patch_size, c)
    x = x.reshape(B, h, w, patch_size, patch_size, c)
    
    # Invert the permutation performed in patchify_image.
    # In patchify_image we applied: torch.einsum('bhpwqc->bhwpqc', x)
    # So we revert it by swapping the same dimensions:
    x = torch.einsum('bhwpqc->bhpwqc', x)
    
    # Finally, merge the patch grid back into full image dimensions.
    imgs = x.reshape(B, h * patch_size, w * patch_size, c)
    
    return imgs

def create_images_from_patches(seq, pos, tgt, mask, patches, patch_size):
    """
    Return provided and predicted images based on the input sequences passed in.
    First, pass the inputs through the forward() method to get the predicted `patches`.
    Then combine that with the original `seq` to get the predicted image.

    Parameters:
        seq (torch.Tensor) of size b, t: The input sequence
        pos (torch.Tensor) of size b, t: The positional indices of the sequence
        tgt (torch.Tensor) of size b, t_tgt: The target sequence
        mask (torch.Tensor) of size b, t: The mask of the sequence
        patches (torch.Tensor) of size b, t: The predicted patches
    
    Returns:
        frame0_img_PIL (PIL.Image): The first frame image
        frame1_img_gt_PIL (PIL.Image): The ground truth second frame image
        frame1_img_predicted_PIL (PIL.Image): The predicted second frame image
        frame1_img_gt_with_mask_PIL (PIL.Image): The second frame image with masked patches
    """
    patches_masked = torch.zeros_like(patches)
        
    # seq: [1, 2048, 192]
    # pos: [1, 2048]
    # patches: [1, 512, 192]
    # tgt: 
    n_patches_per_image = seq.size(1) // 2
    n_patches_predicted = patches.size(1)
    
    # Get frame0
    # frame0_seq: [1, 1024, 192]
    # frame1_seq_provided: [1, 717, 192]
    # frame1_seq_predicted: [1, 1024, 192]
    frame0_seq = seq[:, :n_patches_per_image]
    frame1_seq_provided = seq[:, n_patches_per_image:-n_patches_predicted]
    frame1_seq_gt = torch.cat([frame1_seq_provided, tgt], dim=1)
    frame1_seq_predicted = torch.cat([frame1_seq_provided, patches], dim=1)
    frame1_seq_gt_with_mask = torch.cat([frame1_seq_provided, patches_masked], dim=1)
    
    # All shapes are: [1, 1024]
    # frame0_pos: [598, 323, 881,  ..., 682, 278, 348]
    # pos indicates the position of the patches in the original image
    frame0_pos = pos[:, :n_patches_per_image]
    frame1_pos = pos[:, n_patches_per_image:]
    
    # Use pos to rearrange these patches
    # frame0_sort_indices: [1, 1024]
    _, frame0_sort_indices = frame0_pos.sort(dim=1)
    _, frame1_sort_indices = frame1_pos.sort(dim=1)
    
    # All are: [1, 1024, 192]
    frame0_seq_sorted = frame0_seq.gather(1, frame0_sort_indices.unsqueeze(-1).expand(-1, -1, frame0_seq.size(-1)))
    frame1_seq_gt_sorted = frame1_seq_gt.gather(1, frame1_sort_indices.unsqueeze(-1).expand(-1, -1, frame1_seq_gt.size(-1)))
    frame1_seq_predicted_sorted = frame1_seq_predicted.gather(1, frame1_sort_indices.unsqueeze(-1).expand(-1, -1, frame1_seq_predicted.size(-1)))
    frame1_seq_gt_with_mask_sorted = frame1_seq_gt_with_mask.gather(1, frame1_sort_indices.unsqueeze(-1).expand(-1, -1, frame1_seq_gt_with_mask.size(-1)))
    
    # unpatchify image
    # frame0_img: [1, 256, 256, 3]
    frame0_img = unpatchify_image(frame0_seq_sorted, patch_size)
    frame1_img_gt = unpatchify_image(frame1_seq_gt_sorted, patch_size)
    frame1_img_predicted = unpatchify_image(frame1_seq_predicted_sorted, patch_size)
    frame1_img_gt_with_mask = unpatchify_image(frame1_seq_gt_with_mask_sorted, patch_size)
    
    # Convert to PIL Image
    frame0_img_PIL = inv_in_transform(frame0_img.squeeze(0).permute(2, 0, 1))
    frame1_img_gt_PIL = inv_in_transform(frame1_img_gt.squeeze(0).permute(2, 0, 1))
    frame1_img_predicted_PIL = inv_in_transform(frame1_img_predicted.squeeze(0).permute(2, 0, 1))
    frame1_img_gt_with_mask_PIL = inv_in_transform(frame1_img_gt_with_mask.squeeze(0).permute(2, 0, 1))
    
    return frame0_img_PIL, frame1_img_gt_PIL, frame1_img_predicted_PIL, frame1_img_gt_with_mask_PIL


def unpatchify_input_seq(seq, pos, patch_size):
    """
    Return provided and predicted images based on the input sequences passed in.
    First, pass the inputs through the forward() method to get the predicted `patches`.
    Then combine that with the original `seq` to get the predicted image.

    Parameters:
        seq (torch.Tensor) of size b, t: The input sequence
        pos (torch.Tensor) of size b, t: The positional indices of the sequence
        tgt (torch.Tensor) of size b, t_tgt: The target sequence
        mask (torch.Tensor) of size b, t: The mask of the sequence
        patches (torch.Tensor) of size b, t: The predicted patches
    
    Returns:
        frame0_img_PIL (PIL.Image): The first frame image
        frame1_img_gt_PIL (PIL.Image): The ground truth second frame image
        frame1_img_predicted_PIL (PIL.Image): The predicted second frame image
        frame1_img_gt_with_mask_PIL (PIL.Image): The second frame image with masked patches
    """
    
    n_patches_per_image = seq.size(1) // 2

    # Get frame0
    # frame0_seq: [1, 1024, 192]
    # frame1_seq_provided: [1, 717, 192]
    # frame1_seq_predicted: [1, 1024, 192]
    frame0_seq = seq[:, :n_patches_per_image]
    frame1_seq = seq[:, n_patches_per_image:] 
    
    frame0_pos = pos[:, :n_patches_per_image]
    frame1_pos = pos[:, n_patches_per_image:]
    
    # Use pos to rearrange these patches
    # frame0_sort_indices: [1, 1024]
    _, frame0_sort_indices = frame0_pos.sort(dim=1)
    _, frame1_sort_indices = frame1_pos.sort(dim=1)
    
    # All are: [1, 1024, 192]
    frame0_seq_sorted = frame0_seq.gather(1, frame0_sort_indices.unsqueeze(-1).expand(-1, -1, frame0_seq.size(-1)))
    frame1_seq_gt_sorted = frame1_seq.gather(1, frame1_sort_indices.unsqueeze(-1).expand(-1, -1, frame1_seq.size(-1)))

    # unpatchify image
    # frame0_img: [1, 256, 256, 3]
    frame0_img = unpatchify_image(frame0_seq_sorted, patch_size)
    frame1_img_gt = unpatchify_image(frame1_seq_gt_sorted, patch_size)
    
    # Convert to PIL Image
    frame0_img_PIL = inv_in_transform(frame0_img.squeeze(0).permute(2, 0, 1))
    frame1_img_gt_PIL = inv_in_transform(frame1_img_gt.squeeze(0).permute(2, 0, 1))
    
    return frame0_img_PIL, frame1_img_gt_PIL