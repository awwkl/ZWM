import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
from typing import Tuple, Union, List, Dict
import tqdm
import random

from zwm.data.image_processing import create_images_from_patches, patchify_image, unpatchify_image, get_resize_to_target_area_transform
from zwm.data.sequence_construction import (
    get_pos_idxs, shuffle_and_trim_values_and_positions
)

from zwm.utils.sequence_construction import get_grid_rope_pos_idxs, get_rope_pos_idxs
from zwm.utils.model_wrapper import ModelFactory

class ZWMPredictor:
    
    def __init__(self, model_name: str, device: str = 'cuda'):
        
        try:
            self.model = ModelFactory().load_model(model_name).to(torch.bfloat16).to(device).eval()
        except:
            self.model = ModelFactory().load_model_from_checkpoint(model_name).to(torch.bfloat16).to(device).eval()
        
        self.ctx = torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.bfloat16)
        
        # Set parameters
        self.device = device
        
        self.model.config.dropout = 0.0
        
        # Set transforms
        self.in_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.inv_in_transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
            torchvision.transforms.Lambda(lambda x: torch.clamp(x, 0, 1)), 
            torchvision.transforms.ToPILImage()
        ])
        
    def resize_crop_transform(self, image, resolution):
        resize_crop_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resolution),
            torchvision.transforms.CenterCrop(resolution),
        ])
        return resize_crop_transform(image)
        
    @torch.no_grad()
    def hypothetical_prediction(
        self,
        frame0: Union[Image.Image, np.ndarray],
        move_points: np.array,
        patch_size_move_mult,
        src_idxs: list[int] = [],
        dst_idxs: list[int] = [],
        hold_idxs: list[int] = [],
        frame_gap: int = 10,
        **kwargs,
    ) -> Dict:
        if isinstance(frame0, np.ndarray):
            frame0 = Image.fromarray(frame0)
        
        resolution = self.model.config.resolution
        frame0_PIL = self.resize_crop_transform(frame0, resolution)
        
        patch_size = self.model.config.patch_size
        num_patches_per_image = (resolution // patch_size) ** 2
        
        if move_points is not None:
            # points: np.array of shape (N, 4)
            assert move_points.shape[1] == 4, "Points should be of shape (N, 4), where each point is (x1, y1, x2, y2)"
            # patch_indices: (N, 2), [[494, 530], [526, 562], [495, 531], [527, 563]]
            patch_indices = self.convert_move_points_into_patch_indices(move_points, patch_size, patch_size_move_mult, resolution)
            start_indices = [patch_indices[i][0] for i in range(len(patch_indices))]
            end_indices = [patch_indices[i][1] for i in range(len(patch_indices))]
        else:
            start_indices = src_idxs + hold_idxs
            end_indices = dst_idxs + hold_idxs

        frame0 = self.in_transform(self.resize_crop_transform(frame0, resolution)) # [3, 256, 256]
        frame0 = frame0.unsqueeze(0).permute(0, 2, 3, 1) # [1, 256, 256, 3]
        patches0 = patchify_image(frame0, self.model.config.patch_size) # [1, 1024, 192=8*8*3]
        
        # use start_indices to get the patches from patches0 to be used for patches1
        patches1 = torch.zeros_like(patches0) # [1, 1024, 192]
        patches1[:, end_indices, :] = patches0[:, start_indices, :] # Copy the patches from patches0 to patches1
        
        seq = torch.cat([
            patches0.reshape(-1, patches0.shape[-1]), 
            patches1.reshape(-1, patches1.shape[-1])])
        
        if 'ZWM2' in self.model.config.model_class:
            # patches_0_dummy: [1, 1024, 1]. First create a dummy tensor to pass to get_rope_pos_idxs
            patches_0_dummy = torch.zeros_like(patches0[:, :, :1])
            patches_1_dummy = torch.zeros_like(patches1[:, :, :1])
            
            # patches_0_pos_idx: [1, 1024, 1, 4]
            patches_0_pos_idx = get_rope_pos_idxs(patches_0_dummy, 0, 0)
            patches_1_pos_idx = get_rope_pos_idxs(patches_1_dummy, frame_gap, 0)

            # pos: (2048, 4)
            pos = torch.cat([
                patches_0_pos_idx.reshape(-1, patches_0_pos_idx.shape[-1]), 
                patches_1_pos_idx.reshape(-1, patches_1_pos_idx.shape[-1])],
                dim=0
            )
            mask_value = 1
            mask_indices = [i for i in range(num_patches_per_image) if i not in end_indices]
            pos_mask_indices = [m + num_patches_per_image for m in mask_indices]
            pos[pos_mask_indices, 3] = mask_value
            
            # mask is not used for ZWM2
            mask = torch.zeros(pos.shape[0]).float()
        else:
            patches_0_pos_idx = get_pos_idxs(patches0, 0) # [[   0,    1,    2,  ..., 1021, 1022, 1023]]
            patches_1_pos_idx = get_pos_idxs(patches1, num_patches_per_image) # [[1024, 1025, 1026,  ..., 2045, 2046, 2047]]

            pos = torch.cat([patches_0_pos_idx.reshape(-1).long(), patches_1_pos_idx.reshape(-1).long()])
            
            mask = torch.ones_like(pos).float()
            mask[:num_patches_per_image] = 0.0 # unmask frame0
            indices_to_unmask = [num_patches_per_image + idx for idx in end_indices]
            mask[indices_to_unmask] = 0.0
        
        with self.ctx:
            patches, _ = self.model(seq.unsqueeze(0).to(self.device), pos.unsqueeze(0).to(self.device), tgt=None, mask=mask.unsqueeze(0).to(self.device))
        patches = patches.detach().cpu().float()
        
        frame1_pred_patches = patches[:, num_patches_per_image:] # [1, 1024, 192]
        frame1_pred_patches[:, end_indices, :] = patches1[:, end_indices, :] # Copy the supplied patches from patches1 to frame1_pred_patches
        frame1_pred_img = unpatchify_image(frame1_pred_patches, self.model.config.patch_size) # [1, 256, 256, 3]
        frame1_pred_PIL = self.inv_in_transform(frame1_pred_img.squeeze(0).permute(2, 0, 1))

        frame0_transformed_patches = patches[:, :num_patches_per_image] # [1, 1024, 192]
        frame0_transformed_img = unpatchify_image(frame0_transformed_patches, self.model.config.patch_size) # [1, 256, 256, 3]
        frame0_transformed_PIL = self.inv_in_transform(frame0_transformed_img.squeeze(0).permute(2, 0, 1))
        
        frame1_masked_patches = patches1
        frame1_masked_img = unpatchify_image(frame1_masked_patches, self.model.config.patch_size)
        frame1_masked_PIL = self.inv_in_transform(frame1_masked_img.squeeze(0).permute(2, 0, 1))
        
        
        return {
            "frame0_pil": frame0_PIL,
            "frame0_transformed_pil": frame0_transformed_PIL,
            "frame1_pred_pil": frame1_pred_PIL,
            "frame1_masked_pil": frame1_masked_PIL,
        }
        
    def factual_prediction(
        self,
        frame0: Union[Image.Image, np.ndarray],
        frame1: Union[Image.Image, np.ndarray],
        frame_gap: int,
        mask_ratio: float = 0.9,
        unmask_indices: List[int] = None,
        seed: int = 0,
        return_attention: bool = False,
        **kwargs,
    ):
        self._set_seed(seed)
        num_patches_per_image = int(self.model.config.block_size / 2)
        resolution = self.model.config.resolution
    
        if isinstance(frame0, np.ndarray):
            frame0 = Image.fromarray(frame0)
        if isinstance(frame1, np.ndarray):
            frame1 = Image.fromarray(frame1)
    
        frame0_PIL = self.resize_crop_transform(frame0, resolution)
        frame1_PIL = self.resize_crop_transform(frame1, resolution)
        
        if unmask_indices is None:
            num_unmasked_patches = int(num_patches_per_image * (1 - mask_ratio))
            unmask_indices = np.random.choice(num_patches_per_image, num_unmasked_patches, replace=False).tolist()
        
        frame0 = self.in_transform(frame0_PIL)
        frame1 = self.in_transform(frame1_PIL)
        
        # frame0: [1, 256, 256, 3]
        frame0 = frame0.unsqueeze(0).permute(0, 2, 3, 1)
        frame1 = frame1.unsqueeze(0).permute(0, 2, 3, 1)

        # Patchify frame0 and frame1
        # patches0: [1, 1024, 192=8*8*3]
        patches0 = patchify_image(frame0, self.model.config.patch_size)
        patches1 = patchify_image(frame1, self.model.config.patch_size)

        # Mask out the patches in frame1
        mask_indices = [i for i in list(range(num_patches_per_image)) if i not in unmask_indices]
        patches1[:, mask_indices, :] = 0.0 
        
        # seq: (2048, 192)
        seq = torch.cat([
            patches0.reshape(-1, patches0.shape[-1]), 
            patches1.reshape(-1, patches1.shape[-1])])
        
        if 'ZWM2' in self.model.config.model_class:
            # patches_0_dummy: [1, 1024, 1]. First create a dummy tensor to pass to get_rope_pos_idxs
            patches_0_dummy = torch.zeros_like(patches0[:, :, :1])
            patches_1_dummy = torch.zeros_like(patches1[:, :, :1])
            
            # patches_0_pos_idx: [1, 1024, 1, 4]
            patches_0_pos_idx = get_rope_pos_idxs(patches_0_dummy, 0, 0)
            patches_1_pos_idx = get_rope_pos_idxs(patches_1_dummy, frame_gap, 0)

            # pos: (2048, 4)
            pos = torch.cat([
                patches_0_pos_idx.reshape(-1, patches_0_pos_idx.shape[-1]), 
                patches_1_pos_idx.reshape(-1, patches_1_pos_idx.shape[-1])],
                dim=0
            )
            mask_value = 1
            pos_mask_indices = [m + num_patches_per_image for m in mask_indices]
            pos[pos_mask_indices, 3] = mask_value
            
            # mask is not used for ZWM2
            mask = torch.zeros(pos.shape[0]).float()
            
        else:
            # Define positional indexes for sequence
            # patches_0_pos_idx: [1, 1024]
            patches_0_pos_idx = get_pos_idxs(patches0, 0)
            patches_1_pos_idx = get_pos_idxs(patches1, num_patches_per_image)
            
            # pos: (2048,)
            pos = torch.cat([patches_0_pos_idx.reshape(-1).long(), patches_1_pos_idx.reshape(-1).long()])

            mask = torch.ones_like(pos).float()
            mask[:num_patches_per_image] = 0.0 # unmask frame0
            indices_to_unmask = [num_patches_per_image + idx for idx in unmask_indices]
            mask[indices_to_unmask] = 0.0
        
        with self.ctx:
            att_list = None
            if return_attention:
                patches, _, att_list = self.model.forward_and_return_attention(seq.unsqueeze(0).to(self.device), pos.unsqueeze(0).to(self.device), tgt=None, mask=mask.unsqueeze(0).to(self.device))
            else:
                patches, _ = self.model(seq.unsqueeze(0).to(self.device), pos.unsqueeze(0).to(self.device), tgt=None, mask=mask.unsqueeze(0).to(self.device))
        patches = patches.detach().cpu().float()

        frame1_pred_patches = patches[:, num_patches_per_image:] # [1, 1024, 192]

        # GT - frame 1 with mask, patches1 already has the masked portions zero-ed out
        frame1_with_mask_img = unpatchify_image(patches1, self.model.config.patch_size) # [1, 256, 256, 3]
        frame1_with_mask_PIL = self.inv_in_transform(frame1_with_mask_img.squeeze(0).permute(2, 0, 1))

        # Frame 0 raw
        frame0_pred_patches = patches[:, :num_patches_per_image] # [1, 1024, 192]
        frame0_pred_raw_img = unpatchify_image(frame0_pred_patches, self.model.config.patch_size) # [1, 256, 256, 3]
        frame0_pred_raw_rgb = frame0_pred_raw_img.squeeze(0).permute(2, 0, 1).unsqueeze(0) # [1, 3, 256, 256]
        frame0_pred_raw_PIL = self.inv_in_transform(frame0_pred_raw_rgb[0])

        # Pred Frame 1
        frame1_pred_patches_raw = frame1_pred_patches.clone()  # [1, 1024, 192]
        frame1_pred_raw_img = unpatchify_image(frame1_pred_patches_raw, self.model.config.patch_size) # [1, 256, 256, 3]
        frame1_pred_raw_rgb = frame1_pred_raw_img.squeeze(0).permute(2, 0, 1).unsqueeze(0) # [1, 3, 256, 256]
        frame1_pred_raw_PIL = self.inv_in_transform(frame1_pred_raw_rgb[0])

        # Pred Frame 1 where unmasked patches are zero-ed out
        frame1_pred_patches[:, unmask_indices, :] = 0
        frame1_pred_img = unpatchify_image(frame1_pred_patches, self.model.config.patch_size) # [1, 256, 256, 3]
        frame1_pred_rgb = frame1_pred_img[0].permute(2, 0, 1).unsqueeze(0) # [1, 3, 256, 256]
        frame1_pred_PIL = self.inv_in_transform(frame1_pred_rgb[0])
        
        # Pred Frame 1 where unmasked patches are replaced with the original patches from GT
        frame1_pred_patches[:, unmask_indices, :] = patches1[:, unmask_indices, :] # Copy the supplied patches from patches1 to frame1_pred_patches
        frame1_pred_img_unmasked = unpatchify_image(frame1_pred_patches, self.model.config.patch_size) # [1, 256, 256, 3]
        frame1_pred_rgb_unmasked = frame1_pred_img_unmasked[0].permute(2, 0, 1).unsqueeze(0)
        frame1_pred_PIL_unmasked = self.inv_in_transform(frame1_pred_rgb_unmasked[0])
        
        return {
            "frame0_pil": frame0_PIL,
            "frame1_pil": frame1_PIL,
            "frame1_with_mask_pil": frame1_with_mask_PIL,
            "frame1_pred_rgb": frame1_pred_rgb.detach().cpu(),
            "frame1_pred_pil": frame1_pred_PIL,
            "frame1_pred_pil_unmasked": frame1_pred_PIL_unmasked,
            'frame0_pred_raw_PIL': frame0_pred_raw_PIL,
            'frame1_pred_raw_PIL': frame1_pred_raw_PIL,
            'att_list': att_list,
        }
        
    def factual_prediction_ZWM2_flexibleHW(
        self,
        frame0: Union[Image.Image, np.ndarray],
        frame1: Union[Image.Image, np.ndarray],
        frame_gap: int,
        mask_ratio: float = 0.9,
        unmask_indices: List[int] = None,
        seed: int = 0,
        return_attention: bool = False,
        **kwargs,
    ):
        self._set_seed(seed)
        patch_size = self.model.config.patch_size
        resolution = self.model.config.resolution
    
        if isinstance(frame0, np.ndarray):
            frame0 = Image.fromarray(frame0)
        if isinstance(frame1, np.ndarray):
            frame1 = Image.fromarray(frame1)

        orig_height, orig_width = frame0.size[1], frame0.size[0]
        target_area = 256 * 456
        resize_to_target_area_transform = get_resize_to_target_area_transform(orig_height, orig_width, target_area, patch_size)
    
        frame0_PIL = resize_to_target_area_transform(frame0)
        frame1_PIL = resize_to_target_area_transform(frame1)
        
        frame0 = self.in_transform(frame0_PIL)
        frame1 = self.in_transform(frame1_PIL)
        
        # frame0: [1, 256, 256, 3]
        frame0 = frame0.unsqueeze(0).permute(0, 2, 3, 1)
        frame1 = frame1.unsqueeze(0).permute(0, 2, 3, 1)

        # Patchify frame0 and frame1
        # patches0: [1, 1024, 192=8*8*3]
        n_height_tokens = frame0.size(1) // patch_size
        n_width_tokens = frame0.size(2) // patch_size
        num_patches_per_image = n_height_tokens * n_width_tokens
        
        patches0 = patchify_image(frame0, self.model.config.patch_size, n_height_tokens=n_height_tokens, n_width_tokens=n_width_tokens)
        patches1 = patchify_image(frame1, self.model.config.patch_size, n_height_tokens=n_height_tokens, n_width_tokens=n_width_tokens)

        if unmask_indices is None:
            num_unmasked_patches = int(num_patches_per_image * (1 - mask_ratio))
            unmask_indices = np.random.choice(num_patches_per_image, num_unmasked_patches, replace=False).tolist()
            
        # Mask out the patches in frame1
        mask_indices = [i for i in list(range(num_patches_per_image)) if i not in unmask_indices]
        patches1[:, mask_indices, :] = 0.0 
        
        # seq: (2048, 192)
        seq = torch.cat([
            patches0.reshape(-1, patches0.shape[-1]), 
            patches1.reshape(-1, patches1.shape[-1])])
        
        # patches_0_dummy: [1, 1024, 1]. First create a dummy tensor to pass to get_rope_pos_idxs
        patches_0_dummy = torch.zeros_like(patches0[:, :, :1])
        patches_1_dummy = torch.zeros_like(patches1[:, :, :1])
        
        patches_0_dummy = patches_0_dummy.view(patches_0_dummy.shape[0], n_height_tokens, n_width_tokens, 1)
        patches_1_dummy = patches_1_dummy.view(patches_1_dummy.shape[0], n_height_tokens, n_width_tokens, 1)
            
        # patches_0_pos_idx: [1, 1024, 1, 4]
        patches_0_pos_idx = get_grid_rope_pos_idxs(patches_0_dummy, 0, 0)
        patches_1_pos_idx = get_grid_rope_pos_idxs(patches_1_dummy, frame_gap, 0)
        
        # reshape back to [1, n_patches_height * n_patches_width, 1, 4]
        patches_0_pos_idx = patches_0_pos_idx.view(patches_0_pos_idx.shape[0], num_patches_per_image, 1, patches_0_pos_idx.shape[-1])
        patches_1_pos_idx = patches_1_pos_idx.view(patches_1_pos_idx.shape[0], num_patches_per_image, 1, patches_1_pos_idx.shape[-1])

        # pos: (2048, 4)
        pos = torch.cat([
            patches_0_pos_idx.reshape(-1, patches_0_pos_idx.shape[-1]), 
            patches_1_pos_idx.reshape(-1, patches_1_pos_idx.shape[-1])],
            dim=0
        )
        mask_value = 1
        pos_mask_indices = [m + num_patches_per_image for m in mask_indices]
        pos[:, 3] = 0 # NOTE: Important, because the default value from get_grid_rope_pos_idxs is different
        pos[pos_mask_indices, 3] = mask_value
        
        # mask is not used for ZWM2
        mask = torch.zeros(pos.shape[0]).float()
            
        with self.ctx:
            att_list = None
            if return_attention:
                patches, _, att_list = self.model.forward_and_return_attention(seq.unsqueeze(0).to(self.device), pos.unsqueeze(0).to(self.device), tgt=None, mask=mask.unsqueeze(0).to(self.device))
            else:
                patches, _ = self.model(seq.unsqueeze(0).to(self.device), pos.unsqueeze(0).to(self.device), tgt=None, mask=mask.unsqueeze(0).to(self.device))
        patches = patches.detach().cpu().float()

        frame1_pred_patches = patches[:, num_patches_per_image:] # [1, 1024, 192]

        # GT - frame 1 with mask, patches1 already has the masked portions zero-ed out
        frame1_with_mask_img = unpatchify_image(patches1, self.model.config.patch_size, n_height_tokens=n_height_tokens, n_width_tokens=n_width_tokens) # [1, 256, 256, 3]
        frame1_with_mask_PIL = self.inv_in_transform(frame1_with_mask_img.squeeze(0).permute(2, 0, 1))

        # Frame 0 raw
        frame0_pred_patches = patches[:, :num_patches_per_image] # [1, 1024, 192]
        frame0_pred_raw_img = unpatchify_image(frame0_pred_patches, self.model.config.patch_size, n_height_tokens=n_height_tokens, n_width_tokens=n_width_tokens) # [1, 256, 256, 3]
        frame0_pred_raw_rgb = frame0_pred_raw_img.squeeze(0).permute(2, 0, 1).unsqueeze(0) # [1, 3, 256, 256]
        frame0_pred_raw_PIL = self.inv_in_transform(frame0_pred_raw_rgb[0])

        # Pred Frame 1
        frame1_pred_patches_raw = frame1_pred_patches.clone()  # [1, 1024, 192]
        frame1_pred_raw_img = unpatchify_image(frame1_pred_patches_raw, self.model.config.patch_size, n_height_tokens=n_height_tokens, n_width_tokens=n_width_tokens) # [1, 256, 256, 3]
        frame1_pred_raw_rgb = frame1_pred_raw_img.squeeze(0).permute(2, 0, 1).unsqueeze(0) # [1, 3, 256, 256]
        frame1_pred_raw_PIL = self.inv_in_transform(frame1_pred_raw_rgb[0])

        # Pred Frame 1 where unmasked patches are zero-ed out
        frame1_pred_patches[:, unmask_indices, :] = 0
        frame1_pred_img = unpatchify_image(frame1_pred_patches, self.model.config.patch_size, n_height_tokens=n_height_tokens, n_width_tokens=n_width_tokens) # [1, 256, 256, 3]
        frame1_pred_rgb = frame1_pred_img[0].permute(2, 0, 1).unsqueeze(0) # [1, 3, 256, 256]
        frame1_pred_PIL = self.inv_in_transform(frame1_pred_rgb[0])
        
        # Pred Frame 1 where unmasked patches are replaced with the original patches from GT
        frame1_pred_patches[:, unmask_indices, :] = patches1[:, unmask_indices, :] # Copy the supplied patches from patches1 to frame1_pred_patches
        frame1_pred_img_unmasked = unpatchify_image(frame1_pred_patches, self.model.config.patch_size, n_height_tokens=n_height_tokens, n_width_tokens=n_width_tokens) # [1, 256, 256, 3]
        frame1_pred_rgb_unmasked = frame1_pred_img_unmasked[0].permute(2, 0, 1).unsqueeze(0)
        frame1_pred_PIL_unmasked = self.inv_in_transform(frame1_pred_rgb_unmasked[0])
        
        return {
            "frame0_pil": frame0_PIL,
            "frame1_pil": frame1_PIL,
            "frame1_with_mask_pil": frame1_with_mask_PIL,
            "frame1_pred_rgb": frame1_pred_rgb.detach().cpu(),
            "frame1_pred_pil": frame1_pred_PIL,
            "frame1_pred_pil_unmasked": frame1_pred_PIL_unmasked,
            'frame0_pred_raw_PIL': frame0_pred_raw_PIL,
            'frame1_pred_raw_PIL': frame1_pred_raw_PIL,
            'att_list': att_list,
        }
        
    def single_image_forward(
        self,
        frame0: Union[Image.Image, np.ndarray],
        frame_gap: int,
        mask_ratio: float = 0.0,
        unmask_indices: List[int] = None,
        seed: int = 0,
        return_activations: bool = False,
        **kwargs,
    ):
        self._set_seed(seed)
        num_patches_per_image = int(self.model.config.block_size / 2)
        resolution = self.model.config.resolution
    
        if isinstance(frame0, np.ndarray):
            frame0 = Image.fromarray(frame0)
    
        frame0_PIL = self.resize_crop_transform(frame0, resolution)
        
        if unmask_indices is None:
            num_unmasked_patches = int(num_patches_per_image * (1 - mask_ratio))
            unmask_indices = np.random.choice(num_patches_per_image, num_unmasked_patches, replace=False).tolist()
        
        frame0 = self.in_transform(frame0_PIL)
        
        # frame0: [1, 256, 256, 3]
        frame0 = frame0.unsqueeze(0).permute(0, 2, 3, 1)

        # Patchify frame0 and frame1
        # patches0: [1, 1024, 192=8*8*3]
        patches0 = patchify_image(frame0, self.model.config.patch_size)

        # Mask out the patches in frame1
        mask_indices = [i for i in list(range(num_patches_per_image)) if i not in unmask_indices]

        # seq: (2048, 192)
        seq = patches0.reshape(-1, patches0.shape[-1])
        
        if 'ZWM2' in self.model.config.model_class:
            pass
            raise NotImplementedError("ZWM2 is not implemented")

        else:
            # Define positional indexes for sequence
            # patches_0_pos_idx: [1, 1024]
            patches_0_pos_idx = get_pos_idxs(patches0, 0)
            
            # pos: (2048,)
            pos = patches_0_pos_idx.reshape(-1).long()

            mask = torch.ones_like(pos).float()
            mask[:num_patches_per_image] = 0.0 # unmask frame0
        
        activations = None
        with self.ctx:
            if return_activations:
                patches, _, activations = self.model.forward_and_return_activations(seq.unsqueeze(0).to(self.device), pos.unsqueeze(0).to(self.device), tgt=None, mask=mask.unsqueeze(0).to(self.device))
            else:
                patches, _ = self.model(seq.unsqueeze(0).to(self.device), pos.unsqueeze(0).to(self.device), tgt=None, mask=mask.unsqueeze(0).to(self.device))
        patches = patches.detach().cpu().float()

        # Frame 0 raw
        frame0_pred_patches = patches[:, :num_patches_per_image] # [1, 1024, 192]
        frame0_pred_raw_img = unpatchify_image(frame0_pred_patches, self.model.config.patch_size) # [1, 256, 256, 3]
        frame0_pred_raw_rgb = frame0_pred_raw_img.squeeze(0).permute(2, 0, 1).unsqueeze(0) # [1, 3, 256, 256]
        frame0_pred_raw_PIL = self.inv_in_transform(frame0_pred_raw_rgb[0])

        return {
            "frame0_pil": frame0_PIL,
            'frame0_pred_raw_PIL': frame0_pred_raw_PIL,
            'activations': activations,
        }
        
        
    def convert_move_points_into_patch_indices(
        self,
        move_points,
        patch_size: int,
        patch_size_move_mult: int,
        resolution,
    ):
        patch_indices = []
        num_patches_per_side = resolution // patch_size
        
        for idx, (x1, y1, x2, y2) in enumerate(move_points):
            x1, y1, x2, y2 = round(x1 / patch_size), round(y1 / patch_size), round(x2 / patch_size), round(y2 / patch_size)
            
            # if patch_size_move_mult is 1, create a list [(0, 0)]
            # if patch_size_move_mult is 2, create a list [(0, 0), (0, 1), (1, 0), (1, 1)]
            patch_size_move_list = [(i, j) for i in range(patch_size_move_mult) for j in range(patch_size_move_mult)]
            for i, j in patch_size_move_list:
                x1_i, y1_i, x2_i, y2_i = x1 + i, y1 + j, x2 + i, y2 + j
                patch_idx_1 = int(y1_i * num_patches_per_side + x1_i)
                patch_idx_2 = int(y2_i * num_patches_per_side + x2_i)
                
                # check if the points are within the image bounds
                if 0 <= x1_i < num_patches_per_side and 0 <= y1_i < num_patches_per_side and \
                   0 <= x2_i < num_patches_per_side and 0 <= y2_i < num_patches_per_side:
                    patch_indices.append([patch_idx_1, patch_idx_2])
            
        return patch_indices
        
    def _set_seed(self, seed: int):
        """
        Set the seed for reproducibility.

        Parameters:
            seed: int, the seed to set
        """
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)