"""
ZWM Sequence Dataset

This module contains the SequenceDataset class which is used to load and process the ZWM dataset.
The dataset class accepts a list of paths to h5 files containing the dataset, the config of the model
being trained (to determine token type ranges), the mode of the dataset, and other parameters.

Each mode is specified in a method whose name starts with _mode_. Each of the methods retursn a 
tuple of three tensors: the sequence of tokens, the positional indexes of the tokens, and the target.
The seq and pos tensors have to be of the same size while the target tensor has to be less than or equal.
"""

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

from zwm.data.image_processing import (
    get_resize_to_target_area_transform,
    in_transform, inv_in_transform, resize_transform, resize_transform_512, get_random_square_crop_transform, 
    patchify_image
)
from zwm.data.sequence_construction import (
    get_pos_idxs, shuffle_and_trim_values_and_positions
)

# === ZWM imports ===
from zwm.utils.sequence_construction import (
    get_grid_rope_pos_idxs,
    get_rope_pos_idxs,
    shuffle_and_trim_values_and_positions,
)


class PatchSequenceDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            paths,
            model_config,
            mode='fake',
            num_folds=1,
            fold_idx=0,
            frame0_sparsity=0.0,
            frame1_sparsity=0.0,
            frame0_mask_ratio=0.0,
            frame1_mask_ratio=0.9,
            num_flow_patches=150,
            path_ratios=None,
            debug=False,
            max_seq_len=None,
            shuffle_tokens=True,
            mask_ratio=0.9,
            **kwargs
        ):
        super().__init__()

        # check if the mode is valid
        mode_methods = [method.replace("_mode_", "") for method in dir(self) if method.startswith('_mode_')]
        assert mode in mode_methods, f"Invalid mode: {mode}, available modes: {mode_methods}"

        # if paths is as tring turn it to a list of size 1
        if isinstance(paths, str):
            paths = [paths]

        # check if the path ratios are the same size as the paths
        assert path_ratios is None or len(path_ratios) == len(paths), "Path ratios must be the same size as the paths if not None."

        self.files = []
        # fetch all the files in all the paths
        for path_idx, path in enumerate(paths):
            # Grab the names of all the h5 files at the given path and sort them
            all_files = glob.glob(path + '/**/*.mp4', recursive=True)
            all_files.sort()
            # Multiply the all_files list by path_ratios (integers) to increase the number of files
            if path_ratios is not None:
                all_files = all_files * path_ratios[path_idx]

            print(f"Loaded {len(all_files)} files from {path} with scaling factor {path_ratios[path_idx] if path_ratios is not None else 1}")

            # Subsample the files based on the fold index and number of folds
            self.files = self.files + all_files[fold_idx::num_folds]
            if debug:
                print("OBS: Debug mode is on, only using the first 100 files")
                self.files = self.files[:100]

        # set the mode and patameters
        self.mode = mode
        mode_method = getattr(self, f"_mode_{mode}")
        self.mode_method = mode_method
        # TODO: Add separate parameters for sparsity and mask
        self.frame0_sparsity = frame0_sparsity
        self.frame1_sparsity = frame1_sparsity
        self.frame0_mask_ratio = frame0_mask_ratio
        self.frame1_mask_ratio = frame1_mask_ratio
        self.num_flow_patches = num_flow_patches
        self.model_config = model_config
        self.max_seq_len = max_seq_len
        self.shuffle_tokens = shuffle_tokens
        # self.shuffle = shuffle
        self.cache_data = None

        print(f"Dataset has {len(self.files)} files")

        # if mode is fake, make a list of 1024 random file strings so that the dataset has a length of 1048576
        if mode == 'fake':
            self.files = [f"fake_file_{i}.mp4" for i in range(1048576)]

        # # get two samples form the dataset to confirm that they have the same shape
        # # and set that as the shape of the data
        seq0, pos0, tgt0, mask0 = self.__getitem__(0)
        seq1, pos1, tgt1, mask1 = self.__getitem__(1)
        assert seq0.shape == seq1.shape, f"two seq samples have different shapes: {seq0.shape} != {seq1.shape}"
        assert pos0.shape == pos1.shape, f"two pos samples have different shapes: {pos0.shape} != {pos1.shape}"
        assert tgt0.shape == tgt1.shape, f"two tgt samples have different shapes: {tgt0.shape} != {tgt1.shape}"
        assert seq0.shape[0] == pos0.shape[0], f"seq and pos have different sizes: {seq0.shape} != {pos0.shape}"
        assert tgt0.numel() <= seq0.numel(), f"tgt has more elements than seq: {tgt0.numel()} > {seq0.numel()}"
        self.T = seq0.shape[0]

    def __getitem__(self, index):
        # mod index to be within the length of the dataset
        index = index % len(self.files)
        try:
            return self.mode_method(index)
        except Exception as e:
            # WARNING: never disable the print statement below, it is crucial for debugging.
            print(f"Error {e} in mode {self.mode_method.__name__} for file {self.files[index]}")
            return self.__getitem__(np.random.randint(len(self.files)))

    def __len__(self):
        # fake a 100x larger dataset, to avoid dataloader epoch boundary issues
        return len(self.files) * 100


    # ----------------------------------------------------------------------------------------------------- #
    #### DATASET MODE FUNCTIONS ####

    def _mode_fake(self, index):
        # seq, pos, tgt, mask
        n_patches = 2048
        patch_size = 8 * 8 * 3
        seq = torch.zeros((n_patches, patch_size)).float()
        pos = torch.zeros((n_patches,)).long()
        tgt = seq
        mask = pos.float()
        if self.max_seq_len is not None:
            return (
                seq[:self.max_seq_len],
                pos[:self.max_seq_len],
                tgt[:self.max_seq_len],
                mask[:self.max_seq_len]
            )
        else:
            return (
                seq,
                pos,
                tgt,
                mask
            )

    def _mode_zwm_rgb_256(self, index: int):
        """
        RGB ZWM Data Mode: Shuffle RGB patches from two frames and predict the second frame.
        """

        # TODO: Hardcoded
        resolution = 256
        n_input_channels = self.model_config.n_input_channels
        num_patches_per_image = self.model_config.block_size / 2
        frame1_mask_ratio = self.frame1_mask_ratio
        num_masked_patches = int(num_patches_per_image * frame1_mask_ratio)
        

        # Open the h5 file
        vr = VideoReader(self.files[index])
        num_frames = len(vr)

        # 1) random starting frame: ridx0
        # we ensure enough space for a frame_gap up to 15
        ridx0 = np.random.randint(0, num_frames - 16)

        # 2) pick a random frame_gap in [5..15]
        frame_gap = np.random.randint(5, 16)
        ridx1 = ridx0 + frame_gap
        if ridx1 >= num_frames:
            ridx1 = num_frames - 1  # just clamp to the last frame if needed

        # TODO: Check performance
        # 3) decode frames
        frames_batch = vr.get_batch([ridx0, ridx1]).asnumpy()  # shape will be [2, H, W, 3]
        frame0_np, frame1_np = frames_batch[0], frames_batch[1]

        # 4) transform each
        frame0 = resize_transform(Image.fromarray(frame0_np))
        frame1 = resize_transform(Image.fromarray(frame1_np))
        
        height = frame0.size[1]
        width = frame0.size[0]
        random_square_crop_transform = get_random_square_crop_transform(height, width, resolution)

        frame0 = in_transform(random_square_crop_transform(frame0))
        frame1 = in_transform(random_square_crop_transform(frame1))
        
        # frame0: [1, 256, 256, 3]
        frame0 = frame0.unsqueeze(0).permute(0, 2, 3, 1)
        frame1 = frame1.unsqueeze(0).permute(0, 2, 3, 1)

        # 5) Patchify frame0 and frame1
        # patches0: [1, 1024, 192=8*8*3]
        patches0 = patchify_image(frame0, self.model_config.patch_size)
        patches1 = patchify_image(frame1, self.model_config.patch_size)
        
        # 6) Define positional indexes for sequence
        # patches_0_pos_idx: [1, 1024]
        patches_0_pos_idx = get_pos_idxs(patches0, 0)
        patches_1_pos_idx = get_pos_idxs(patches1, num_patches_per_image)
        
        # 7) Shuffle frame0 and frame1, and their positional tokens
        # shuffled_patches0: [1, 1024, 192]
        # shuffled_img_0_pos_idx: [1, 1024]
        shuffled_patches0, shuffled_img_0_pos_idx = shuffle_and_trim_values_and_positions(
            patches0, patches_0_pos_idx, mask=self.frame0_sparsity)
        shuffled_patches1, shuffled_img_1_pos_idx = shuffle_and_trim_values_and_positions(
            patches1, patches_1_pos_idx, mask=self.frame1_sparsity)
            
        # seq: (2048, 192)
        # pos: (2048,)
        seq = torch.cat([
            shuffled_patches0.reshape(-1, shuffled_patches0.shape[-1]), 
            shuffled_patches1.reshape(-1, shuffled_patches1.shape[-1])])
        pos = torch.cat([shuffled_img_0_pos_idx.reshape(-1).long(), shuffled_img_1_pos_idx.reshape(-1).long()])

        # 1. create the flow labels (flow_seq, flow_pos) and append them to the end of seq, pos
        # the flow labels are sparse; flow_pos should correspond to the pixel locations of the flow labels
        # 2. tgt = seq.clone(); tgt = tgt[-num_flow_patches:]
        # 3. mask = torch.zeros_like(pos).float(); mask[-num_flow_patches:] = 1.0
        

        # tgt: (921, 192)
        tgt = seq.clone()
        tgt = tgt[-num_masked_patches:]

        # 8) Create mask for sequence
        # mask: (2048,), tensor([0., 0., 0.,  ..., 1., 1., 1.])
        mask = torch.zeros_like(pos).float()
        mask[-num_masked_patches:] = 1.0
        
        # 9) Zero out masked part of sequence
        seq[-num_masked_patches:] = 0.0
        
        return seq, pos, tgt, mask

    def _mode_zwm_rgb_256_mask_ratio(self, index: int):
        """
        RGB ZWM Data Mode: Shuffle RGB patches from two frames and predict the second frame.
        """

        resolution = 256
        num_patches_per_image = self.model_config.block_size / 2
        frame0_mask_ratio = self.frame0_mask_ratio
        frame1_mask_ratio = self.frame1_mask_ratio
        

        # Open the h5 file
        vr = VideoReader(self.files[index])
        num_frames = len(vr)

        # 1) random starting frame: ridx0
        # we ensure enough space for a frame_gap up to 15
        ridx0 = np.random.randint(0, num_frames - 16)

        # 2) pick a random frame_gap in [5..15]
        frame_gap = np.random.randint(5, 16)
        ridx1 = ridx0 + frame_gap
        if ridx1 >= num_frames:
            ridx1 = num_frames - 1  # just clamp to the last frame if needed

        # TODO: Check performance
        # 3) decode frames
        frames_batch = vr.get_batch([ridx0, ridx1]).asnumpy()  # shape will be [2, H, W, 3]
        frame0_np, frame1_np = frames_batch[0], frames_batch[1]

        # 4) transform each
        frame0 = resize_transform(Image.fromarray(frame0_np))
        frame1 = resize_transform(Image.fromarray(frame1_np))
        
        height = frame0.size[1]
        width = frame0.size[0]
        random_square_crop_transform = get_random_square_crop_transform(height, width, resolution)

        frame0 = in_transform(random_square_crop_transform(frame0))
        frame1 = in_transform(random_square_crop_transform(frame1))
        
        # frame0: [1, 256, 256, 3]
        frame0 = frame0.unsqueeze(0).permute(0, 2, 3, 1)
        frame1 = frame1.unsqueeze(0).permute(0, 2, 3, 1)

        # 5) Patchify frame0 and frame1
        # patches0: [1, 1024, 192=8*8*3]
        patches0 = patchify_image(frame0, self.model_config.patch_size)
        patches1 = patchify_image(frame1, self.model_config.patch_size)
        
        # 6) Define positional indexes for sequence
        # patches_0_pos_idx: [1, 1024]
        patches_0_pos_idx = get_pos_idxs(patches0, 0)
        patches_1_pos_idx = get_pos_idxs(patches1, num_patches_per_image)
        
        # 7) Shuffle frame0 and frame1, and their positional tokens
        # shuffled_patches0: [1, 1024, 192]
        # shuffled_img_0_pos_idx: [1, 1024]
        shuffled_patches0, shuffled_img_0_pos_idx = shuffle_and_trim_values_and_positions(
            patches0, patches_0_pos_idx, mask=self.frame0_sparsity)
        shuffled_patches1, shuffled_img_1_pos_idx = shuffle_and_trim_values_and_positions(
            patches1, patches_1_pos_idx, mask=self.frame1_sparsity)
            
        # seq: (2048, 192)
        # pos: (2048,)
        seq = torch.cat([
            shuffled_patches0.reshape(-1, shuffled_patches0.shape[-1]), 
            shuffled_patches1.reshape(-1, shuffled_patches1.shape[-1])])
        pos = torch.cat([shuffled_img_0_pos_idx.reshape(-1).long(), shuffled_img_1_pos_idx.reshape(-1).long()])

        # masking both frame0 and frame1        
        n0 = shuffled_patches0.shape[1]  # number of kept patches from frame 0
        n1 = shuffled_patches1.shape[1]  # number of kept patches from frame 1
        num_masked_0 = int(round(n0 * frame0_mask_ratio))
        num_masked_1 = int(round(n1 * frame1_mask_ratio))
        mask = torch.zeros(n0 + n1, device=seq.device, dtype=torch.float32)
        
        if num_masked_0 > 0:
            mask[(n0 - num_masked_0):n0] = 1.0

        # Frame 1 block: [n0:n0+n1)
        if num_masked_1 > 0:
            mask[n0 + (n1 - num_masked_1): n0 + n1] = 1.0
        
        # tgt: (921, 192)
        tgt = seq.clone()[mask.bool()]         # shape: (num_masked_0 + num_masked_1, d)

        # 9) Zero out masked part of sequence
        seq[mask.bool()] = 0.0
        
        return seq, pos, tgt, mask


    def _mode_zwm2_rgb_256(self, index: int):
        """
        RGB ZWM Data Mode: Shuffle RGB patches from two frames and predict the second frame.
        """

        # TODO: Hardcoded
        resolution = 256
        n_input_channels = self.model_config.n_input_channels
        num_patches_per_image = self.model_config.block_size / 2
        frame1_mask_ratio = self.frame1_mask_ratio
        num_masked_patches = int(num_patches_per_image * frame1_mask_ratio)
        

        # Open the h5 file
        vr = VideoReader(self.files[index])
        num_frames = len(vr)

        # 1) random starting frame: ridx0
        # we ensure enough space for a frame_gap up to 15
        ridx0 = np.random.randint(0, num_frames - 16)

        # 2) pick a random frame_gap in [5..15]
        frame_gap = np.random.randint(5, 16)
        ridx1 = ridx0 + frame_gap
        if ridx1 >= num_frames:
            ridx1 = num_frames - 1  # just clamp to the last frame if needed

        # TODO: Check performance
        # 3) decode frames
        frames_batch = vr.get_batch([ridx0, ridx1]).asnumpy()  # shape will be [2, H, W, 3]
        frame0_np, frame1_np = frames_batch[0], frames_batch[1]

        # 4) transform each
        frame0 = resize_transform(Image.fromarray(frame0_np))
        frame1 = resize_transform(Image.fromarray(frame1_np))
        
        height = frame0.size[1]
        width = frame0.size[0]
        random_square_crop_transform = get_random_square_crop_transform(height, width, resolution)

        frame0 = in_transform(random_square_crop_transform(frame0))
        frame1 = in_transform(random_square_crop_transform(frame1))
        
        # frame0: [1, 256, 256, 3]
        frame0 = frame0.unsqueeze(0).permute(0, 2, 3, 1)
        frame1 = frame1.unsqueeze(0).permute(0, 2, 3, 1)

        # 5) Patchify frame0 and frame1
        # patches0: [1, 1024, 192=8*8*3]
        patches0 = patchify_image(frame0, self.model_config.patch_size)
        patches1 = patchify_image(frame1, self.model_config.patch_size)
        
        # 6) Define positional indexes for sequence
        # patches_0_dummy: [1, 1024, 1]. First create a dummy tensor to pass to get_rope_pos_idxs
        patches_0_dummy = torch.zeros_like(patches0[:, :, :1])
        patches_1_dummy = torch.zeros_like(patches1[:, :, :1])
        
        # patches_0_pos_idx: [1, 1024, 1, 4]
        patches_0_pos_idx = get_rope_pos_idxs(patches_0_dummy, 0, 0)
        patches_1_pos_idx = get_rope_pos_idxs(patches_1_dummy, frame_gap, 0)
        
        
        # 7) Shuffle frame0 and frame1, and their positional tokens
        # shuffled_patches0: [1, 1024, 192]
        # shuffled_img_0_pos_idx: [1, 1024, 1, 4]
        shuffled_patches0, shuffled_img_0_pos_idx = shuffle_and_trim_values_and_positions(
            patches0, patches_0_pos_idx, mask=self.frame0_sparsity)
        shuffled_patches1, shuffled_img_1_pos_idx = shuffle_and_trim_values_and_positions(
            patches1, patches_1_pos_idx, mask=self.frame1_sparsity)
            
        # seq: (2048, 192)
        # pos: (2048, 4)
        seq = torch.cat([
            shuffled_patches0.reshape(-1, shuffled_patches0.shape[-1]), 
            shuffled_patches1.reshape(-1, shuffled_patches1.shape[-1])])
        pos = torch.cat([
            shuffled_img_0_pos_idx.reshape(-1, shuffled_img_0_pos_idx.shape[-1]), 
            shuffled_img_1_pos_idx.reshape(-1, shuffled_img_1_pos_idx.shape[-1])],
            dim=0
        )

        # tgt: (921, 192)
        tgt = seq.clone()
        tgt = tgt[-num_masked_patches:]

        # 8) Create mask for sequence
        # mask: (2048,), tensor([0., 0., 0.,  ..., 1., 1., 1.])
        # mask is not used in ZWM2 forward()
        mask = torch.zeros(pos.shape[0]).float()
        
        # 9) Handle masked portions of the second frame
        # For seq, the masked patches have their values set to 0
        # For pos, the masked patches have the 4th dimension (channel) set to 1
        seq[-num_masked_patches:] = 0.0
        mask_idx = 1
        pos[-num_masked_patches:, 3] = mask_idx
        
        # In conclusion:
        # seq: (2048, 192)
        # tgt: (921, 192)
        # pos: (2048, 4) for x,y,t,channel
            # ints from 0-31, 0-31, 0 or frame_gap, 0 or 1 (0 for 95%masked)
            # see: get_rope_pos_idxs()
        # mask: (2048,), all 0s, not used in ZWM2 forward()
        return seq, pos, tgt, mask

    def _mode_zwm2_rgb_512(self, index: int):
        """
        RGB ZWM Data Mode: Shuffle RGB patches from two frames and predict the second frame.
        """

        # TODO: Hardcoded
        resolution = 512
        n_input_channels = self.model_config.n_input_channels
        num_patches_per_image = self.model_config.block_size / 2
        frame1_mask_ratio = self.frame1_mask_ratio
        num_masked_patches = int(num_patches_per_image * frame1_mask_ratio)
        

        # Open the h5 file
        vr = VideoReader(self.files[index])
        num_frames = len(vr)

        # 1) random starting frame: ridx0
        # we ensure enough space for a frame_gap up to 15
        ridx0 = np.random.randint(0, num_frames - 16)

        # 2) pick a random frame_gap in [5..15]
        frame_gap = np.random.randint(5, 16)
        ridx1 = ridx0 + frame_gap
        if ridx1 >= num_frames:
            ridx1 = num_frames - 1  # just clamp to the last frame if needed

        # TODO: Check performance
        # 3) decode frames
        frames_batch = vr.get_batch([ridx0, ridx1]).asnumpy()  # shape will be [2, H, W, 3]
        frame0_np, frame1_np = frames_batch[0], frames_batch[1]

        # 4) transform each
        frame0 = resize_transform_512(Image.fromarray(frame0_np))
        frame1 = resize_transform_512(Image.fromarray(frame1_np))
        
        height = frame0.size[1]
        width = frame0.size[0]
        random_square_crop_transform = get_random_square_crop_transform(height, width, resolution)

        frame0 = in_transform(random_square_crop_transform(frame0))
        frame1 = in_transform(random_square_crop_transform(frame1))
        
        # frame0: [1, 512, 512, 3]
        frame0 = frame0.unsqueeze(0).permute(0, 2, 3, 1)
        frame1 = frame1.unsqueeze(0).permute(0, 2, 3, 1)

        # 5) Patchify frame0 and frame1
        # patches0: [1, 4096, 192=8*8*3]
        patches0 = patchify_image(frame0, self.model_config.patch_size)
        patches1 = patchify_image(frame1, self.model_config.patch_size)
        
        # 6) Define positional indexes for sequence
        # patches_0_dummy: [1, 4096, 1]. First create a dummy tensor to pass to get_rope_pos_idxs
        patches_0_dummy = torch.zeros_like(patches0[:, :, :1])
        patches_1_dummy = torch.zeros_like(patches1[:, :, :1])
        
        # patches_0_pos_idx: [1, 4096, 1, 4]
        patches_0_pos_idx = get_rope_pos_idxs(patches_0_dummy, 0, 0)
        patches_1_pos_idx = get_rope_pos_idxs(patches_1_dummy, frame_gap, 0)
        
        
        # 7) Shuffle frame0 and frame1, and their positional tokens
        # shuffled_patches0: [1, 4096, 192]
        # shuffled_img_0_pos_idx: [1, 4096, 1, 4]
        shuffled_patches0, shuffled_img_0_pos_idx = shuffle_and_trim_values_and_positions(
            patches0, patches_0_pos_idx, mask=self.frame0_sparsity)
        shuffled_patches1, shuffled_img_1_pos_idx = shuffle_and_trim_values_and_positions(
            patches1, patches_1_pos_idx, mask=self.frame1_sparsity)
            
        # seq: (8192, 192)
        # pos: (8192, 4)
        seq = torch.cat([
            shuffled_patches0.reshape(-1, shuffled_patches0.shape[-1]), 
            shuffled_patches1.reshape(-1, shuffled_patches1.shape[-1])])
        pos = torch.cat([
            shuffled_img_0_pos_idx.reshape(-1, shuffled_img_0_pos_idx.shape[-1]), 
            shuffled_img_1_pos_idx.reshape(-1, shuffled_img_1_pos_idx.shape[-1])],
            dim=0
        )

        # tgt: (3686, 192)
        tgt = seq.clone()
        tgt = tgt[-num_masked_patches:]

        # 8) Create mask for sequence
        # mask: (2048,), tensor([0., 0., 0.,  ..., 1., 1., 1.])
        # mask is not used in ZWM2 forward()
        mask = torch.zeros(pos.shape[0]).float()
        
        # 9) Handle masked portions of the second frame
        # For seq, the masked patches have their values set to 0
        # For pos, the masked patches have the 4th dimension (channel) set to 1
        seq[-num_masked_patches:] = 0.0
        mask_idx = 1
        pos[-num_masked_patches:, 3] = mask_idx
        
        # In conclusion:
        # seq: (2048, 192)
        # tgt: (921, 192)
        # pos: (2048, 4) for x,y,t,channel
            # ints from 0-31, 0-31, 0 or frame_gap, 0 or 1 (0 for 95%masked)
            # see: get_rope_pos_idxs()
        # mask: (2048,), all 0s, not used in ZWM2 forward()
        return seq, pos, tgt, mask

    def _mode_zwm2_rgb_flexibleHW(self, index: int):
        """
        RGB ZWM Data Mode: Shuffle RGB patches from two frames and predict the second frame.
        """

        # TODO: Hardcoded
        n_input_channels = self.model_config.n_input_channels
        frame1_mask_ratio = self.frame1_mask_ratio
        patch_size = self.model_config.patch_size

        # Open the h5 file
        vr = VideoReader(self.files[index])
        num_frames = len(vr)

        # 1) random starting frame: ridx0
        # we ensure enough space for a frame_gap up to 15
        ridx0 = np.random.randint(0, num_frames - 16)

        # 2) pick a random frame_gap in [5..15]
        frame_gap = np.random.randint(5, 16)
        ridx1 = ridx0 + frame_gap
        if ridx1 >= num_frames:
            ridx1 = num_frames - 1  # just clamp to the last frame if needed

        # TODO: Check performance
        # 3) decode frames
        frames_batch = vr.get_batch([ridx0, ridx1]).asnumpy()  # shape will be [2, H, W, 3]
        frame0_np, frame1_np = frames_batch[0], frames_batch[1]

        # 4) transform each 
        orig_height = frame0_np.shape[0]
        orig_width = frame0_np.shape[1]
        target_area = 256 * 456
        resize_to_target_area_transform = get_resize_to_target_area_transform(orig_height, orig_width, target_area, patch_size)
                
        frame0 = in_transform(resize_to_target_area_transform(Image.fromarray(frame0_np)))
        frame1 = in_transform(resize_to_target_area_transform(Image.fromarray(frame1_np)))

        # frame0: [1, H, W, 3]
        frame0 = frame0.unsqueeze(0).permute(0, 2, 3, 1)
        frame1 = frame1.unsqueeze(0).permute(0, 2, 3, 1)

        n_height_tokens = frame0.size(1) // patch_size
        n_width_tokens = frame0.size(2) // patch_size
        num_patches_per_image = n_height_tokens * n_width_tokens
        num_masked_patches = int(num_patches_per_image * frame1_mask_ratio)

        # 5) Patchify frame0 and frame1
        # patches0: [1, n_patches_height * n_patches_width, 192=8*8*3]
        patches0 = patchify_image(frame0, self.model_config.patch_size, n_height_tokens=n_height_tokens, n_width_tokens=n_width_tokens)
        patches1 = patchify_image(frame1, self.model_config.patch_size, n_height_tokens=n_height_tokens, n_width_tokens=n_width_tokens)
        
        # 6) Define positional indexes for sequence
        # patches_0_dummy: [1, n_patches_height * n_patches_width, 1]. First create a dummy tensor to pass to get_rope_pos_idxs
        # instead, i now want it to be [1, n_patches_height, n_patches_width, 1, 1]
        patches_0_dummy = torch.zeros_like(patches0[:, :, :1])
        patches_1_dummy = torch.zeros_like(patches1[:, :, :1])
        
        # instead of 1, it should be patches_0_dummy.shape[0]
        patches_0_dummy = patches_0_dummy.view(patches_0_dummy.shape[0], n_height_tokens, n_width_tokens, 1)
        patches_1_dummy = patches_1_dummy.view(patches_1_dummy.shape[0], n_height_tokens, n_width_tokens, 1)
        
        # patches_0_pos_idx: [1, 4096, 1, 4]
        patches_0_pos_idx = get_grid_rope_pos_idxs(patches_0_dummy, 0, 0)
        patches_1_pos_idx = get_grid_rope_pos_idxs(patches_1_dummy, frame_gap, 0)
        
        # reshape back to [1, n_patches_height * n_patches_width, 1, 4]
        patches_0_pos_idx = patches_0_pos_idx.view(patches_0_pos_idx.shape[0], num_patches_per_image, 1, patches_0_pos_idx.shape[-1])
        patches_1_pos_idx = patches_1_pos_idx.view(patches_1_pos_idx.shape[0], num_patches_per_image, 1, patches_1_pos_idx.shape[-1])
        
        # 7) Shuffle frame0 and frame1, and their positional tokens
        # shuffled_patches0: [1, 4096, 192]
        # shuffled_img_0_pos_idx: [1, 4096, 1, 4]
        shuffled_patches0, shuffled_img_0_pos_idx = shuffle_and_trim_values_and_positions(
            patches0, patches_0_pos_idx, mask=self.frame0_sparsity)
        shuffled_patches1, shuffled_img_1_pos_idx = shuffle_and_trim_values_and_positions(
            patches1, patches_1_pos_idx, mask=self.frame1_sparsity)
            
        # seq: (8192, 192)
        # pos: (8192, 4)
        seq = torch.cat([
            shuffled_patches0.reshape(-1, shuffled_patches0.shape[-1]), 
            shuffled_patches1.reshape(-1, shuffled_patches1.shape[-1])])
        pos = torch.cat([
            shuffled_img_0_pos_idx.reshape(-1, shuffled_img_0_pos_idx.shape[-1]), 
            shuffled_img_1_pos_idx.reshape(-1, shuffled_img_1_pos_idx.shape[-1])],
            dim=0
        )

        # tgt: (3686, 192)
        tgt = seq.clone()
        # tgt = tgt[-num_masked_patches:] # Commented out so we supervise on all patches, unmasked too

        # 8) Create mask for sequence
        # mask: (2048,), tensor([0., 0., 0.,  ..., 1., 1., 1.])
        # mask is not used in ZWM2 forward()
        mask = torch.zeros(pos.shape[0]).float()
        
        # 9) Handle masked portions of the second frame
        # For seq, the masked patches have their values set to 0
        # For pos, the masked patches have the 4th dimension (channel) set to 1
        seq[-num_masked_patches:] = 0.0
        mask_idx = 1
        pos[:, 3] = 0 # NOTE: Important, because the default value from get_grid_rope_pos_idxs is different
        pos[-num_masked_patches:, 3] = mask_idx
        
        # In conclusion:
        # seq: (2048, 192)
        # tgt: (921, 192)
        # pos: (2048, 4) for x,y,t,channel
            # ints from 0-31, 0-31, 0 or frame_gap, 0 or 1 (0 for 95%masked)
            # see: get_rope_pos_idxs()
        # mask: (2048,), all 0s, not used in ZWM2 forward()
        
        if self.max_seq_len is not None:
            seq = seq[:self.max_seq_len]
            pos = pos[:self.max_seq_len]
            tgt = tgt[:self.max_seq_len]
            mask = mask[:self.max_seq_len]

        # every 1024 samples, print the shapes
        if np.random.randint(0, 1024) == 0:
            print(f"orig_height: {orig_height}, orig_width: {orig_width}")
            print(f"frame0 size: {frame0.size()}, frame1 size: {frame1.size()}")
            print(f"patches_0_pos_idx: {patches_0_pos_idx.shape}, patches_0_pos_idx: {patches_1_pos_idx}")
            print(f"shuffled_patches0: {shuffled_patches0.shape}, shuffled_img_0_pos_idx: {shuffled_img_0_pos_idx}")
            print(f"seq shape {seq.shape}, pos shape {pos.shape}, tgt shape {tgt.shape}, mask shape {mask.shape}")
        
        return seq, pos, tgt, mask

