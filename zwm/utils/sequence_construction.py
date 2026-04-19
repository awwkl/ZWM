"""
Utility function for converting data into seqeunce of tokens and positional indexes.
"""
import math
import torch
import numpy as np
import h5py
from typing import Tuple
from zwm.utils.image_processing import patchify as patchify_func


def get_frame(f: h5py.File, frame_idx: int, patch_size: int = None, key: str = 'rgb', 
              patchify: bool = True, flatten: bool = True) -> torch.LongTensor:
    """
    Get the specified frame from the h5 file and return it as a tensor of patches.

    Parameters:
        f (h5py.File): Opened h5 file pointer.
        frame_idx (int): Index of the frame to get.
        patch_size (int): Size of the patches to create.
        key (str): Key of the frame to get.
    
    Returns:
        patches (torch.LongTensor) of shape 1, N, P:
            Tensor of N patches each of size P in sorted order along the N axis.
    """
    # Get the video frames and camera poses
    frame = torch.from_numpy(f[key][frame_idx].astype(np.int64)).unsqueeze(0)
    if patchify:
        patches = patchify_func(frame, patch_size=patch_size)
    elif flatten:
        # If not patchifying, just reshape the frame to be 1, N, P
        # where N is the number of patches and P is the patch size
        patches = frame.reshape(1, -1, frame.shape[-1])
    else:
        patches = frame
    return patches


def get_frame_flattened_hw(f: h5py.File, frame_idx: int, H: int = 32, W: int = 32, patch_size: int = None, key: str = 'rgb', patchify: bool = True) -> torch.LongTensor:
    """
    Get the specified frame, stored in flattened (H*W, C) format, from the h5 file 
    and return it as a tensor of patches.

    Parameters:
        f (h5py.File): Opened h5 file pointer.
        frame_idx (int): Index of the frame to get.
        H (int): Original height of the image.
        W (int): Original width of the image.
        patch_size (int): Size of the patches to create if patchify is True.
        key (str): Key of the frame to get in the HDF5 file.
        patchify (bool): Whether to patchify the image or return flattened pixels as patches.
    
    Returns:
        patches (torch.LongTensor) of shape 1, N, P:
            Tensor of N patches each of size P. If patchify is True, N is the number
            of patches and P is patch_size*patch_size*C (or similar). If patchify
            is False, N is H*W and P is C.
    """
    # Get the frame data (expected shape: H*W, C)
    frame_data_hw_c = f[key][frame_idx]
    
    # Assert that the flattened dimension matches H * W
    expected_flattened_dim = H * W
    if frame_data_hw_c.shape[0] != expected_flattened_dim:
        raise ValueError(
            f"Frame data first dimension {frame_data_hw_c.shape[0]} does not match "
            f"expected H*W ({expected_flattened_dim}) for frame_idx {frame_idx} and key '{key}'."
        )
    if len(frame_data_hw_c.shape) != 2:
        raise ValueError(
            f"Frame data is expected to be 2D (H*W, C), but got shape {frame_data_hw_c.shape} "
            f"for frame_idx {frame_idx} and key '{key}'."
        )

    # Convert to tensor and add batch dimension (shape: 1, H*W, C)
    frame_tensor_b_hw_c = torch.from_numpy(frame_data_hw_c.astype(np.int64)).unsqueeze(0)
    
    if patchify:
        # Reshape to (1, H, W, C) for patchify_func
        # patchify_func is assumed to take (B, H, W, C) based on get_frame usage
        image_b_h_w_c = frame_tensor_b_hw_c.reshape(1, H, W, frame_tensor_b_hw_c.shape[-1])
        patches = patchify_func(image_b_h_w_c, patch_size=patch_size)
    else:
        # If not patchifying, the frame_tensor_b_hw_c (1, H*W, C) is the desired output.
        # Each of the H*W elements is a "patch" of size C.
        patches = frame_tensor_b_hw_c
        
    return patches


def add_patch_indexes(patches: torch.LongTensor, start_idx: int) -> torch.LongTensor:
    """
    Adds patch indexes to the ordered tensor of patches provided starting at the given index.

    Parameters:
        patches (torch.LongTensor) of shape B, N, P:
            Tensor of N patches each of size P in sorted order along the N axis
        start_idx (int): Starting index for the patch indexes.
    
    Returns:
        patches_with_indexes (torch.LongTensor) of shape B, N, P+1:
            Tensor of N patches each of size P with an additional index at the end
    """
    indexes = torch.arange(start_idx, start_idx + patches.shape[1]).reshape(1, -1, 1).to(patches.device)
    patches_with_indexes = torch.cat([indexes, patches], axis=2)
    return patches_with_indexes


def add_constant_patch_idx(patches: torch.LongTensor, index: int) -> torch.LongTensor:
    """
    Adds patch indexes to the ordered tensor of patches provided starting at the given index.

    Parameters:
        patches (torch.LongTensor) of shape B, N, P:
            Tensor of N patches each of size P in sorted order along the N axis
        start_idx (int): Starting index for the patch indexes.
    
    Returns:
        patches_with_indexes (torch.LongTensor) of shape B, N, P+1:
            Tensor of N patches each of size P with an additional index at the end
    """
    if len(patches.shape) == 3:
        indexes = (index * torch.ones((1, patches.shape[1], 1))).to(patches.device)
    elif len(patches.shape) == 4:
        indexes = (index * torch.ones((1, patches.shape[1], patches.shape[2], 1))).to(patches.device)
    patches_with_indexes = torch.cat([indexes, patches], axis=-1)
    return patches_with_indexes


def get_pos_idxs(tokens: torch.LongTensor, start_idx: int) -> torch.LongTensor:
    """
    Generate positional indexes for the sequence of tokens.

    Parameters:
        tokens (torch.LongTensor) of shape B, N, P:
            Tensor of N patches each of size P in sorted order along the N axis
        start_idx (int): Starting index for the positional indexes.
    
    Returns:
        pos_idx (torch.LongTensor) of shape B, N, P:
            Tensor of positional indexes for the sequence of tokens
    """
    # Create positional indexes for the sequence of tokens
    pos_idx = torch.arange(start_idx, start_idx + tokens.numel()).reshape(tokens.shape).to(tokens.device)
    return pos_idx


def get_rope_pos_idxs(tokens: torch.LongTensor, temporal_idx: int, channel_start_idx: int) -> torch.LongTensor:
    """
    Generate RoPE positional indexes for the sequence of tokens.

    Parameters:
        tokens (torch.LongTensor) of shape B, N, P:
            Tensor of N patches each of size P in sorted order along the N axis
        start_idx (int): Starting index for the positional indexes.
    
    Returns:
        pos_idx (torch.LongTensor) of shape B, N, P:
            Tensor of positional indexes for the sequence of tokens
    """
    # Convert tokens to hxw grid (B, N, P) -> (B, H, W, P)
    grid_tokens = tokens.reshape(
        tokens.shape[0], int(np.sqrt(tokens.shape[1])), int(np.sqrt(tokens.shape[1])), tokens.shape[-1])
    # Create x index for every token based on its position on the H axis
    x_idx = torch.arange(0, grid_tokens.shape[1]).reshape(1, -1, 1, 1).to(tokens.device)
    x_idx = x_idx.repeat(tokens.shape[0], grid_tokens.shape[1], 1, tokens.shape[-1]).reshape(tokens.shape)
    # Create y index for every token based on its position on the W axis
    y_idx = torch.arange(0, grid_tokens.shape[2]).reshape(1, 1, -1, 1).to(tokens.device)
    y_idx = y_idx.repeat(tokens.shape[0], 1, grid_tokens.shape[1], tokens.shape[-1]).reshape(tokens.shape)
    y_idx = y_idx.reshape(grid_tokens.shape).permute(0, 2, 1, 3).reshape(tokens.shape)
    # Create temporal indexes for every token based on the temporal index (same for all tokens)
    temporal_idx = torch.full(tokens.shape, temporal_idx).to(tokens.device)


    # Create channel indexes for every token based on its position on the P axis
    channel_idx = torch.cat([
        torch.tensor([0], device=tokens.device),
        torch.arange(channel_start_idx, channel_start_idx + tokens.shape[-1] - 1, device=tokens.device)
    ]).reshape(1, 1, -1)
    
    
    # channel_idx = torch.arange(channel_start_idx, channel_start_idx + tokens.shape[-1]).reshape(1, 1, -1).to(tokens.device)
    
    
    channel_idx = channel_idx.repeat(tokens.shape[0], tokens.shape[1], 1).reshape(tokens.shape)
    # Merge all indexes into a single tensor of shape (B, N, P, 4)
    pos_idx = torch.stack([x_idx, y_idx, temporal_idx, channel_idx], axis=3)
    # Convert to long tensor
    pos_idx = pos_idx.long()

    return pos_idx


def get_grid_rope_pos_idxs(tokens: torch.Tensor, temporal_idx: int, channel_start_idx: int) -> torch.LongTensor:
    """
    Generate integer positional indices (y, x, t, c) for RoPE-style embeddings.

    Args:
        tokens: Tensor of shape (B, N, P) or (B, H, W, P).
        temporal_idx: Integer time index assigned to all tokens.
        channel_start_idx: Starting channel index (c-axis).

    Returns:
        LongTensor of shape:
            - (B, N, P, 4) if input was (B, N, P)
            - (B, H, W, P, 4) if input was (B, H, W, P)
        The last axis is ordered as (y, x, t, c).
    """
    device = tokens.device
    dtype = torch.long

    if tokens.dim() == 3:
        B, N, P = tokens.shape
        H = W = math.isqrt(N)
        if H * W != N:
            raise ValueError(f"N={N} is not a perfect square; pass a 4D (B,H,W,P) tensor or specify H,W explicitly.")

        # Compute x/y directly in flattened space to avoid ordering mistakes.
        idx = torch.arange(N, device=device, dtype=dtype)
        y = (idx // W).view(1, N, 1).expand(B, N, P)
        x = (idx %  W).view(1, N, 1).expand(B, N, P)

        t = torch.full((B, N, P), int(temporal_idx), device=device, dtype=dtype)

        ch = torch.arange(channel_start_idx - 1, channel_start_idx + P - 1, device=device, dtype=dtype)
        if P > 1:
            ch[0] = ch[1]  # pointer shares the first non-pointer's channel embedding as it is pointing to it
        c = ch.view(1, 1, P).expand(B, N, P)

        return torch.stack((y, x, t, c), dim=-1).long()

    elif tokens.dim() == 4:
        B, H, W, P = tokens.shape

        y = torch.arange(H, device=device, dtype=dtype).view(1, H, 1, 1).expand(B, H, W, P)
        x = torch.arange(W, device=device, dtype=dtype).view(1, 1, W, 1).expand(B, H, W, P)
        t = torch.full((B, H, W, P), int(temporal_idx), device=device, dtype=dtype)

        ch = torch.arange(channel_start_idx - 1, channel_start_idx + P - 1, device=device, dtype=dtype)
        if P > 1:
            ch[0] = ch[1]  # pointer shares the first non-pointer's channel embedding as it is pointing to it
        c = ch.view(1, 1, 1, P).expand(B, H, W, P)

        return torch.stack((y, x, t, c), dim=-1).long()

    else:
        raise ValueError(f"Unsupported tokens shape: {tuple(tokens.shape)}")


def shuffle_and_trim_values_and_positions(
        tokens: torch.LongTensor, positions: torch.LongTensor, 
        mask: float = 0.0, shuffle: bool = True, shuffle_order: np.ndarray = None,
        num_patches_to_keep: int = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Shuffle the tokens and positions along the N (1st) axis and remove mask amount of the tokens.

    Parameters:
        tokens (torch.LongTensor) of shape B, N, P:
            Tensor of N patches each of size P in sorted order along the N axis
        positions (torch.LongTensor) of shape B, N, P:
            Tensor of positional indexes for the sequence of tokens
        mask (float): Amount of the tokens to remove
        shuffle (bool): Whether to shuffle the tokens or not
        shuffle_order (np.ndarray): Order to shuffle the tokens in
        num_patches_to_keep (int): Number of patches to keep
    Returns:
        shuffled_tokens (torch.LongTensor) of shape B, N, P:
            Shuffled tensor of N patches each of size P in sorted order along the N axis
        shuffled_positions (torch.LongTensor) of shape B, N, P:
            Shuffled tensor of positional indexes for the sequence of tokens
    """
    # Shuffle patches on the 1st axis, as well as positions, if shuffle is True
    if shuffle_order is None:
        if shuffle:
            shuffle_order = np.random.permutation(tokens.shape[1])
        else:
            shuffle_order = np.arange(tokens.shape[1])
    else:
        assert len(shuffle_order) == tokens.shape[1], \
            (f"Provided shuffle order does not account for all tokens in sequence. "
             f"Expected {tokens.shape[1]} indices, got {len(shuffle_order)}.")

    # Remove mask amount of the patches
    if num_patches_to_keep is not None:
        shuffle_order = shuffle_order[:int(num_patches_to_keep)]
    else:
        shuffle_order = shuffle_order[:int(shuffle_order.shape[0] * (1-mask))]
    shuffled_tokens = tokens[:, shuffle_order]
    shuffled_positions = positions[:, shuffle_order]
    return shuffled_tokens, shuffled_positions


# Deprecated alias for suppress_targets
def supress_targets(tokens: torch.LongTensor, range: Tuple[int, int]) -> torch.LongTensor:
    return suppress_targets(tokens, range)


def suppress_targets(tokens: torch.LongTensor, range: Tuple[int, int]) -> torch.LongTensor:
    """
    Suppress the targets in the given range by setting them to -1

    Parameters:
        tokens (torch.LongTensor) of shape B, N, P:
            Tensor of N patches each of size P in sorted order along the N axis
        range (Tuple[int, int]): Range of values to suppress
    
    Returns:
        tokens (torch.LongTensor) of shape B, N, P:
            Tensor of N patches each of size P in sorted order along the N axis with suppressed values
    """
    tokens[(range[0] <= tokens) & (tokens < range[1])] = -1
    return tokens


def suppress_duplicated_targets(targets: torch.LongTensor, max_duplicates: int = 10) -> torch.LongTensor:
    """
    For each unique value in targets, keeps at most `max_duplicates` of them and suppresses the rest by setting them to -1.
    The kept targets are chosen randomly.

    Parameters:
        targets (torch.LongTensor): 1D tensor of targets.
        max_duplicates (int): Maximum number of identical targets to keep.

    Returns:
        torch.LongTensor: a new tensor with extra targets suppressed.
    """
    new_targets = targets.clone()
    unique_values, counts = torch.unique(targets, return_counts=True)
    
    for val, count in zip(unique_values, counts):
        if val == -1:
            continue
        if count > max_duplicates:
            indices = (targets == val).nonzero(as_tuple=True)[0]
            # shuffle indices
            shuffled_indices = indices[torch.randperm(len(indices))]
            indices_to_suppress = shuffled_indices[max_duplicates:]
            new_targets[indices_to_suppress] = -1
    
    return new_targets

def sample_frames_and_positions(video_length, num_wanted_video_frames,
                                frame_gap_mode, last_frame_gap_mode=None,
                                flow_length=None, num_frame_positions=20, flip_last_two_probability=0.1):

    if num_wanted_video_frames == 1:
        return [np.random.randint(0, video_length)], [np.random.randint(0, num_frame_positions)]

    if num_wanted_video_frames == 2:
        if last_frame_gap_mode == "1to5":
            # (num_wanted_video_frames - 1) * max_frame_gap + 1 = video_length
            max_frame_gap = min(5, ((video_length - 1) // (num_wanted_video_frames - 1)) + 1)
            frame_gap = np.random.randint(1, max_frame_gap + 1)
            # start_idx + frame_gap <= video_length - 1 (index can't be video_length)
            start_idx = np.random.randint(0, video_length - frame_gap)
            start_pos = np.random.randint(0, num_frame_positions - frame_gap)
            frame_idxs = [start_idx, start_idx + frame_gap]
            frame_positions = [start_pos, start_pos + frame_gap]
        elif last_frame_gap_mode == "1to20":
            max_frame_gap = min(20, ((video_length - 1) // (num_wanted_video_frames - 1)) + 1)
            frame_gap = np.random.randint(1, max_frame_gap + 1)
            start_idx = np.random.randint(0, video_length - frame_gap)
            start_pos = np.random.randint(0, num_frame_positions - frame_gap)
            frame_idxs = [start_idx, start_idx + frame_gap]
            frame_positions = [start_pos, start_pos + frame_gap]
        elif last_frame_gap_mode == "flow_gap":
            frame_gap = video_length - flow_length
            last_frame_gap = frame_gap  # Set last_frame_gap for use in flip logic
            start_idx = np.random.randint(0, video_length - frame_gap)
            start_pos = np.random.randint(0, num_frame_positions - frame_gap)
            frame_idxs = [start_idx, start_idx + frame_gap]
            frame_positions = [start_pos, start_pos + frame_gap]
        elif last_frame_gap_mode == "anywhere":
            # even if unsorted, we still want to let the model know the time order (like it's forward or backward)
            frame_idxs = np.random.choice(video_length, size=num_wanted_video_frames, replace=False)
            frame_idxs.sort()
            frame_positions = np.random.choice(num_frame_positions, size=num_wanted_video_frames, replace=False)
            frame_positions.sort()
            random_shuffle_idx = np.random.permutation(num_wanted_video_frames)
            frame_idxs = frame_idxs[random_shuffle_idx]
            frame_positions = frame_positions[random_shuffle_idx]
        else:
            raise ValueError(f"Unsupported last_frame_gap_mode: {last_frame_gap_mode} in num_wanted_video_frames == 2")

    if num_wanted_video_frames > 2:
        if last_frame_gap_mode == "1to5" and frame_gap_mode == "1to5":
            max_frame_gap = min(5, ((video_length - 1) // (num_wanted_video_frames - 1)) + 1)
            frame_gap = np.random.randint(1, max_frame_gap + 1)
            start_idx = np.random.randint(0, video_length - frame_gap * (num_wanted_video_frames - 1))
            start_pos = np.random.randint(0, num_frame_positions - frame_gap * (num_wanted_video_frames - 1))
            frame_idxs = [start_idx + i * frame_gap for i in range(num_wanted_video_frames)]
            frame_positions = [start_pos + i * frame_gap for i in range(num_wanted_video_frames)]
        elif last_frame_gap_mode == "1to20" and frame_gap_mode == "1to20":
            max_frame_gap = min(20, ((video_length - 1) // (num_wanted_video_frames - 1)) + 1)
            frame_gap = np.random.randint(1, max_frame_gap + 1)
            start_idx = np.random.randint(0, video_length - frame_gap * (num_wanted_video_frames - 1))
            start_pos = np.random.randint(0, num_frame_positions - frame_gap * (num_wanted_video_frames - 1))
            frame_idxs = [start_idx + i * frame_gap for i in range(num_wanted_video_frames)]
            frame_positions = [start_pos + i * frame_gap for i in range(num_wanted_video_frames)]
        elif last_frame_gap_mode == "flow_gap" and frame_gap_mode == "flow_gap":
            frame_gap = video_length - flow_length
            last_frame_gap = frame_gap  # Set last_frame_gap for use in flip logic
            start_idx = np.random.randint(0, video_length - frame_gap * (num_wanted_video_frames - 1))
            start_pos = np.random.randint(0, num_frame_positions - frame_gap * (num_wanted_video_frames - 1))
            frame_idxs = [start_idx + i * frame_gap for i in range(num_wanted_video_frames)]
            frame_positions = [start_pos + i * frame_gap for i in range(num_wanted_video_frames)]
        elif last_frame_gap_mode == "anywhere" and frame_gap_mode == "anywhere":
            frame_idxs = np.random.choice(video_length, size=num_wanted_video_frames, replace=False)
            frame_idxs.sort()
            frame_positions = np.random.choice(num_frame_positions, size=num_wanted_video_frames, replace=False)
            frame_positions.sort()
            random_shuffle_idx = np.random.permutation(num_wanted_video_frames)
            frame_idxs = frame_idxs[random_shuffle_idx]
            frame_positions = frame_positions[random_shuffle_idx]
        elif last_frame_gap_mode == "flow_gap" and frame_gap_mode == "1to5":
            last_frame_gap = video_length - flow_length
            max_frame_gap = min(5, ((flow_length - 1) // (num_wanted_video_frames - 2)) + 1)
            frame_gap = np.random.randint(1, max_frame_gap + 1)
            start_idx = np.random.randint(0, video_length - frame_gap * (num_wanted_video_frames - 2) - last_frame_gap)
            start_pos = np.random.randint(0, num_frame_positions - frame_gap * (num_wanted_video_frames - 2) - last_frame_gap)
            frame_idxs = [start_idx + i * frame_gap for i in range(num_wanted_video_frames - 1)]
            frame_positions = [start_pos + i * frame_gap for i in range(num_wanted_video_frames - 1)]
            frame_idxs.append(frame_idxs[-1] + last_frame_gap)
            frame_positions.append(frame_positions[-1] + last_frame_gap)
        elif last_frame_gap_mode == "flow_gap" and frame_gap_mode == "1to20":
            last_frame_gap = video_length - flow_length
            max_frame_gap = min(20, ((flow_length - 1) // (num_wanted_video_frames - 2)) + 1)
            frame_gap = np.random.randint(1, max_frame_gap + 1)
            start_idx = np.random.randint(0, video_length - frame_gap * (num_wanted_video_frames - 2) - last_frame_gap)
            start_pos = np.random.randint(0, num_frame_positions - frame_gap * (num_wanted_video_frames - 2) - last_frame_gap)
            frame_idxs = [start_idx + i * frame_gap for i in range(num_wanted_video_frames - 1)]
            frame_positions = [start_pos + i * frame_gap for i in range(num_wanted_video_frames - 1)]
            frame_idxs.append(frame_idxs[-1] + last_frame_gap)
            frame_positions.append(frame_positions[-1] + last_frame_gap)
        elif last_frame_gap_mode == "flow_gap" and frame_gap_mode == "anywhere":
            last_frame_gap = video_length - flow_length
            start_idx = np.random.randint(0, video_length - last_frame_gap)
            start_pos = np.random.randint(0, num_frame_positions - last_frame_gap)
            last_two_frames_idxs = [start_idx + i * last_frame_gap for i in range(2)]
            last_two_frames_positions = [start_pos + i * last_frame_gap for i in range(2)]
            all_idxs = np.arange(video_length - last_frame_gap)
            valid_idxs = np.setdiff1d(all_idxs, last_two_frames_idxs)
            all_positions = np.arange(num_frame_positions - last_frame_gap)
            valid_positions = np.setdiff1d(all_positions, last_two_frames_positions)
            remaining_frames_idxs = np.random.choice(valid_idxs, size=num_wanted_video_frames - 2, replace=False)
            remaining_frames_positions = np.random.choice(valid_positions, size=num_wanted_video_frames - 2, replace=False)
            remaining_frames_idxs.sort()
            remaining_frames_positions.sort()
            random_shuffle_idx = np.random.permutation(num_wanted_video_frames-2)
            remaining_frames_idxs = remaining_frames_idxs[random_shuffle_idx]
            remaining_frames_positions = remaining_frames_positions[random_shuffle_idx]
            frame_idxs = remaining_frames_idxs + last_two_frames_idxs
            frame_positions = remaining_frames_positions + last_two_frames_positions
        else:
            raise ValueError(f"Unsupported last_frame_gap_mode: {last_frame_gap_mode} and frame_gap_mode: {frame_gap_mode} in num_wanted_video_frames > 2")

    if np.random.random() < flip_last_two_probability:
        if last_frame_gap_mode == "flow_gap":
            if num_wanted_video_frames <= 2:
                pass
            else:
                last_three_frames_idxs = frame_idxs[-3:]
                last_three_frames_positions = frame_positions[-3:]
                new_last_three_frame_idxs = [last_three_frames_idxs[2], last_three_frames_idxs[0], last_three_frames_idxs[0] + last_frame_gap]
                new_last_three_frame_positions = [last_three_frames_positions[2], last_three_frames_positions[0], last_three_frames_positions[0] + last_frame_gap]
                frame_idxs = frame_idxs[:-3] + new_last_three_frame_idxs
                frame_positions = frame_positions[:-3] + new_last_three_frame_positions
        else:
            last_two_frames_idxs = frame_idxs[-2:]
            last_two_frames_positions = frame_positions[-2:]
            flipped_last_two_frames_idxs = last_two_frames_idxs[::-1]
            flipped_last_two_frames_positions = last_two_frames_positions[::-1]
            frame_idxs = frame_idxs[:-2] + flipped_last_two_frames_idxs
            frame_positions = frame_positions[:-2] + flipped_last_two_frames_positions
    return frame_idxs, frame_positions
