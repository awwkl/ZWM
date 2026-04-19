"""
Visualization utilities
"""
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange
try:
    from moviepy.editor import ImageSequenceClip
except ImportError as e:
    from moviepy import ImageSequenceClip

def visualize_rgb(rgb, ax, fig=None):
    # if tensor, convert to numpy
    if type(rgb) == torch.Tensor:
        rgb = rgb[0].cpu().detach().numpy().transpose(1, 2, 0)
    ax.imshow(rgb)
    # ax.axis('off')
    ax.set_title("RGB Image")
    return ax


def fig_to_img(fig: plt.Figure) -> Image.Image:
    """
    Converts a matplotlib figure to a PIL image

    Parameters:
        fig: matplotlib figure

    Returns:
        img: PIL image
    """
    plt.tight_layout()
    fig.canvas.draw()
    img = Image.frombytes('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
    # convert image to RGB
    img = img.convert('RGB')
    return img


def frames_to_video(frames, video_path, fps=30//4, high_quality=False):
    """
    Writes frames to a video file in either .mp4 or .webm format based on the file extension

    Parameters:
        frames: list of numpy arrays, list of frames
        video_path: str, path to the video file
        fps: int, frames per second for the output video
        high_quality: bool, whether to use high quality encoding settings. If True, uses higher bitrate and better quality settings for webm encoding.

    Returns:

        None
    """
    import cv2
    
    if video_path.endswith('.mp4'):
        # Get frame size
        height, width, _ = np.array(frames[0]).shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for frame in frames:
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

        out.release()
    elif video_path.endswith('.webm'):
        # Convert frames to numpy arrays if they are not already
        frames = [np.array(frame) for frame in frames]

        # Create a video clip from the frames
        clip = ImageSequenceClip(frames, fps=fps)

        # Write the clip to a file in .webm format
        if high_quality:
            # High quality settings
            clip.write_videofile(video_path, 
                               codec='libvpx',
                               bitrate='8000k',  # Higher bitrate
                               ffmpeg_params=[
                                   '-crf', '4',  # Lower CRF = higher quality (range 0-63)
                                   '-b:v', '8000k',  # Video bitrate
                                   '-quality', 'best'  # Best quality
                               ])
        else:
            # Default quality settings
            clip.write_videofile(video_path, codec='libvpx')
    else:
        raise ValueError("Unsupported file extension. Only .mp4 and .webm are supported.")


def unpatchify(labels):
    # Define the input tensor
    B = labels.shape[0]  # batch size
    N_patches = int(np.sqrt(labels.shape[1]))  # number of patches along each dimension
    patch_size = int(np.sqrt(labels.shape[2] / 3))  # patch size along each dimension
    channels = 3  # number of channels

    rec_imgs = rearrange(labels, 'b n (p c) -> b n p c', c=3)
    # Notice: To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
    rec_imgs = rearrange(rec_imgs,
                         'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)',
                         p0=1,
                         p1=patch_size,
                         p2=patch_size,
                         h=N_patches,
                         w=N_patches)

    return rec_imgs


def normalize_img(img):
    """
    Applies imagenet normalization to an image

    Parameters:
        img: torch.Tensor, image

    Returns:
        img: torch.Tensor, normalized image
    """
    MEAN = torch.from_numpy(np.array((0.485, 0.456, 0.406))[None, :, None, None, None]).to(img.device)
    STD = torch.from_numpy(np.array((0.229, 0.224, 0.225))[None, :, None, None, None]).to(img.device)

    img = (img - MEAN) / STD

    return img


def un_normalize_img(img):
    """
    Applies inverse imagenet normalization to an image

    Parameters:
        img: torch.Tensor, image

    Returns:
        img: torch.Tensor, unnormalized image
    """
    MEAN = torch.from_numpy(np.array((0.485, 0.456, 0.406))[None, None, :, None, None]).to(img.device).half()
    STD = torch.from_numpy(np.array((0.229, 0.224, 0.225))[None, None, :, None, None]).to(img.device).half()

    img = img * STD + MEAN

    return img


def kp_to_xy(kp):
    """
    Converts keypoint indexes (of range 784) to x, y coordinates on a 224x224 image

    Parameters:
        kp: torch.Tensor, keypoints

    Returns:
        xy: torch.Tensor, x, y coordinates
    """
    x = (kp % 28 + 0.5) * 8
    y = (kp // 28 + 0.5) * 8
    return torch.stack((x.int(), y.int()), dim=-1)


def mask_out_image(img, mask_idxs, patch_size=16, color=0):
    """
    Modifies a PIL Image by blacking out patches specified by mask_idxs.

    The patches are indexed from left to right, top to bottom, using patch_size x patch_size patches.
    """
    # start by converting the image to a numpy array
    img = np.array(img)
    grid_size = img.shape[0]//patch_size
    for idx in mask_idxs:
        x_start = (idx % grid_size) * (patch_size)
        y_start = (idx // grid_size) * (patch_size)
        img[y_start:y_start + patch_size, x_start:x_start + patch_size] = color
    # convert the numpy array back to a PIL Image
    img = Image.fromarray(img)
    return img


def draw_rgb(img, rgb_color, mask_idxs, patch_size=16, color=0):
    """
    Modifies a PIL Image by blacking out patches specified by mask_idxs.

    The patches are indexed from left to right, top to bottom, using patch_size x patch_size patches.
    """
    # start by converting the image to a numpy array
    img = np.zeros_like(img)
    grid_size = 256//patch_size
    rgb_color = np.array(rgb_color)
    for ct, idx in enumerate(mask_idxs):
        x_start = (idx % grid_size) * (patch_size)
        y_start = (idx // grid_size) * (patch_size)
        xx_rgb = ct // 4
        yy_rgb = ct % 4
        color = rgb_color[xx_rgb:xx_rgb+patch_size, yy_rgb:yy_rgb+patch_size]
        img[y_start:y_start + patch_size, x_start:x_start + patch_size] = color
    # convert the numpy array back to a PIL Image
    img = Image.fromarray(img)
    return img
