import torchvision
from PIL import Image, ImageOps
import torch
from einops import rearrange
import numpy as np
try:
    from moviepy.editor import ImageSequenceClip
except ImportError as e:
    from moviepy import ImageSequenceClip


def center_crop_image(image, img_size=256):
    """
    Crops an image to a square centered around the center of the image

    Parameters:
        image: numpy array, the image
        img_size: int, size of the cropped image

    Returns:
        cropped_image: numpy array, the cropped image
    """
    import cv2
    
    if image.shape[0] > image.shape[1]:
        cropped_image = image[image.shape[0] // 2 - image.shape[1] // 2: image.shape[0] // 2 + image.shape[1] // 2, :]
    else:
        cropped_image = image[:, image.shape[1] // 2 - image.shape[0] // 2: image.shape[1] // 2 + image.shape[0] // 2]
    cropped_image = cv2.resize(cropped_image, (img_size, img_size))
    return cropped_image

def video_to_frames(video_path, frame_skip=None, target_fps=None, img_size=(256, 256), 
                    center_crop=False, round_to_multiple=16):
    """
    Extracts frames from a video file and resizes them.
    
    Parameters:
        video_path: str, path to the video file
        frame_skip: int, number of frames to skip between each frame (if None, calculated from target_fps)
        target_fps: float, target fps for frame extraction (overrides frame_skip if provided)
        img_size: tuple or int
            - If tuple: (width, height) for target dimensions
            - If int: resize shortest side to this value, maintaining aspect ratio
            - If None: no resizing applied
        center_crop: bool, whether to center crop (only used if img_size is a tuple)
        round_to_multiple: int or False
            - If int > 0: rounds dimensions to nearest multiple of this value
            - If 0/False/None: no rounding applied
            Default is 16 for compatibility with models requiring specific dimension multiples
    
    Returns:
        frames: list of numpy arrays (H, W, C) in RGB format
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    
    # Get video FPS and calculate frame_skip if target_fps is provided
    if target_fps is not None:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(round(video_fps / target_fps)))
    elif frame_skip is None:
        frame_skip = 1
    
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Handle resizing
        if isinstance(img_size, int):
            # Resize shortest side to img_size, maintain aspect ratio
            h, w = frame.shape[:2]
            if h < w:
                new_h = img_size
                new_w = int(w * (img_size / h))
            else:
                new_w = img_size
                new_h = int(h * (img_size / w))
            
            # Round to multiple if specified
            if round_to_multiple:
                new_w = round(new_w / round_to_multiple) * round_to_multiple
                new_h = round(new_h / round_to_multiple) * round_to_multiple
                # Ensure dimensions are at least the rounding multiple
                new_w = max(new_w, round_to_multiple)
                new_h = max(new_h, round_to_multiple)
            
            frame = cv2.resize(frame, (new_w, new_h))
            
        elif img_size is not None:
            if center_crop:
                h, w = frame.shape[:2]
                target_w, target_h = img_size
                
                # Resize so the smaller dimension matches target
                if h / w > target_h / target_w:
                    # Height is the limiting factor, scale by width
                    new_w = target_w
                    new_h = int(h * (target_w / w))
                else:
                    # Width is the limiting factor, scale by height
                    new_h = target_h
                    new_w = int(w * (target_h / h))
                
                frame = cv2.resize(frame, (new_w, new_h))
                
                # Center crop to exact dimensions
                h, w = frame.shape[:2]
                start_y = (h - target_h) // 2
                start_x = (w - target_w) // 2
                frame = frame[start_y:start_y + target_h, start_x:start_x + target_w]
            else:
                # Direct resize to target dimensions
                new_w, new_h = img_size
                
                # Round to multiple if specified
                if round_to_multiple:
                    new_w = round(new_w / round_to_multiple) * round_to_multiple
                    new_h = round(new_h / round_to_multiple) * round_to_multiple
                    # Ensure dimensions are at least the rounding multiple
                    new_w = max(new_w, round_to_multiple)
                    new_h = max(new_h, round_to_multiple)
                
                frame = cv2.resize(frame, (new_w, new_h))
        
        frames.append(frame)
        
        # Skip frames
        for _ in range(frame_skip):
            cap.read()
    
    cap.release()
    return frames

def frames_to_video(frames, video_path, fps=30//4, high_quality=False):
    """
    Writes frames to a video file in either .mp4 or .webm format based on the file extension

    Parameters:
        frames: list of numpy arrays, list of frames
        video_path: str, path to the video file

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

def load_centercropped_image_tensor_from_path(image_path, img_size=256):
    image = Image.open(image_path).convert('RGB')
    return load_centercropped_image_tensor_from_pil(image, img_size)

def load_centercropped_image_tensor_from_pil(image_pil, img_size=256):
    if isinstance(img_size, tuple) or isinstance(img_size, list):
        img_size = img_size[0]
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.CenterCrop(img_size),
        torchvision.transforms.ToTensor(),
    ])
    image_tensor = transform(image_pil)
    return image_tensor

def load_image(image_path, img_size=(256, 256), as_tensor=False, round_to_multiple=None):
    """
    Loads an image from a file and resizes it.
    
    Parameters:
        image_path: str, path to the image file
        img_size: tuple or int
            - If tuple: exact (width, height) to resize to
            - If int: resize shortest side to this value, maintaining aspect ratio
        as_tensor: bool, if True returns a PyTorch tensor instead of numpy array
        round_to_multiple: int or False
            - If int > 0: rounds dimensions to nearest multiple of this value
            - If 0/False/None: no rounding applied
            Default is 16 for compatibility with models requiring specific dimension multiples
    
    Returns:
        image: PIL Image, numpy array, or torch.Tensor depending on as_tensor flag
    """
    image = Image.open(image_path).convert('RGB')
    
    # Calculate target dimensions
    if isinstance(img_size, int):
        # Resize shortest side to img_size, maintain aspect ratio
        w, h = image.size
        if w < h:
            new_w = img_size
            new_h = int(h * (img_size / w))
        else:
            new_h = img_size
            new_w = int(w * (img_size / h))
    else:
        # Use exact dimensions provided
        new_w, new_h = img_size
    
    # Round to multiple if specified
    if round_to_multiple:
        new_w = round(new_w / round_to_multiple) * round_to_multiple
        new_h = round(new_h / round_to_multiple) * round_to_multiple
        # Ensure dimensions are at least the rounding multiple
        new_w = max(new_w, round_to_multiple)
        new_h = max(new_h, round_to_multiple)
    
    # Resize image
    image = image.resize((new_w, new_h))
    
    # Convert to tensor if requested
    if as_tensor:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        image = transform(image)
    
    return image

def load_image_center_crop(image_path, img_size=(256, 256), as_tensor=False):
    """
    Loads an image from a file and resizes it to 224x224

    Parameters:
        image_path: str, path to the image file

    Returns:
        image: numpy array, the image
    """
    image = Image.open(image_path)
    image = ImageOps.fit(image, img_size)
    image = image.convert('RGB')
    
    if as_tensor:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        image = transform(image)

    return image


def convert_to_16bit_color(image):
    """
    Converts a 24-bit RGB image to a 16-bit color image.
    
    Args:
    image (numpy array): Input image in 24-bit RGB format.

    Returns:
    numpy array: Converted image in 16-bit color format.
    """
    # Convert imagr from uint8 to uint16
    image = image.astype('uint16')

    # Extract the RGB channels
    r = image[:, :, 2] >> 3
    g = image[:, :, 1] >> 2
    b = image[:, :, 0] >> 3
    
    # Combine the channels into 16-bit color
    rgb16 = (r << 11) | (g << 5) | b
    
    return rgb16


def convert_from_16bit_color(image_16bit):
    """
    Converts a 16-bit color image back to a 24-bit RGB image.
    
    Args:
    image_16bit (numpy array): Input image in 16-bit color format.

    Returns:
    numpy array: Converted image in 24-bit RGB format.
    """
    # Extract the 5-bit Red, 6-bit Green, and 5-bit Blue channels
    r = (image_16bit >> 11) & 0x1F
    g = (image_16bit >> 5) & 0x3F
    b = image_16bit & 0x1F
    
    # Convert the channels back to 8-bit by left-shifting and scaling
    r = (r << 3) | (r >> 2)
    g = (g << 2) | (g >> 4)
    b = (b << 3) | (b >> 2)
    
    # Combine the channels into a 24-bit RGB image
    image_rgb = np.stack((b, g, r), axis=-1)
    
    return image_rgb.astype('uint8')


def patchify(imgs: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    """
    Convert images with no channel dimension into patches.

    Parameters:
        - imgs: Tensor of shape (B, H, W) 
            where B is the batch size, H is the height, and W is the width.
        - patch_size: 
            The size of each patch.

    Returns:
        Tensor of shape (B, L, patch_size**2)
            where L is the number of patches (H//patch_size * W//patch_size).
    """
    assert imgs.shape[1] == imgs.shape[2] and imgs.shape[1] % patch_size == 0, \
        "Image dimensions must be square and divisible by the patch size."

    h = w = imgs.shape[1] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], h, patch_size, w, patch_size))
    x = torch.einsum('bhpwq->bhwpq', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2))
    return x


def patchify_logits(logits: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    """
    Convert logits of image codes into patches.

    Parameters:
        - logits: Tensor of shape (B, H, W, D) 
            where B is the batch size, H is the height, W is the width, and D is the number of classes.
        - patch_size: 
            The size of each patch.
    
    Returns:
        Tensor of shape (B, L, patch_size**2, D)
            where L is the number of patches (H//patch_size * W//patch_size).
    """
    assert logits.shape[1] == logits.shape[2] and logits.shape[1] % patch_size == 0, \
        "Image dimensions must be square and divisible by the patch size."

    h = w = logits.shape[1] // patch_size
    x = logits.reshape(shape=(logits.shape[0], h, patch_size, w, patch_size, logits.shape[3]))
    x = torch.einsum('bhpwqd->bhwpqd', x)
    x = x.reshape(shape=(logits.shape[0], h * w, patch_size**2, logits.shape[3]))
    return x


def unpatchify(patches):
    """
    Reconstruct images without color channel from patches.

    Parameters:
        - patches: Tensor of shape (B, L, patch_size**2)
            where B is the batch size, L is the number of patches, and patch_size**2 is the size of each patch.

    Returns:
        Tensor of shape (B, H, W) where H is the height and W is the width of the reconstructed image.
    """
    N_patches = int(np.sqrt(patches.shape[1]))  # number of patches along each dimension
    patch_size = int(np.sqrt(patches.shape[2]))  # patch size along each dimension

    rec_imgs = rearrange(patches, 'b (h w) (p0 p1) -> b h w p0 p1', h=N_patches, w=N_patches, p0=patch_size, p1=patch_size)
    rec_imgs = rearrange(rec_imgs, 'b h w p0 p1 -> b (h p0) (w p1)')
    
    return rec_imgs


def unpatchify_logits(patches):
    """
    Reconstruct images without color channel from patches.

    Parameters:
        - patches: Tensor of shape (B, L, patch_size**2, D)
            where B is the batch size, L is the number of patches, and patch_size**2 is the size of each patch.

    Returns:
        Tensor of shape (B, H, W, D) where H is the height and W is the width of the reconstructed image.
    """
    N_patches = int(np.sqrt(patches.shape[1]))  # number of patches along each dimension
    patch_size = int(np.sqrt(patches.shape[2]))  # patch size along each dimension

    rec_imgs = rearrange(patches, 'b (h w) (p0 p1) d -> b h w p0 p1 d', h=N_patches, w=N_patches, p0=patch_size,
                         p1=patch_size)
    rec_imgs = rearrange(rec_imgs, 'b h w p0 p1 d -> b (h p0) (w p1) d')

    return rec_imgs


def patchify_rgb(imgs: torch.Tensor, patch_size: int = 4, norm=True) -> torch.Tensor:
    """
    Convert images with color channel into patches.

    Parameters:
        - imgs: Tensor of shape (B, C, H, W) 
            where B is the batch size, C is the number of channels, H is the height, and W is the width.
        - patch_size: 
            The size of each patch.

    Returns:
        Tensor of shape (B, L, patch_size**2 * C)
            where L is the number of patches (H//patch_size * W//patch_size).
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0, \
        "Image dimensions must be square and divisible by the patch size."

    if norm:
        # Normalization
        MEAN = torch.from_numpy(np.array((0.485, 0.456, 0.406))[None, :, None, None]
                                ).to(imgs.device).to(imgs.dtype)
        STD = torch.from_numpy(np.array((0.229, 0.224, 0.225))[None, :, None, None]
                               ).to(imgs.device).to(imgs.dtype)
        imgs = (imgs -  MEAN)/STD

    h = w = imgs.shape[2] // patch_size
    c = imgs.shape[1]
    x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, patch_size, w, patch_size))
    x = torch.einsum('bchpwq->bhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2 * c))

    return x


def unpatchify_rgb(patches, norm=True):
    # Define the input tensor
    B = patches.shape[0]  # batch size
    N_patches = int(np.sqrt(patches.shape[1]))  # number of patches along each dimension
    patch_size = int(np.sqrt(patches.shape[2]/3))  # patch size along each dimension
    channels = 3  # number of channels

    rec_imgs = rearrange(patches, 'b n (p c) -> b n p c', c=3)
    # Notice: To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
    rec_imgs = rearrange(rec_imgs,
                             'b (h w) (p q) c -> b c (h p) (w q)',
                             p=patch_size,
                             q=patch_size,
                             h=N_patches,
                             w=N_patches)
    
    if norm:
        # Inverse normalization
        MEAN = torch.from_numpy(np.array((0.485, 0.456, 0.406))[None, :, None, None]
                                ).to(patches.device).to(patches.dtype)
        STD = torch.from_numpy(np.array((0.229, 0.224, 0.225))[None, :, None, None]
                               ).to(patches.device).to(patches.dtype)
        rec_imgs = rec_imgs * STD + MEAN

    return rec_imgs


def visualize_prob_distribution_entropy(probs, entropy=None, im_size=32, vocab_size=67585, skip = 2, max_y=0.05, save_path=None):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    """
    Visualizes probs and entropy on a grid of subplots, where each subplot shows:
    - probs as a line plot representing probability distributions.
    - Entropy as a colored rectangle in the background, indicating uncertainty of the prediction.

    Args:
        probs: A tensor of shape [N, N, vocab_size] containing normalized probabilities
                for each grid location.
        entropy (optional): A tensor of shape [N, N] containing entropy values for each grid location,
                 representing the uncertainty of the probability distribution at that location.
        im_size: An integer representing the size of the grid (assumed square). Determines
                 the number of rows and columns in the grid.
        vocab_size: An integer representing the vocabulary size (i.e., the number of elements
                    in each probability distribution).
        skip: An optional integer, default = 2. Determines the step size for skipping grid points
              when populating the subplots. A larger value reduces the number of subplots.
        max_y: An optional float, each subplot will be plotted in range (0, max_y)
        save_path: An optional string for saving the plot
    """
    print('Visualizes probs and entropy on a grid of subplots...')
    # Create subplots (limit the number of displayed distributions)
    fig = make_subplots(
        rows=im_size // skip,
        cols=im_size // skip,
        horizontal_spacing=0.005,  # Smaller horizontal space between subplots
        vertical_spacing=0.005  # Smaller vertical space between subplots
    )

    # Populate the subplots with sampled data

    def intensity_to_color(intensity, cmap_name='viridis'):
        cmap = cm.get_cmap(cmap_name)  # Get the colormap
        norm = mcolors.Normalize(vmin=0, vmax=1)  # Normalize the intensity values between 0 and 1
        rgba = cmap(norm(intensity))  # Map intensity to RGBA
        r, g, b, a = rgba  # Unpack RGBA values
        return f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 1.0)'  # Convert to rgba format

    for i in range(im_size // skip):
        for j in range(im_size // skip):
            sampled_data = probs[i * skip, j * skip].cpu().float()  # Sample every 65th value

            # Add heatmap as the background
            if entropy is not None:
                fig.add_shape(
                    type="rect",
                    x0=0, x1=vocab_size, y0=0, y1=max_y,  # Cover the entire subplot area
                    fillcolor=intensity_to_color(
                        entropy[i * skip, j * skip].cpu().float() / torch.tensor(vocab_size).log()),
                    # Background color from intensity
                    line=dict(color="rgba(0,0,0,0)"),  # No border line
                    layer="below",  # Ensure the rectangle is drawn below the plot
                    xref="x", yref="y",  # Reference the plot's axes
                    row=i + 1, col=j + 1
                )

            # Add bar plot on top of the heatmap
            fig.add_trace(
                go.Scatter(
                    y=sampled_data,
                    mode='lines',
                    # line=dict(color='white'),
                    showlegend=False),
                row=i + 1, col=j + 1
            )

    # Customize the layout
    fig.update_layout(
        height=2400, width=2400,
        title_text=f"Sampled {im_size // skip}x{im_size // skip} Probability Distributions (vocab size: {vocab_size}), y-limit (0, {max_y})",
        title_font_size=48,  # Bigger title font size
        title_x=0.5,  # Center the title
        title_xanchor='center',  # Ensure proper alignment in the center
    )

    fig.update_yaxes(range=[0, max_y])  # Example: setting y-limits to [0, 0.02]
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Remove grid lines and background color for all subplots
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    if save_path is not None:
        fig.write_image(save_path, format="png")

    # Display the plot
    fig.show()


