from setuptools import setup, find_packages


setup(
    name="zwm",
    version="0.1",
    packages=find_packages(),
    description="Zero-shot World Model",
    author="Khai Loong Aw",
    install_requires=[
        'numpy',
        'torch>=2.4,<2.9',
        'scipy',
        'tqdm',
        'wandb',
        'einops',
        'matplotlib',
        'h5py',
        'torchvision',
        'future',
        'opencv-python',
        'decord',
        'pandas',
        'matplotlib',
        'moviepy',
        'scikit-image',
        'scikit-learn',
        'vector_quantize_pytorch',
        'google-cloud-storage',
        'huggingface_hub',
        'gradio',
    ],
)
