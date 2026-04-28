# Software and code

Policy information about availability of computer code.

This document lists the software used in this study.

## Data collection

> *Provide a description of all commercial, open source and custom code used to collect the data in this study, specifying the version used OR state that no software was used.*

No software was used for data collection. This study uses pre-existing video datasets (e.g., BabyView); no new data was collected.

## Data analysis

> *Provide a description of all commercial, open source and custom code used to analyse the data in this study, specifying the version used OR state that no software was used.*

All analyses were performed with custom code written in Python 3.10, available under the MIT license at https://github.com/awwkl/ZWM, with pretrained model weights at https://huggingface.co/awwkl/models.

Model training and inference used PyTorch 2.8.0, torchvision 0.23.0, and Triton 3.4.0, with CUDA 13.0 / cuDNN 9.19. Numerical and scientific computing: NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, einops 0.8.2, einx 0.4.3. Image and video I/O: scikit-image 0.25.2, Pillow 11.3.0, opencv-python 4.13.0.92, decord 0.6.0, moviepy 2.2.1, imageio 2.37.3, h5py 3.16.0, tifffile 2025.5.10. Machine learning utilities: scikit-learn 1.7.2, vector-quantize-pytorch 1.28.2. Visualization: matplotlib 3.10.8. Experiment tracking with Weights & Biases (wandb) 0.26.0. Pretrained model distribution via huggingface_hub 1.11.0 and google-cloud-storage 3.10.1. Interactive demos built with Gradio 6.12.0.

A complete pinned dependency list is provided in requirements.txt for full reproducibility.
