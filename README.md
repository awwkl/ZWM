# ZWM
Code for the paper "Zero-shot World Models Are Developmentally Efficient Learners"

<p align="center">
  <a href="https://arxiv.org/abs/2604.10333">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2604.10333-blue">
  </a>
  <a href="https://huggingface.co/awwkl/models">
    <img alt="Models" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-yellow">
  </a>
</p>

<p align="center">
  <img alt="ZWM" src="assets/zwm_thumbnail.png" width="85%">
</p>

Today's best AI needs orders of magnitude more data than a human child to achieve visual competence. We introduce the Zero-shot World Model (ZWM), an approach that substantially narrows this gap. Even when trained on a single child's visual experience, BabyZWM matches state-of-the-art models on diverse visual-cognitive tasks – with no task-specific training, i.e., zero-shot. Our work presents a blueprint for efficient and flexible learning from human-scale data, advancing a path toward data-efficient AI systems.

## System requirements

- **OS**: Linux (tested on Ubuntu 20.04, kernel 5.4). Should also work on other Linux distributions; not tested on macOS or Windows.
- **Python**: 3.10
- **GPU**: NVIDIA GPU required for training and recommended for inference. Reference hardware: NVIDIA A40 (48 GB) with CUDA driver ≥ 12.4. Other modern NVIDIA GPUs with bfloat16 support (A100, H100, RTX 30/40 series, etc.) should also work.
- **VRAM**: ~14 GB peak for the 170M training smoke test (`per_device_batch_size=8`, bfloat16); a single 16 GB GPU is sufficient. Replicating the full 170M / 1B BabyView training runs uses an 8-GPU node at the recipe's batch size.
- **Dependencies**: full pinned list in [requirements.txt](requirements.txt); high-level summary in [SOFTWARE.md](SOFTWARE.md).

## Installation

```bash
git clone https://github.com/awwkl/ZWM.git
cd ZWM

# Create a fresh conda env (recommended)
conda create -n zwm python=3.10 -y
conda activate zwm

# Install ZWM and its dependencies
pip install -e .
```

If your CUDA driver is older than 12.4, install a CUDA-matched PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -e .`.

Verify the install:
```bash
python -c "from zwm.zwm_predictor import ZWMPredictor; print('ok')"
```

Typical install time on a workstation with a fast network: ~5–10 minutes (dominated by the PyTorch / CUDA wheel download).

## Model zoo

| HF repo | Training data | Params | Resolution |
|---|---|---|---|
| [`awwkl/zwm-bvd-170m`](https://huggingface.co/awwkl/zwm-bvd-170m) | BigVideoDataset | 170M | 256 |
| [`awwkl/zwm-bvd-1b`](https://huggingface.co/awwkl/zwm-bvd-1b) | BigVideoDataset | 1B | 256 |
| [`awwkl/zwm-kinetics-170m`](https://huggingface.co/awwkl/zwm-kinetics-170m) | Kinetics-400 | 170M | 256 |
| [`awwkl/zwm-kinetics-1b`](https://huggingface.co/awwkl/zwm-kinetics-1b) | Kinetics-400 | 1B | 256 |
| [`awwkl/zwm-babyview-170m`](https://huggingface.co/awwkl/zwm-babyview-170m) | BabyView | 170M | 256 |
| [`awwkl/zwm-babyview-1b`](https://huggingface.co/awwkl/zwm-babyview-1b) | BabyView | 1B | 256 |

Download a checkpoint into `./out/`:
```bash
python scripts/hf_model_download.py awwkl/zwm-bvd-170m
```

## Quickstart — Hypothetical prediction

Predict a future frame by moving patches:

```python
import numpy as np
from PIL import Image
from zwm.zwm_predictor import ZWMPredictor

predictor = ZWMPredictor("awwkl/zwm-bvd-170m/model.pt")
frame0 = Image.open("demos/assets/examples/bag.jpg").convert("RGB")

# Each row is (x1, y1, x2, y2) in pixel coords at the model resolution (256).
# With patch_size_move_mult=2, a 2x2 block of 8x8 patches (16x16 px) starting
# at (120, 120) is moved so its top-left lands at (140, 100).
move_points = np.array([[120, 120, 140, 100]])

out = predictor.hypothetical_prediction(frame0, move_points, patch_size_move_mult=2)
out["frame1_pred_pil"].save("hypothetical.png")
```

<p align="center">
  <img alt="Hypothetical prediction example" src="assets/zwm_hypothetical_example.png" width="85%">
</p>

## Quickstart — Factual prediction

Reconstruct masked patches in `frame1` given `frame0` and a few visible hints:

```python
from PIL import Image
from zwm.zwm_predictor import ZWMPredictor

predictor = ZWMPredictor("awwkl/zwm-bvd-170m/model.pt")
# Replace these with two consecutive frames from your own video.
frame0 = Image.open("frame0.jpg").convert("RGB")
frame1 = Image.open("frame1.jpg").convert("RGB")

out = predictor.factual_prediction(frame0, frame1, frame_gap=10, mask_ratio=0.9)
out["frame1_pred_pil"].save("factual_prediction.png")
```

For batched evaluation over a video dataset (samples frame pairs, runs prediction, saves visualizations), see [zwm/inv/inv_zwm_factual_prediction.py](zwm/inv/inv_zwm_factual_prediction.py).

The repo bundles three short CC-BY 3.0 clips in [data/demo_videos/](data/demo_videos/) (excerpts of Blender's *Tears of Steel*, *Caminandes 1*, and *Sintel*) so you can run this end-to-end immediately:

```bash
bash demos/run_inference_demo.sh
```

This is equivalent to:

```bash
python -m zwm.inv.inv_zwm_factual_prediction \
    --videos_dir data/demo_videos/ \
    --n_samples_to_eval 10 \
    --num_viz 10 \
    --model_name awwkl/zwm-bvd-170m/model.pt \
    --frame1_mask_ratio 0.0
```

The wrapper script also auto-downloads the HuggingFace checkpoint into `out/` on first run. With only 3 bundled clips, the script cycles over them with replacement; each sample uses a randomly chosen frame pair (gap in [5, 16) frames) so 10 samples surface a range of gap sizes. To run on your own data, point `--videos_dir` at any directory of `.mp4` files (recursively globbed).

**Expected output:** Saves a 7-panel visualization for each sampled frame pair to `viz/zwm_factual_predictions/<model_name>/iter_*.png` (panels: frame 0, frame 1 ground truth, prediction, prediction unmasked, frame 1 with mask, frame 0 raw, frame 1 raw), plus `loss_value.txt` with the mean MSE between predicted and ground-truth frame 1.

**Expected run time:** ~2 minutes for 10 samples on a single NVIDIA A40, plus a one-time HuggingFace checkpoint download on first run (~30 s for the 170M model).

## Interactive Gradio demo

Launch a browser UI for hypothetical prediction — click an image to mark a patch, drag to specify where it should move, and watch the model fill in the rest:

```bash
python -m demos.gradio_hypothetical --model_name awwkl/zwm-bvd-170m/model.pt
```

## Training

Training entry point is [zwm/train.py](zwm/train.py).

To replicate the released BabyView-170M model ([`awwkl/zwm-babyview-170m`](https://huggingface.co/awwkl/zwm-babyview-170m)) end-to-end, edit `--train_data_dir` in [scripts/train_zwm_babyview_170m.sh](scripts/train_zwm_babyview_170m.sh) to point at your local copy of the BabyView 10s 256p clips, then run:

```bash
bash scripts/train_zwm_babyview_170m.sh
```

This launches an 8-GPU torchrun with the exact recipe used to produce the released checkpoint (170M params, batch size 512, 200k iters, bfloat16, A40 reference hardware). The 1B variant uses the same recipe with a smaller `--per_device_batch_size`; see [scripts/train_zwm_babyview_1b.sh](scripts/train_zwm_babyview_1b.sh).

Large-scale training is not the focus of this release — we recommend starting from a released checkpoint for most downstream use.

### Training smoke test

To verify the training loop runs end-to-end on your machine without setting up a real dataset:

```bash
bash demos/run_training_smoketest.sh
```

This runs 301 iterations of the 170M config on the bundled clips in [data/demo_videos/](data/demo_videos/) at `--per_device_batch_size 8`, logs loss every 10 iterations, and saves checkpoints at iters 100, 200, 300 to `out/zwm_smoketest/`. Expected run time: ~10–15 minutes on a single NVIDIA A40. This is a smoke test, not a usable training run.

## Progress Update

- [x] Model code + training script ([zwm/model.py](zwm/model.py), [zwm/train.py](zwm/train.py))
- [x] Pretrained checkpoints on HuggingFace — 6 variants: {170M, 1B} × {BigVideoDataset, Kinetics, BabyView}
- [x] Inference API — `ZWMPredictor` supports factual and hypothetical prediction ([zwm/zwm_predictor.py](zwm/zwm_predictor.py))
- [x] Interactive Gradio demo for hypothetical generation ([demos/gradio_hypothetical.py](demos/gradio_hypothetical.py))
- [ ] Inference + eval scripts for visual-cognitive tasks (optical flow, depth, object segments, intuitive physics)
- [ ] Training datasets (BabyView release)

## Citation

If you use this code/model in your research, please cite it as follows:

```bibtex
@misc{aw2026zeroshotworldmodelsdevelopmentally,
      title={Zero-shot World Models Are Developmentally Efficient Learners}, 
      author={Khai Loong Aw and Klemen Kotar and Wanhee Lee and Seungwoo Kim and Khaled Jedoui and Rahul Venkatesh and Lilian Naing Chen and Michael C. Frank and Daniel L. K. Yamins},
      year={2026},
      eprint={2604.10333},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.10333}, 
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).