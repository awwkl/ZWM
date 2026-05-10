---
license: cc-by-nc-sa-3.0
task_categories:
  - depth-estimation
tags:
  - stereo
  - kitti
  - zwm
  - counterfactual-flow
---

# ZWM stereo-depth eval dataset

Stereo image pairs with two annotated points per pair (with ground-truth
depth) used to evaluate depth perception in the [ZWM repo](https://github.com/awwkl/ZWM).
The eval reuses ZWM's counterfactual flow engine: predict EPE for each
point's "motion" between stereo halves, then order depths by EPE magnitude
(closer = larger EPE).

## Layout

```
stereo_depth/
├── dataset.json                # 328 entries (164 pairs across kitti_1000 + kitti_500_flipud)
└── frames/
    ├── cropped_without_crosses/    # query frames (model input)
    ├── cropped_stereo_mate/        # target frames (model input)
    └── cropped_with_crosses/       # annotated viz overlays (not consumed by the eval)
```

`query_frame_file` / `target_frame_file` / `cropped_with_crosses_file` in the
JSON are paths relative to `stereo_depth/frames/`.

## Usage with the ZWM repo

```
huggingface-cli download awwkl/zwm-stereo-depth-eval --repo-type dataset --local-dir data/evals/depth/
bash scripts/eval/depth/eval_stereo_depth.sh
bash scripts/eval/depth/grade_stereo_depth.sh
```

## Source

Stereo crops + point annotations from the UniQA-3D stereo benchmark
(KITTI-1000 and KITTI-500-flipud subsets, ~164 image pairs total).
