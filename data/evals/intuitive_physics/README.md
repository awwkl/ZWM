---
license: cc-by-nc-sa-3.0
task_categories:
  - image-to-image
tags:
  - intuitive-physics
  - counterfactual-world-model
  - zwm
---

# ZWM intuitive-physics eval dataset

A short-timescale physical-reasoning benchmark used to evaluate world models
in the [ZWM repo](https://github.com/awwkl/ZWM). It contains tabletop
hand-object interactions across five categories of physical reasoning:

1. **Object cohesion** — moving part of an object moves the whole object.
2. **Support - top** — pushing the top object only moves the top object.
3. **Support - bottom** — pushing the bottom object moves both objects.
4. **Force transfer** — pushing object A into B makes B move.
5. **Force separation** — moving A does not affect a spatially separated B.

The benchmark contains 100 image pairs (5 categories × 20 pairs). For each
pair the model predicts a target frame (`frame_03`) conditioned on a context
frame (`frame_02`) plus a small set of revealed patches around a grounding
point. Each pair is evaluated under 8 random mask configurations, yielding
5 × 20 × 8 = 800 evaluations per model.

**Accuracy** is the proportion of examples for which the model's prediction
is closer to the ground-truth target (`frame_03`) than to the context frame
(`frame_02`), measured in MSE and LPIPS.

## Layout

```
annotations.csv                                       # 100 rows: category, video_id, factual_x/y, counterfactual_x/y, keyframes
keyframes/<category>/<video_id>/frame_02.png          # 512x512 RGB context frame
keyframes/<category>/<video_id>/frame_03.png          # 512x512 RGB target frame
segment_masks/<category>/<video_id>/frame<N>_<primary|secondary|overall>_mask.npy
                                                      # (1, 512, 512) binary masks (secondary may be absent)
```

## Usage with the ZWM repo

```
huggingface-cli download awwkl/zwm-intuitive-physics --repo-type dataset --local-dir data/evals/intuitive_physics/
bash scripts/eval/intuitive_physics/eval_intuitive_physics.sh
bash scripts/eval/intuitive_physics/grade_intuitive_physics.sh
```
