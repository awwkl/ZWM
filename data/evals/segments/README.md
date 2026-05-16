---
license: cc-by-nc-sa-3.0
task_categories:
  - image-segmentation
tags:
  - segmentation
  - spelke-bench
  - spelke-net
  - zwm
---

# ZWM SpelkeBench eval dataset

A mirror of the SpelkeBench object-segmentation benchmark used to evaluate
object-level scene understanding in the [ZWM repo](https://github.com/awwkl/ZWM).
The eval uses ZWM's counterfactual-prediction engine: for each annotated
object centroid, perturb the centroid patch, predict the next frame with and
without the perturbation, compute optical flow between the perturbed and
unperturbed predictions, and threshold the dot-product-with-perturbation
heatmap into a binary segment.

## Layout

```
spelke_bench.h5     # one group per image; each group has:
                    #   rgb       (H, W, 3) uint8
                    #   segment   (N, H, W) uint8 binary masks
                    #   centroid  (N, 2)   optional precomputed centroids
```

## Usage with the ZWM repo

```
huggingface-cli download awwkl/zwm-spelke-bench --repo-type dataset --local-dir data/evals/segments/
bash scripts/eval/segments/eval_spelke_seg.sh
bash scripts/eval/segments/grade_spelke_seg.sh
```

## Source

SpelkeBench from the SpelkeNet release:
https://storage.googleapis.com/stanford_neuroai_models/SpelkeNet/SpelkeBench/final_dataset.h5
