---
license: cc-by-4.0
task_categories:
  - other
tags:
  - point-tracking
  - optical-flow
  - tap-vid
  - davis
  - zwm
  - counterfactual-flow
---

# ZWM TAP-Vid-DAVIS eval dataset

A mirror of the TAP-Vid-DAVIS point-tracking benchmark, laid out for the optical-flow eval in
the [ZWM repo](https://github.com/awwkl/ZWM). The eval uses ZWM's counterfactual-prediction
engine: perturb the patch around a query point in frame 0, predict frame 1 with and without the
perturbation, and locate the perturbation in the difference between the two predictions. End-point
error against the TAP-Vid ground-truth tracks is the metric.

## Layout

```
tapvid_davis_first/
├── dataset.json          # 38881 query/target point pairs over 30 DAVIS videos (full benchmark)
├── dataset_demo.json     # 30 pairs, one per video (smoketest)
├── tapvid_davis.pkl      # official TAP-Vid-DAVIS pickle, unchanged (ground-truth tracks; used by the grader)
├── TAPVID_README.md      # README.md from the official zip (terms of use)
├── SOURCES.md            # SOURCES.md from the official zip (video credits)
└── frames/<video>/video/<idx>.png   # 1999 frames extracted from the pickle (model input)
```

`query_frame_file` / `target_frame_file` in the JSONs are paths relative to
`tapvid_davis_first/frames/`.

## Usage with the ZWM repo

```
huggingface-cli download awwkl/zwm-tapvid-davis-eval --repo-type dataset --local-dir data/evals/flow/
bash scripts/eval/flow/eval_tapvid_flow_demo.sh   # smoketest
bash scripts/eval/flow/eval_tapvid_flow.sh        # full benchmark
bash scripts/eval/flow/grade_tapvid_flow.sh
```

## Source and license

`tapvid_davis.pkl` is the official file from DeepMind's
[TAP-Vid release](https://storage.googleapis.com/dm-tapnet/index.html)
(`tapvid_davis.zip`, Dec 2022), redistributed unchanged; the PNG frames are extracted from it. TAP-Vid
annotations are released under CC-BY 4.0, and the videos (from DAVIS 2017) are owned by their
creators: see `TAPVID_README.md` and `SOURCES.md`, copied from the zip. If you use this data, please cite:

```
@inproceedings{doersch2022tapvid,
  title     = {TAP-Vid: A Benchmark for Tracking Any Point in a Video},
  author    = {Doersch, Carl and Gupta, Ankush and Markeeva, Larisa and Recasens, Adri{\`a} and Smaira, Lucas and Aytar, Yusuf and Carreira, Jo{\~a}o and Zisserman, Andrew and Yang, Yi},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2022}
}

@article{pont2017davis,
  title   = {The 2017 DAVIS Challenge on Video Object Segmentation},
  author  = {Pont-Tuset, Jordi and Perazzi, Federico and Caelles, Sergi and Arbel{\'a}ez, Pablo and Sorkine-Hornung, Alex and Van Gool, Luc},
  journal = {arXiv:1704.00675},
  year    = {2017}
}
```
