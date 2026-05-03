"""Extract per-frame PNGs from the official TapVID-DAVIS pickle.

The TapVID-DAVIS distribution (CC-BY) ships as a single pickle with the
videos embedded as uint8 arrays of shape (T, H, W, 3). The flow eval
(`zwm.eval.flow.eval_tapvid_flow`) reads frames as PNGs from
`<out_dir>/<video_name>/video/<idx>.png`. This script materializes that
on-disk layout from the pickle.

Step 1 — download the pickle (one-time, ~2.4 GB):

  Official source (TAP-Vid project page):
      https://storage.googleapis.com/dm-tapnet/index.html
  Direct download:
      https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip
  After unzipping, move `tapvid_davis.pkl` to:
      data/evals/flow/tapvid_davis_first/tapvid_davis.pkl

  (License: CC-BY 4.0, see the TAP-Vid README inside the zip.)

Step 2 — run this script:

    python scripts/eval/flow/extract_tapvid_davis_pkl.py \\
        --pkl_path data/evals/flow/tapvid_davis_first/tapvid_davis.pkl \\
        --out_dir data/evals/flow/tapvid_davis_first/frames/

Expected output: 30 video subdirectories, each with ~50-90 PNGs at the
native TapVID-DAVIS resolution (480x854). ~1.2 GB total.
"""
import argparse
import os
import pickle

import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True,
                        help="Path to tapvid_davis.pkl from the official TapVID release.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory; PNGs land at <out_dir>/<video>/video/<idx>.png.")
    args = parser.parse_args()

    with open(args.pkl_path, "rb") as f:
        data = pickle.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    total = 0
    for video_name, entry in tqdm(data.items(), desc="videos"):
        frames = entry["video"]  # (T, H, W, 3) uint8
        assert frames.dtype == np.uint8 and frames.ndim == 4, \
            f"unexpected frames shape/dtype for {video_name}: {frames.shape} {frames.dtype}"
        video_dir = os.path.join(args.out_dir, video_name, "video")
        os.makedirs(video_dir, exist_ok=True)
        for idx in range(frames.shape[0]):
            Image.fromarray(frames[idx]).save(os.path.join(video_dir, f"{idx:03d}.png"))
        total += frames.shape[0]

    print(f"Wrote {total} PNG frames across {len(data)} videos to {args.out_dir}")


if __name__ == "__main__":
    main()
