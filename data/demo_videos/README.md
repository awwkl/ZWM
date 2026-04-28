# Demo videos

A small set of short video clips bundled with the repo so that the inference and training demo scripts (`demos/run_inference_demo.sh`, `demos/run_training_smoketest.sh`) can be run out-of-the-box without downloading any external data.

## Contents

| File | Source | License | Duration | Resolution |
|---|---|---|---|---|
| `clip_01_tears_of_steel.mp4` | Blender Foundation, *Tears of Steel* (2012), 10s excerpt (≈05:00–05:10) from the [720p Internet Archive release](https://archive.org/details/Tears-of-Steel), center-cropped from 862×360 to 640×360 | [CC-BY 3.0](https://creativecommons.org/licenses/by/3.0/) | 10 s, 24 fps, 240 frames | 640×360 |
| `clip_02_caminandes.mp4` | Blender Foundation, *Caminandes 1: Llama Drama* (2013), 10s excerpt (≈00:15–00:25) from the [1080p Internet Archive release](https://archive.org/details/Caminandes1LlamaDrama), scaled from 1920×1080 to 640×360 | [CC-BY 3.0](https://creativecommons.org/licenses/by/3.0/) | 10 s, 24 fps, 240 frames | 640×360 |
| `clip_03_sintel.mp4` | Blender Foundation, *Sintel* (2010), 10s excerpt re-encoded by [test-videos.co.uk](https://test-videos.co.uk/sintel/mp4-h264) | [CC-BY 3.0](https://creativecommons.org/licenses/by/3.0/) | 10 s, 24 fps, 240 frames | 640×360 |

All three clips are H.264 / 10 s (~2 MB total bundled). Resolution is left at 640×360 rather than pre-cropped to 256×256, so the training-side random square crop has slack to vary.

## Attribution

Per CC-BY 3.0, attribution is preserved here:
- *Tears of Steel* © 2012 Blender Foundation — https://mango.blender.org/
- *Caminandes 1: Llama Drama* © 2013 Blender Foundation — https://caminandes.com/
- *Sintel* © 2010 Blender Foundation — https://durian.blender.org/

## How these clips are used

- `demos/run_inference_demo.sh` runs `zwm.inv.inv_zwm_factual_prediction` over this directory and saves frame-pair reconstruction visualizations to `viz/zwm_factual_predictions/`.
- `demos/run_training_smoketest.sh` runs `zwm.train` for a small number of steps over this directory to demonstrate the training loop.

Both scripts call `glob.glob('data/demo_videos/**/*.mp4', recursive=True)`; any additional `.mp4` dropped into this directory will be picked up automatically.

## Adding more clips

If you want to add or replace clips, please ensure each new clip:

1. Has a permissive license (CC0 / CC-BY / public domain) and update the table above with the source + license.
2. Is short (~3-15 s) and small (~1-5 MB).
3. Has at least 16 frames (required by the factual-prediction frame-pair sampler in `zwm/inv/inv_zwm_factual_prediction.py`).

Suggested sources: [Pexels](https://www.pexels.com/), [Pixabay](https://pixabay.com/videos/), [Wikimedia Commons](https://commons.wikimedia.org/), [Internet Archive](https://archive.org/), [test-videos.co.uk](https://test-videos.co.uk/).