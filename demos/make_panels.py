"""
Turn Gradio capture triples into per-run panels and flicker GIFs.

Each Save in demos/gradio_hypothetical.py writes a set of files sharing one stem
(the timestamp, plus your optional name):

    <stem>_input.png            clean frame the model saw
    <stem>_input_annotated.png  same frame with your arrows / anchor boxes
    <stem>_pred.png             the prediction
    <stem>_meta.json            model, seed, move points

Groups them by stem (one stem is one run) and writes:

    demos/outputs/panels/<stem>.png   annotated input | prediction, side by side
    demos/outputs/gifs/<stem>.gif     clean input <-> prediction, flickering

GIFs use the clean input, not the annotated one — flashing arrows swamp a
small displacement. The arrows stay on the panel, where they are static.

Usage:
    python demos/make_panels.py                 # everything not already built
    python demos/make_panels.py --force         # rebuild all
    python demos/make_panels.py --no-gifs
"""

import argparse
import glob
import json
import os

from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SAVE_DIR = os.path.join(_REPO_ROOT, "demos", "outputs")

PAD = 16          # gutter around and between the two images
HEADER_H = 52     # caption strip above the images
LABEL_H = 26      # per-image label strip
BG = (255, 255, 255)
FG = (38, 38, 38)        # seaborn's ".15" text grey
MUTED = (115, 115, 115)

# Source Sans 3, matching personal_scripts/plotting/plot_utils.py so these
# figures sit alongside the matplotlib ones without a font mismatch.
_SOURCE_SANS_DIR = "/ccn2/u/khaiaw/.local/share/fonts/source-sans"
_DEJAVU_DIR = "/usr/share/fonts/truetype/dejavu"
_HEAVY_WEIGHTS = {"Bold", "SemiBold", "Black", "ExtraBold", "Medium"}


def _font(size, weight="Regular"):
    """Source Sans 3 at `weight`, falling back to DejaVu then PIL's bitmap font."""
    bold_ish = weight in _HEAVY_WEIGHTS
    for path in [
        os.path.join(_SOURCE_SANS_DIR, f"SourceSans3-{weight}.ttf"),
        os.path.join(_SOURCE_SANS_DIR, "SourceSans3-Regular.ttf"),
        os.path.join(_DEJAVU_DIR, "DejaVuSans-Bold.ttf" if bold_ish else "DejaVuSans.ttf"),
        os.path.join(_DEJAVU_DIR, "DejaVuSans.ttf"),
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def find_runs(save_dir):
    """Group capture files by stem. Returns [(stem, {kind: path}), ...] sorted by stem."""
    runs = {}
    for meta_path in glob.glob(os.path.join(save_dir, "*_meta.json")):
        stem = os.path.basename(meta_path)[: -len("_meta.json")]
        parts = {"meta": meta_path}
        for kind, suffix in (("clean", "_input.png"),
                             ("annotated", "_input_annotated.png"),
                             ("pred", "_pred.png")):
            path = os.path.join(save_dir, stem + suffix)
            if os.path.exists(path):
                parts[kind] = path
        runs[stem] = parts
    return sorted(runs.items())


def describe(meta):
    """One-line subtitle: model, seed, and how many real drags vs fixed anchors."""
    model = os.path.basename(os.path.dirname(meta.get("model_name", ""))) or "?"
    seed = meta.get("seed", "?")
    moves = meta.get("move_points_model_res") or []
    drags = sum(1 for m in moves if (m[0], m[1]) != (m[2], m[3]))
    holds = len(moves) - drags
    return f"{model}  ·  seed {seed}  ·  {drags} drag{'s' * (drags != 1)}, {holds} fixed"


def make_panel(stem, parts, out_path):
    left_path = parts.get("annotated") or parts.get("clean")
    right_path = parts.get("pred")
    if left_path is None or right_path is None:
        return False

    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    h = max(left.height, right.height)
    if left.height != h:
        left = left.resize((round(left.width * h / left.height), h), Image.LANCZOS)
    if right.height != h:
        right = right.resize((round(right.width * h / right.height), h), Image.LANCZOS)

    W = PAD * 3 + left.width + right.width
    H = HEADER_H + LABEL_H + h + PAD
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    draw.text((PAD, 10), stem, font=_font(19, "SemiBold"), fill=FG)
    try:
        meta = json.load(open(parts["meta"]))
        draw.text((PAD, 32), describe(meta), font=_font(14), fill=MUTED)
    except (OSError, ValueError, KeyError):
        pass

    for img, x, label in ((left, PAD, "input (annotated)"),
                          (right, PAD * 2 + left.width, "prediction")):
        draw.text((x, HEADER_H + 4), label, font=_font(14), fill=MUTED)
        canvas.paste(img, (x, HEADER_H + LABEL_H))

    canvas.save(out_path, dpi=(300, 300))
    return True


def make_gif(parts, out_path, duration_ms):
    clean_path = parts.get("clean") or parts.get("annotated")
    pred_path = parts.get("pred")
    if clean_path is None or pred_path is None:
        return False
    a = Image.open(clean_path).convert("RGB")
    b = Image.open(pred_path).convert("RGB")
    if b.size != a.size:
        b = b.resize(a.size, Image.LANCZOS)
    a.save(out_path, save_all=True, append_images=[b],
           duration=duration_ms, loop=0, optimize=False)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", default=DEFAULT_SAVE_DIR,
                    help="Where the Gradio captures live (default: demos/outputs)")
    ap.add_argument("--panels-dir", default=None, help="Default: <save-dir>/panels")
    ap.add_argument("--gifs-dir", default=None, help="Default: <save-dir>/gifs")
    ap.add_argument("--duration", type=int, default=600,
                    help="Milliseconds per GIF frame (default 600)")
    ap.add_argument("--no-gifs", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild outputs that already exist")
    args = ap.parse_args()

    panels_dir = args.panels_dir or os.path.join(args.save_dir, "panels")
    gifs_dir = args.gifs_dir or os.path.join(args.save_dir, "gifs")
    os.makedirs(panels_dir, exist_ok=True)
    if not args.no_gifs:
        os.makedirs(gifs_dir, exist_ok=True)

    runs = find_runs(args.save_dir)
    if not runs:
        print(f"No captures found in {args.save_dir}")
        return

    n_panels = n_gifs = n_skipped = 0
    for stem, parts in runs:
        if "pred" not in parts:
            print(f"  skip {stem} — no prediction saved")
            n_skipped += 1
            continue

        panel_path = os.path.join(panels_dir, f"{stem}.png")
        if args.force or not os.path.exists(panel_path):
            if make_panel(stem, parts, panel_path):
                n_panels += 1

        if not args.no_gifs:
            gif_path = os.path.join(gifs_dir, f"{stem}.gif")
            if args.force or not os.path.exists(gif_path):
                if make_gif(parts, gif_path, args.duration):
                    n_gifs += 1

    print(f"{len(runs)} run(s) found in {args.save_dir}")
    print(f"  {n_panels} panel(s) -> {panels_dir}")
    if not args.no_gifs:
        print(f"  {n_gifs} gif(s)   -> {gifs_dir}")
    if n_skipped:
        print(f"  {n_skipped} skipped (saved before Run ZWM)")


if __name__ == "__main__":
    main()
