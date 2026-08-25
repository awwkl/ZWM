"""
Build a two-panel qualitative figure from the Gradio captures.

    (A) Collision into a lighter object     6 examples in one row
    (B) Collision into a heavier object    6 examples in one row

Each example is one column: the annotated input, with the model's
prediction directly beneath it.

Reads demos/outputs/labels.csv:

    stem,target,scene
    20260825_105611,heavy,blue ball -> dumbbell

`target` is the mass of the object pushed INTO, not the object grabbed: in
105611 the ball is dragged into the dumbbell, so the target is heavy.
Examples are laid out in labels.csv order.

Writes:
    demos/outputs/light/, demos/outputs/heavy/       capture files copied by class
    demos/outputs/figures/figure_qualitative.png     the figure

Usage:
    python demos/make_figure.py
    python demos/make_figure.py --cell-px 300
    python demos/make_figure.py --rows 2 --cols 3   # back to a 2x3 block
"""

import argparse
import csv
import os
import shutil

from PIL import Image, ImageDraw

try:
    from make_panels import _font, find_runs, DEFAULT_SAVE_DIR
except ImportError:  # invoked as `python -m demos.make_figure`
    from demos.make_panels import _font, find_runs, DEFAULT_SAVE_DIR

# Panel order is the figure's panel lettering: A first, then B.
PANELS = [("light", "Collision into a lighter object"),
          ("heavy", "Collision into a heavier object")]

BG = (255, 255, 255)
FG = (38, 38, 38)        # seaborn's ".15" text grey
MUTED = (115, 115, 115)
RULE = (210, 210, 210)

VGAP = 7           # between the input and the prediction beneath it
EX_GAP = 22        # between example columns
ROW_GAP = 30       # between the two bands of examples
PANEL_GAP = 44
MARGIN = 26
HEAD_H = 42        # panel heading strip
SUB_H = 25         # per-example caption
LABEL_W = 52       # left gutter holding the rotated "input" / "prediction" labels
BORDER = (200, 200, 200)


def read_labels(path):
    """-> [(stem, target, scene)] in file order."""
    rows = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            stem = row[0].strip()
            if not stem:
                continue
            target = row[1].strip().lower() if len(row) > 1 else ""
            scene = row[2].strip() if len(row) > 2 else ""
            rows.append((stem, target, scene))
    return rows


def copy_by_class(runs, labelled, save_dir):
    counts = {}
    for stem, target, _ in labelled:
        if target not in {c for c, _ in PANELS} or stem not in runs:
            continue
        os.makedirs(os.path.join(save_dir, target), exist_ok=True)
        for path in runs[stem].values():
            shutil.copy2(path, os.path.join(save_dir, target, os.path.basename(path)))
        counts[target] = counts.get(target, 0) + 1
    return counts


def load_pair(parts, cell_px):
    left = parts.get("annotated") or parts.get("clean")
    right = parts.get("pred")
    if left is None or right is None:
        return None
    a = Image.open(left).convert("RGB").resize((cell_px, cell_px), Image.LANCZOS)
    b = Image.open(right).convert("RGB").resize((cell_px, cell_px), Image.LANCZOS)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    ap.add_argument("--labels", default=None, help="Default: <save-dir>/labels.csv")
    ap.add_argument("--out", default=None,
                    help="Default: <save-dir>/figures/figure_qualitative.png")
    ap.add_argument("--cell-px", type=int, default=230, help="Size of each image (default 230)")
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--no-copy", action="store_true")
    args = ap.parse_args()

    labels_path = args.labels or os.path.join(args.save_dir, "labels.csv")
    out_path = args.out or os.path.join(args.save_dir, "figures",
                                        "figure_qualitative.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    labelled = read_labels(labels_path)
    runs = dict(find_runs(args.save_dir))

    for stem, _, _ in labelled:
        if stem not in runs:
            print(f"  labelled but no capture found: {stem}")
    labelled_stems = {s for s, _, _ in labelled}
    for stem in sorted(runs):
        if stem not in labelled_stems:
            print(f"  capture has no label, skipped:  {stem}")

    if not args.no_copy:
        counts = copy_by_class(runs, labelled, args.save_dir)
        print("copied: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))

    per_panel = args.rows * args.cols
    picks = {}
    for cls, _ in PANELS:
        avail = [s for s, t, _ in labelled if t == cls and s in runs]
        if len(avail) < per_panel:
            print(f"  only {len(avail)} {cls} example(s) for {per_panel} slots — "
                  f"panel will have gaps")
        elif len(avail) > per_panel:
            print(f"  {len(avail)} {cls} examples, using the first {per_panel} "
                  f"(reorder labels.csv to change which)")
        picks[cls] = avail[:per_panel]

    scene_of = {s: sc for s, _, sc in labelled}

    cp = args.cell_px
    panel_w = LABEL_W + args.cols * cp + (args.cols - 1) * EX_GAP
    W = MARGIN * 2 + panel_w
    # One band = caption, the input, then the prediction stacked beneath it.
    band_h = SUB_H + cp + VGAP + cp
    panel_h = HEAD_H + args.rows * band_h + (args.rows - 1) * ROW_GAP
    H = MARGIN * 2 + len(PANELS) * panel_h + (len(PANELS) - 1) * PANEL_GAP

    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    f_head, f_cap, f_row = _font(22, "Bold"), _font(16, "SemiBold"), _font(19, "Bold")

    def row_label(text, y_top, y_bottom):
        """Rotated label in the left gutter, reading bottom-to-top, centred on an image."""
        bbox = f_row.getbbox(text)
        strip = Image.new("RGBA", (bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4), (0, 0, 0, 0))
        ImageDraw.Draw(strip).text((2 - bbox[0], 2 - bbox[1]), text,
                                   font=f_row, fill=MUTED + (255,))
        strip = strip.rotate(90, expand=True)
        x = MARGIN + (LABEL_W - 14 - strip.width) // 2
        y = (y_top + y_bottom) // 2 - strip.height // 2
        canvas.paste(strip, (max(x, MARGIN), y), strip)

    def framed(img, x, y):
        canvas.paste(img, (x, y))
        draw.rectangle([x, y, x + img.width - 1, y + img.height - 1], outline=BORDER, width=1)

    for pi, (cls, title) in enumerate(PANELS):
        py = MARGIN + pi * (panel_h + PANEL_GAP)
        letter = chr(ord("A") + pi)
        draw.text((MARGIN, py), f"({letter})  {title}", font=f_head, fill=FG)
        draw.line([(MARGIN, py + HEAD_H - 11), (MARGIN + panel_w, py + HEAD_H - 11)],
                  fill=RULE, width=1)

        for i, stem in enumerate(picks[cls]):
            r, c = divmod(i, args.cols)
            x0 = MARGIN + LABEL_W + c * (cp + EX_GAP)
            band_y = py + HEAD_H + r * (band_h + ROW_GAP)
            y_in = band_y + SUB_H
            y_pred = y_in + cp + VGAP

            if c == 0:
                row_label("input", y_in, y_in + cp)
                row_label("prediction", y_pred, y_pred + cp)

            pair = load_pair(runs[stem], cp)
            if pair is None:
                draw.text((x0, band_y), f"{stem} — no prediction", font=f_cap, fill=MUTED)
                continue
            a, b = pair
            draw.text((x0, band_y), scene_of.get(stem) or stem, font=f_cap, fill=FG)
            framed(a, x0, y_in)
            framed(b, x0, y_pred)

    canvas.save(out_path, dpi=(300, 300))
    print(f"figure -> {out_path}  ({W}x{H})")


if __name__ == "__main__":
    main()
