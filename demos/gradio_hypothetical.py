"""
Gradio demo for ZWM hypothetical prediction.

Launch:
    cd ZWM
    python -m demos.gradio_hypothetical
    python -m demos.gradio_hypothetical --model_name awwkl/zwm-bvd-1b/model.pt

The model checkpoint is auto-downloaded from HuggingFace if not already present
under ./out/.
"""

import argparse
import datetime
import glob
import json
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(_REPO_ROOT, "tmp", "gradio"))
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from zwm.zwm_predictor import ZWMPredictor


arrow_color = (0, 255, 0)
dot_color = (0, 255, 0)
dot_color_fixed = (255, 0, 0)
thickness = 3
tip_length = 0.3
dot_radius = 7
dot_thickness = -1

DISPLAY_RES = 512


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="awwkl/zwm-bvd-170m/model.pt",
                        help="ZWM checkpoint name (path under ./out/ or HF-downloaded path)")
    parser.add_argument("--examples_dir", type=str, default="demos/assets/examples",
                        help="Directory of sample images displayed in the gallery")
    parser.add_argument("--share", default=True, help="Create a public shareable link")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Checkpoints offered in the model dropdown. "
                             "Default: every out/*/*/model.pt found, plus --model_name.")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(_REPO_ROOT, "demos", "outputs"),
                        help="Where the Save button writes captures. "
                             'Pass --save_dir "" to hide the button entirely.')
    return parser.parse_args()


_PREDICTORS = {}


def get_predictor(model_name: str) -> ZWMPredictor:
    """Load a checkpoint once and reuse it across model switches."""
    if model_name not in _PREDICTORS:
        print(f"Loading ZWM predictor: {model_name}")
        _PREDICTORS[model_name] = ZWMPredictor(model_name)
    return _PREDICTORS[model_name]


def discover_checkpoints():
    """Every model.pt under ./out/, as the <org>/<repo>/model.pt name ZWMPredictor wants."""
    out_root = os.path.join(_REPO_ROOT, "out")
    found = glob.glob(os.path.join(out_root, "*", "*", "model.pt"))
    return sorted(os.path.relpath(f, out_root) for f in found)


def _set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def resize_to_square(img, size=DISPLAY_RES):
    img = Image.fromarray(img)
    return np.array(transforms.Resize((size, size))(img))


def build_demo(predictor: ZWMPredictor, examples_dir: str,
               save_dir: str = "", model_name: str = "", model_choices=None):
    model_choices = model_choices or [model_name]
    patch_size_move_mult = 2
    # All released ZWM checkpoints share resolution=256 / patch_size=8, so the
    # click-to-patch geometry below is the same for every entry in the dropdown.
    model_patch_px = predictor.model.config.patch_size * patch_size_move_mult
    patch_viz_px = int(round(model_patch_px * DISPLAY_RES / predictor.model.config.resolution))
    rect_thickness = 2

    def draw_patch_rect(img, pt, color):
        x, y = pt
        cv2.rectangle(img, (x, y), (x + patch_viz_px, y + patch_viz_px), color, rect_thickness)

    with gr.Blocks() as demo:
        gr.Markdown("# Patch-translation Hypothetical with ZWM")

        with gr.Row():
            with gr.Column():
                original_image = gr.State(value=None)
                input_image = gr.Image(type="numpy", label="Upload Image")

                selected_points = gr.State([])
                with gr.Row():
                    gr.Markdown(
                        "1. **Click the image** to specify patch motion: first click = start, second click = end.\n"
                        "2. Enable **\"Select patches to be kept fixed\"** to mark an anchor point that should not move.\n"
                        "3. Click **\"Run ZWM\"** to generate the hypothetical frame."
                    )
                    with gr.Column():
                        zero_length_toggle = gr.Checkbox(label="Select patches to be kept fixed", value=False)
                        undo_button = gr.Button("Undo last action")
                        clear_button = gr.Button("Clear All")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Hypothetical Prediction")
                model_dropdown = gr.Dropdown(
                    choices=model_choices, value=model_name, label="Model",
                    info="First use of a checkpoint loads it (a few seconds "
                         "for 170M, longer for 1B); after that switching is instant.")
                model_status = gr.Markdown("")
                run_model_button = gr.Button("Run ZWM")
                seed_text = gr.Textbox(label="Seed", value="1110")

        def store_img(img):
            resized_img = resize_to_square(img, DISPLAY_RES)
            return resized_img, resized_img, []

        input_image.upload(store_img, [input_image], [input_image, original_image, selected_points])

        def get_point(img, sel_pix, zero_length, evt: gr.SelectData):
            sel_pix.append(evt.index)
            if zero_length:
                cv2.circle(img, sel_pix[-1], dot_radius, dot_color_fixed, dot_thickness, lineType=cv2.LINE_AA)
                draw_patch_rect(img, sel_pix[-1], dot_color_fixed)
                sel_pix.append(evt.index)
            else:
                if len(sel_pix) % 2 == 1:
                    cv2.circle(img, sel_pix[-1], dot_radius, dot_color, dot_thickness, lineType=cv2.LINE_AA)
                    draw_patch_rect(img, sel_pix[-1], dot_color)
                if len(sel_pix) % 2 == 0:
                    start_point = sel_pix[-2]
                    end_point = sel_pix[-1]
                    cv2.arrowedLine(img, start_point, end_point, arrow_color, thickness,
                                    tipLength=tip_length, line_type=cv2.LINE_AA)
                    cv2.circle(img, end_point, dot_radius, dot_color, dot_thickness, lineType=cv2.LINE_AA)
                    draw_patch_rect(img, end_point, dot_color)
            return img if isinstance(img, np.ndarray) else np.array(img)

        input_image.select(get_point, [input_image, selected_points, zero_length_toggle], [input_image])

        def undo_arrows(orig_img, sel_pix, zero_length):
            temp = orig_img.copy()
            if len(sel_pix) >= 2:
                sel_pix.pop()
                sel_pix.pop()
            for i in range(0, len(sel_pix), 2):
                start_point = sel_pix[i]
                end_point = sel_pix[i + 1]
                if start_point == end_point:
                    color = dot_color_fixed
                else:
                    cv2.arrowedLine(temp, start_point, end_point, arrow_color, thickness,
                                    tipLength=tip_length, line_type=cv2.LINE_AA)
                    color = arrow_color
                cv2.circle(temp, start_point, dot_radius, color, dot_thickness, lineType=cv2.LINE_AA)
                cv2.circle(temp, end_point, dot_radius, color, dot_thickness, lineType=cv2.LINE_AA)
                draw_patch_rect(temp, start_point, color)
                if start_point != end_point:
                    draw_patch_rect(temp, end_point, color)
            if len(sel_pix) == 1:
                cv2.circle(temp, sel_pix[0], dot_radius, dot_color, dot_thickness, lineType=cv2.LINE_AA)
                draw_patch_rect(temp, sel_pix[0], dot_color)
            return temp if isinstance(temp, np.ndarray) else np.array(temp)

        undo_button.click(undo_arrows, [original_image, selected_points, zero_length_toggle], [input_image])

        def clear_all_points(orig_img, sel_pix):
            sel_pix.clear()
            return orig_img

        clear_button.click(clear_all_points, [original_image, selected_points], [input_image])

        def load_selected(name):
            if not name:
                return "No model selected."
            already = name in _PREDICTORS
            p = get_predictor(name)
            n_params = sum(x.numel() for x in p.model.parameters()) / 1e6
            verb = "Ready (cached)" if already else "Loaded"
            return f"{verb}: `{name}` — {n_params:.0f}M params, "\
                   f"{p.model.config.resolution}px / patch {p.model.config.patch_size}."

        model_dropdown.change(load_selected, [model_dropdown], [model_status])

        def run_model_on_points(points, original_image_square, seed_text, name):
            predictor = get_predictor(name)
            model_res = predictor.model.config.resolution
            factor = model_res / DISPLAY_RES
            move_points = np.array(points).reshape(-1, 4) * factor

            _set_seed(int(seed_text))

            results = predictor.hypothetical_prediction(
                Image.fromarray(original_image_square),
                move_points,
                patch_size_move_mult=patch_size_move_mult,
            )
            pred_pil = results["frame1_pred_pil"]
            pred_pil = transforms.Resize((DISPLAY_RES, DISPLAY_RES))(pred_pil)
            return np.array(pred_pil)

        run_model_button.click(
            run_model_on_points,
            [selected_points, original_image, seed_text, model_dropdown],
            [output_image],
        )

        if save_dir:
            with gr.Row():
                save_name = gr.Textbox(label="Save name (optional)", value="", scale=3,
                                       placeholder="e.g. pulley_gemini_2_drag_heavy_down")
                save_button = gr.Button("Save inputs + output", scale=1)
            save_status = gr.Markdown("")

            def save_capture(annotated_img, clean_img, pred_img, points, seed, name,
                             used_model):
                if clean_img is None:
                    return "**Nothing to save** — load or upload an image first."
                os.makedirs(save_dir, exist_ok=True)
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                slug = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (name or "")).strip("_")
                stem = f"{stamp}_{slug}" if slug else stamp

                written = []
                Image.fromarray(clean_img).save(os.path.join(save_dir, f"{stem}_input.png"))
                written.append(f"{stem}_input.png")
                if annotated_img is not None:
                    Image.fromarray(annotated_img).save(
                        os.path.join(save_dir, f"{stem}_input_annotated.png"))
                    written.append(f"{stem}_input_annotated.png")
                if pred_img is not None:
                    Image.fromarray(pred_img).save(os.path.join(save_dir, f"{stem}_pred.png"))
                    written.append(f"{stem}_pred.png")
                else:
                    written.append("(no prediction yet — output not saved)")

                model_res = predictor.model.config.resolution
                pts = list(points) if points else []
                meta = {
                    "model_name": used_model,
                    "seed": seed,
                    "display_res": DISPLAY_RES,
                    "model_resolution": model_res,
                    "patch_size": predictor.model.config.patch_size,
                    "patch_size_move_mult": patch_size_move_mult,
                    "points_display_res": pts,
                    "move_points_model_res": (
                        (np.array(pts).reshape(-1, 4) * (model_res / DISPLAY_RES)).tolist()
                        if len(pts) >= 2 else []
                    ),
                }
                with open(os.path.join(save_dir, f"{stem}_meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
                written.append(f"{stem}_meta.json")
                return f"Saved to `{save_dir}`:\n\n" + "\n".join(f"- `{w}`" for w in written)

            save_button.click(
                save_capture,
                [input_image, original_image, output_image, selected_points, seed_text, save_name,
                 model_dropdown],
                [save_status],
            )

        gallery = gr.Gallery(value=[], columns=5, allow_preview=False,
                             label="Click an example to load it")

        def list_examples():
            paths = sorted(
                glob.glob(f"{examples_dir}/*.jpg")
                + glob.glob(f"{examples_dir}/*.png")
            )
            return [(p, os.path.splitext(os.path.basename(p))[0]) for p in paths]

        def load_example(evt: gr.SelectData):
            img_path = evt.value["image"]["path"]
            img = np.array(Image.open(img_path).convert("RGB"))
            resized_img = resize_to_square(img, DISPLAY_RES)
            return resized_img, resized_img, [], gr.update(value=None)

        gallery.select(
            load_example,
            inputs=[],
            outputs=[input_image, original_image, selected_points, output_image],
        )

        demo.load(fn=list_examples, inputs=[], outputs=gallery)

    return demo


def main():
    args = parse_args()
    choices = args.models if args.models else discover_checkpoints()
    if args.model_name not in choices:
        choices = [args.model_name] + choices
    print(f"Model dropdown: {choices}")
    predictor = get_predictor(args.model_name)
    demo = build_demo(predictor, args.examples_dir, args.save_dir, args.model_name,
                      model_choices=choices)
    demo.queue().launch(inbrowser=True, share=args.share)


if __name__ == "__main__":
    main()