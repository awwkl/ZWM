#!/usr/bin/env python3
"""Neural-alignment analyses for the ZWM paper.

Faithful, standalone reproduction of the figures previously made in
BBScore/test_eval.ipynb, now reading from a local copy of the eval table:

  Fig 6B - developmental trajectory: noise-corrected predictivity vs equivalent
           days of a child's waking experience, one line per NSD ROI.
  Fig 6C - layer-area correspondence: earliest encoder layer reaching the noise
           ceiling (else best layer) per ROI, one panel per model.
  Fig 6D - layer-wise profile: predictivity vs encoder depth, one figure per ROI,
           one line per model. Includes the V-JEPA2 baselines (R2.8).

The plotting logic mirrors the notebook cell-for-cell (same data cleaning, same
layer-depth normalisation `layer_num / depth`, same per-benchmark aggregation).

Usage:
    python analyze_neural_alignment.py
    python analyze_neural_alignment.py --score ridge_final_unceiled_pearson

Outputs land in scripts/eval/neural_alignment/plots/{fig6B,fig6C,fig6D}/.
"""
import argparse
import math
import os
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "data", "neural_eval.csv")
DEFAULT_OUT = os.path.join(HERE, "plots")

# Models shown in the layer-wise panels (notebook `models_to_plot`), with the
# published display name / colour / marker / linestyle. Colours follow the
# matplotlib default cycle order the notebook produced (C0..C5).
MODEL_STYLE = {
    "cwm_170m_babyview_one_image_ln_1": dict(name="BabyZWM (170M)",       color="tab:blue",   marker="o", ls="-."),
    "cwm_1b_babyview_one_image_ln_1":   dict(name="BabyZWM (1B)",         color="tab:orange", marker="o", ls="-."),
    "cwm_170m_one_image_ln_1":          dict(name="ZWM (170M)",           color="tab:green",  marker="s", ls="-"),
    "cwm_1b_one_image_ln_1":            dict(name="ZWM (1B)",             color="tab:red",    marker="s", ls="-"),
    "vjepa2_large_16_babyview":         dict(name="Baby V-JEPA 2 (300M)", color="tab:purple", marker="^", ls="-."),
    "vjepa2_large_16":                  dict(name="V-JEPA 2 (300M)",      color="tab:brown",  marker="^", ls="--"),
}
MODELS_TO_PLOT = list(MODEL_STYLE)

# Fig 6C uses a single model (BabyZWM 170M), NSD ROIs only.
FIG6C_MODEL = "cwm_170m_babyview_one_image_ln_1"

# Fig 6D published panel selection + titles (one row of 4 ROIs).
FIG6D_PANELS = ["NSDV1vShared", "NSDV2vShared", "NSDV4Shared", "NSDHighVentralShared"]
FIG6D_TITLES = {
    "NSDV1vShared": "V1 ventral (NSD human fMRI data)",
    "NSDV2vShared": "V2 ventral (NSD)",
    "NSDV4Shared": "V4 (NSD)",
    "NSDHighVentralShared": "High Ventral (NSD)",
}

# Benchmarks, ordered low -> high in the visual hierarchy (notebook order).
NSD_BENCHMARKS = [
    "NSDV1vShared", "NSDV1dShared", "NSDV2vShared", "NSDV2dShared",
    "NSDV3vShared", "NSDV3dShared", "NSDV4Shared",
    "NSDMidLateralShared", "NSDMidVentralShared", "NSDMidParietalShared",
    "NSDHighLateralShared", "NSDHighVentralShared", "NSDHighParietalShared",
]
TVSD_BENCHMARKS = ["TVSDV1", "TVSDV4", "TVSDIT"]
BENCHMARKS_TO_PLOT = NSD_BENCHMARKS + TVSD_BENCHMARKS

BENCHMARK_NAME = {
    "NSDV1vShared": "V1 ventral (NSD)", "NSDV1dShared": "V1 dorsal (NSD)",
    "NSDV2vShared": "V2 ventral (NSD)", "NSDV2dShared": "V2 dorsal (NSD)",
    "NSDV3vShared": "V3 ventral (NSD)", "NSDV3dShared": "V3 dorsal (NSD)",
    "NSDV4Shared": "V4 (NSD)",
    "NSDHighLateralShared": "High lateral (NSD)", "NSDHighVentralShared": "High ventral (NSD)",
    "NSDHighParietalShared": "High parietal (NSD)",
    "NSDMidLateralShared": "Mid lateral (NSD)", "NSDMidVentralShared": "Mid ventral (NSD)",
    "NSDMidParietalShared": "Mid parietal (NSD)",
    "TVSDV1": "V1 (TVSD)", "TVSDV4": "V4 (TVSD)", "TVSDIT": "IT (TVSD)",
}

# Baby-data penalty: (family label, standard model, baby-trained model) triples
# compared at matched architecture (same model, standard vs developmental diet).
PENALTY_PAIRS = [
    ("V-JEPA 2 (300M)", "vjepa2_large_16", "vjepa2_large_16_babyview"),
    ("ZWM (170M)", "cwm_170m_one_image_ln_1", "cwm_170m_babyview_one_image_ln_1"),
    ("ZWM (1B)", "cwm_1b_one_image_ln_1", "cwm_1b_babyview_one_image_ln_1"),
]
# Ventral-stream progression (NSD) + macaque (TVSD), early -> high.
PENALTY_ROIS = [
    "NSDV1vShared", "NSDV2vShared", "NSDV3vShared", "NSDV4Shared",
    "NSDMidVentralShared", "NSDHighVentralShared",
    "TVSDV1", "TVSDV4", "TVSDIT",
]
# ROIs averaged into the "higher-order areas" headline row.
PENALTY_HIGH_ROIS = ["NSDV4Shared", "NSDMidVentralShared", "NSDHighVentralShared", "TVSDV4", "TVSDIT"]

# Fig 6B developmental checkpoints (steps -> model id).
DEVELOPMENTAL_MODELS = {
    0: "cwm_170m_babyview_0k_one_image_ln_1",
    5000: "cwm_170m_babyview_5k_one_image_ln_1",
    10000: "cwm_170m_babyview_10k_one_image_ln_1",
    20000: "cwm_170m_babyview_20k_one_image_ln_1",
    40000: "cwm_170m_babyview_40k_one_image_ln_1",
    80000: "cwm_170m_babyview_80k_one_image_ln_1",
    120000: "cwm_170m_babyview_120k_one_image_ln_1",
    200000: "cwm_170m_babyview_one_image_ln_1",
}
STEPS_TO_DAYS = 95.0 / 200_000.0  # 200k steps == 95 days of waking experience


def setup_fonts():
    """Use Source Sans 3 with editable PDF text if the fonts are installed."""
    import matplotlib.font_manager as fm

    font_files = [
        "/ccn2/u/khaiaw/.local/share/fonts/source-sans/SourceSans3-Bold.ttf",
        "/ccn2/u/khaiaw/.local/share/fonts/source-sans/SourceSans3-SemiBold.ttf",
        "/ccn2/u/khaiaw/.local/share/fonts/source-sans/SourceSans3-Italic.ttf",
        "/ccn2/u/khaiaw/.local/share/fonts/source-sans/SourceSans3-Regular.ttf",
    ]
    family = "DejaVu Sans"
    if all(os.path.exists(f) for f in font_files):
        for f in font_files:
            fm.fontManager.addfont(f)
        family = "Source Sans 3"
    mpl.rcParams.update({
        "font.family": family,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,
    })


def load_table(csv_path, score_name):
    """Load and clean the eval table, mirroring test_eval.ipynb prep.

    - keep only rows with a finite score
    - fold the `.ln_1` layer-norm read-out into the model name
    - deduplicate (model, layer, benchmark), keeping the latest timestamp
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=[score_name])
    is_ln1 = df["layer"].astype(str).str.contains(r"\.ln_1", regex=True)
    df["model"] = np.where(is_ln1, df["model"] + "_ln_1", df["model"])
    df["layer"] = df["layer"].str.replace(".ln_1", "", regex=False)
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["model", "layer", "benchmark"], keep="last"
    )
    return df


def layer_percentage(model, layer):
    """Fractional encoder depth (notebook `layer_percentage`): layer_num / depth.

    None for predictor layers / unrecognised models so they drop out of the axis.
    """
    if not isinstance(layer, str) or "predictor" in layer:
        return None
    m = re.search(r"(\d+)$", layer)
    if m is None:
        return None
    num = int(m.group(1))
    ml = model.lower()
    if "vjepa2_large" in ml:
        return num / 24.0
    if "vjepa2_great" in ml:
        return num / 40.0
    if "170m" in model:
        return num / 24.0
    if "1b" in model:
        return num / 48.0
    if "psi_7b" in ml:
        return num / 32.0
    if "videomae_large" in ml:
        return num / 48.0
    return -1


def model_layer_curve(df, model, benchmark, score_name):
    """(layer_percentage, score) for one model/benchmark, sorted by depth."""
    sub = df[(df["model"] == model) & (df["benchmark"] == benchmark)].copy()
    if sub.empty:
        return sub
    sub["layer_percentage"] = [layer_percentage(model, l) for l in sub["layer"]]
    sub = sub.dropna(subset=["layer_percentage"])
    return sub.sort_values("layer_percentage")


def _draw_layerwise_axis(ax, df, benchmark, score_name):
    """Draw one ROI's layer-wise curves (one line per model) onto an axis."""
    for model, st in MODEL_STYLE.items():
        d = model_layer_curve(df, model, benchmark, score_name)
        if d.empty:
            continue
        ax.plot(d["layer_percentage"], d[score_name], marker=st["marker"],
                linestyle=st["ls"], color=st["color"], label=st["name"], markersize=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.25)
    ax.set_xticks(np.arange(0, 1.0001, 0.2))
    ax.set_xlabel("Layer Percentage (0 = first layer, 1 = last layer)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_fig6D_layerwise(df, score_name, out_dir):
    """Fig 6D: predictivity vs encoder depth, published 4-panel row + all ROIs.

    Reproduces notebook cell 14, laid out as the published figure: one row of
    [V1 ventral, V2 ventral, V4, High ventral] with a shared legend on top using
    the model display names. Individual per-ROI panels are also written to all/.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- published 4-panel row ---
    fig, axes = plt.subplots(1, len(FIG6D_PANELS), figsize=(4.2 * len(FIG6D_PANELS), 4.6),
                             sharey=True)
    for ax, b in zip(axes, FIG6D_PANELS):
        _draw_layerwise_axis(ax, df, b, score_name)
        ax.set_title(FIG6D_TITLES.get(b, BENCHMARK_NAME.get(b, b)))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    png = os.path.join(out_dir, "fig6D_layerwise.png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    # --- supplementary: every ROI as its own panel ---
    all_dir = os.path.join(out_dir, "all")
    os.makedirs(all_dir, exist_ok=True)
    for b in BENCHMARKS_TO_PLOT:
        fig, ax = plt.subplots(figsize=(5, 5))
        _draw_layerwise_axis(ax, df, b, score_name)
        ax.set_title(BENCHMARK_NAME.get(b, b))
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(all_dir, f"line_{b}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("saved Fig 6D ->", png)


def plot_fig6C_layer_area(df, score_name, out_dir, model=FIG6C_MODEL,
                          title="First layer to reach noise ceiling", fname=None):
    """Fig 6C: first encoder layer to reach the noise ceiling per NSD ROI.

    For each ROI, the bar is the depth of the earliest layer at the ceiling (else
    the best layer). Hierarchical correspondence = depth rising from early (blue)
    to mid/high (green) areas. Reproduces notebook cell 15. Returns the bar
    values, or None if the model has no data.
    """
    os.makedirs(out_dir, exist_ok=True)
    if df[df["model"] == model].empty:
        return None
    bar_vals = []
    for bench in NSD_BENCHMARKS:
        d = model_layer_curve(df, model, bench, score_name)
        if d.empty:
            bar_vals.append(np.nan)
            continue
        hit = d[d[score_name] >= 1.0]
        if not hit.empty:                                  # earliest layer at ceiling
            best = hit.loc[hit["layer_percentage"].idxmin()]
        else:                                              # else best score (earliest tie)
            tied = d[d[score_name] == d[score_name].max()]
            best = tied.loc[tied["layer_percentage"].idxmin()]
        bar_vals.append(float(best["layer_percentage"]))

    y = np.arange(len(NSD_BENCHMARKS))
    colors = ["tab:green" if (("High" in b) or ("Mid" in b)) else "tab:blue"
              for b in NSD_BENCHMARKS]
    fig, ax = plt.subplots(figsize=(4.5, 6))
    ax.barh(y, np.nan_to_num(bar_vals, nan=0.0), color=colors)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlim(0, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([BENCHMARK_NAME.get(b, b) for b in NSD_BENCHMARKS])
    ax.invert_yaxis()
    ax.set_xlabel("Layer Percentage (0 = first, 1 = last layer)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    png = os.path.join(out_dir, fname or "fig6C_first_layer_ceiling.png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved Fig 6C ->", png)
    return bar_vals


# Baselines whose converged (best-layer) predictivity is shown as endpoint markers
# in the extended Fig 6B margin. BabyZWM 170M is the trajectory itself, so it is
# not repeated here.
FIG6B_MARGIN_MODELS = [
    "cwm_1b_babyview_one_image_ln_1",
    "cwm_170m_one_image_ln_1",
    "cwm_1b_one_image_ln_1",
    "vjepa2_large_16_babyview",
    "vjepa2_large_16",
]


def _fig6B_area_style():
    """Shared area-group config for the Fig 6B trajectory and its extended variant.

    Returns (AREA_GROUPS, group_names, bench_to_group, ordered_benches,
    group_color, group_marker): lines/markers are coloured by area group (V1..High)
    with a distinct marker per benchmark within a group.
    """
    AREA_GROUPS = [
        ("V1", ["NSDV1vShared", "NSDV1dShared"]),
        ("V2", ["NSDV2vShared", "NSDV2dShared"]),
        ("V3", ["NSDV3vShared", "NSDV3dShared"]),
        ("V4", ["NSDV4Shared"]),
        ("Mid", ["NSDMidLateralShared", "NSDMidVentralShared", "NSDMidParietalShared"]),
        ("High", ["NSDHighLateralShared", "NSDHighVentralShared", "NSDHighParietalShared"]),
    ]
    group_names = [g for g, _ in AREA_GROUPS]
    bench_to_group = {b: g for g, bs in AREA_GROUPS for b in bs}
    ordered_benches = [b for _, bs in AREA_GROUPS for b in bs]

    cmap = mpl.colormaps["RdBu"].resampled(len(group_names))
    group_color = {g: cmap(i) for i, g in enumerate(group_names)}
    marker_pool = ["o", "s", "^", "D", "x", "P", "*", "v", "<", ">"]
    group_marker = {}
    for g, bs in AREA_GROUPS:
        group_marker.update({b: marker_pool[i % len(marker_pool)] for i, b in enumerate(bs)})
    return (AREA_GROUPS, group_names, bench_to_group, ordered_benches,
            group_color, group_marker)


def plot_fig6B_developmental(df, score_name, out_dir):
    """Fig 6B: predictivity vs equivalent days, one line per NSD ROI (notebook cell 18).

    Lines are coloured by area group (V1..High, shared colour) with a distinct
    marker per benchmark within a group. Score = max across layers per checkpoint.
    """
    os.makedirs(out_dir, exist_ok=True)
    (_, group_names, bench_to_group, ordered_benches,
     group_color, group_marker) = _fig6B_area_style()

    ckpts = sorted(DEVELOPMENTAL_MODELS)
    plt.figure(figsize=(7, 7))
    handles = {}
    for b in ordered_benches:
        g = bench_to_group[b]
        xs, ys = [], []
        for ck in ckpts:
            sub = df[(df["model"] == DEVELOPMENTAL_MODELS[ck]) & (df["benchmark"] == b)]
            xs.append(ck * STEPS_TO_DAYS)
            ys.append(sub[score_name].max() if not sub.empty else np.nan)
        (h,) = plt.plot(xs, ys, linestyle="-", marker=group_marker[b],
                        color=group_color[g], label=b)
        handles[b] = h

    if score_name == "ridge_final_pearson":
        plt.axhline(1.0, linestyle="-", color="red", linewidth=2, label="Ceiling = 1.0")

    plt.xlim(0, 100)
    plt.ylim(0.5, 1.09)
    plt.xticks(np.arange(0, 101, 20), fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Training progress: equivalent days of child's waking experience")
    plt.title("Developmental trajectory of noise-corrected predictivity")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_order = [b for g in group_names for b in ordered_benches if bench_to_group[b] == g]
    plt.legend([handles[b] for b in legend_order], legend_order, title="Benchmark", ncol=1)
    plt.tight_layout()
    out = os.path.join(out_dir, f"fig6B_developmental_{score_name}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("saved Fig 6B ->", out)


def plot_fig6B_extended(df, score_name, out_dir, collapse_groups=False):
    """Extended Fig 6B (Supplementary): BabyZWM's developmental trajectory, with the
    baselines' converged (best-layer) predictivity added as endpoint markers in a
    horizontally extended margin past the end of training.

    The trajectory uses the same colour (area group) / marker scheme as
    `plot_fig6B_developmental`, and the baseline endpoints are plotted on the same
    predictivity axis, so the reader sees where each baseline's converged alignment
    lands relative to BabyZWM. The baselines have no comparable training-progress axis
    (it is calibrated to BabyZWM's developmental diet), so they appear only as
    converged endpoints rather than trajectories (rebuttal R2.8).

    collapse_groups=False  -> per-ROI lines/markers (13 NSD ROIs), matches main-text 6B.
    collapse_groups=True   -> both trajectory and margin reduced to the 6 area-group
                              means (one line/marker per group), for a cleaner panel.
    """
    os.makedirs(out_dir, exist_ok=True)
    (AREA_GROUPS, group_names, bench_to_group, ordered_benches,
     group_color, group_marker) = _fig6B_area_style()
    group_benches = {g: bs for g, bs in AREA_GROUPS}
    grp_marker_pool = ["o", "s", "^", "D", "v", "P"]
    group_single_marker = {g: grp_marker_pool[i % len(grp_marker_pool)]
                           for i, g in enumerate(group_names)}

    def _maxscore(model, bench):
        sub = df[(df["model"] == model) & (df["benchmark"] == bench)]
        return sub[score_name].max() if not sub.empty else np.nan

    def _group_mean(model, g):
        vals = [v for v in (_maxscore(model, b) for b in group_benches[g]) if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan

    ckpts = sorted(DEVELOPMENTAL_MODELS)
    fig, ax = plt.subplots(figsize=(10, 7))

    # --- BabyZWM developmental trajectory ---
    handles = {}
    if collapse_groups:
        for g in group_names:
            xs = [ck * STEPS_TO_DAYS for ck in ckpts]
            ys = [_group_mean(DEVELOPMENTAL_MODELS[ck], g) for ck in ckpts]
            (h,) = ax.plot(xs, ys, linestyle="-", marker=group_single_marker[g],
                           color=group_color[g], label=g, markersize=7)
            handles[g] = h
    else:
        for b in ordered_benches:
            xs = [ck * STEPS_TO_DAYS for ck in ckpts]
            ys = [_maxscore(DEVELOPMENTAL_MODELS[ck], b) for ck in ckpts]
            (h,) = ax.plot(xs, ys, linestyle="-", marker=group_marker[b],
                           color=group_color[bench_to_group[b]], label=b)
            handles[b] = h
    if score_name == "ridge_final_pearson":
        ax.axhline(1.0, linestyle="-", color="red", linewidth=2)

    # --- baselines' converged predictivity as endpoints in the extended margin ---
    traj_end = max(ckpts) * STEPS_TO_DAYS          # 95 days, where BabyZWM's curve ends
    divider = traj_end + 6
    col_gap = 11
    col_xs = [divider + col_gap * (i + 1) for i in range(len(FIG6B_MARGIN_MODELS))]
    right = col_xs[-1] + col_gap

    ax.axvspan(divider, right, color="gray", alpha=0.06, zorder=0)
    ax.axvline(divider, color="gray", linestyle=":", linewidth=1)
    for cx, m in zip(col_xs, FIG6B_MARGIN_MODELS):
        ax.axvline(cx, color="gray", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
        if collapse_groups:
            for g in group_names:
                y = _group_mean(m, g)
                if np.isfinite(y):
                    ax.scatter(cx, y, marker=group_single_marker[g], color=group_color[g],
                               s=70, edgecolor="black", linewidth=0.5, zorder=3)
        else:
            for b in ordered_benches:
                y = _maxscore(m, b)
                if np.isfinite(y):
                    ax.scatter(cx, y, marker=group_marker[b], color=group_color[bench_to_group[b]],
                               s=45, edgecolor="black", linewidth=0.4, zorder=3)
        ax.text(cx, -0.012, MODEL_STYLE[m]["name"], transform=ax.get_xaxis_transform(),
                rotation=30, ha="right", va="top", fontsize=9)
    ax.text((divider + right) / 2, 0.985, "Baselines (converged)",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=11, fontstyle="italic")

    ax.set_xlim(0, right)
    ax.set_ylim(0.5, 1.25)                          # taller than main-text 6B: baselines exceed 1.0
    ax.set_xticks(np.arange(0, int(traj_end) + 1, 20))
    ax.tick_params(axis="both", labelsize=13)
    ax.set_xlabel("Training progress: equivalent days of child's waking experience")
    ax.xaxis.set_label_coords(traj_end / (2 * right), -0.10)
    ax.set_ylabel("Noise-corrected neural predictivity\n(NSD human fMRI)")
    suffix = " (area-group means)" if collapse_groups else ""
    ax.set_title("Developmental trajectory of BabyZWM neural alignment vs. converged baselines" + suffix)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if collapse_groups:
        ax.legend([handles[g] for g in group_names], group_names,
                  title="Area group (NSD)", ncol=1, fontsize=9,
                  loc="center left", bbox_to_anchor=(1.02, 0.5))
    else:
        legend_order = [b for g in group_names for b in ordered_benches if bench_to_group[b] == g]
        ax.legend([handles[b] for b in legend_order],
                  [BENCHMARK_NAME.get(b, b) for b in legend_order],
                  title="Benchmark (NSD ROI)", ncol=1, fontsize=8,
                  loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    tag = "_collapsed" if collapse_groups else ""
    out = os.path.join(out_dir, f"fig6B_extended{tag}_{score_name}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved Fig 6B (extended%s) ->" % (" collapsed" if collapse_groups else ""), out)


def peak_score(df, model, benchmark, score_name):
    """Best-layer (max over layers) predictivity for one model/benchmark."""
    vals = df[(df["model"] == model) & (df["benchmark"] == benchmark)][score_name]
    return float(vals.max()) if not vals.empty else np.nan


def baby_data_penalty_table(df, score_name, out_dir):
    """Tabulate the baby-data penalty: peak predictivity loss (standard -> baby).

    Writes a tidy CSV plus rendered Markdown and LaTeX tables, and prints the
    headline averaged-over-higher-areas drop per family.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for family, std_m, baby_m in PENALTY_PAIRS:
        for roi in PENALTY_ROIS:
            std = peak_score(df, std_m, roi, score_name)
            baby = peak_score(df, baby_m, roi, score_name)
            rows.append({
                "family": family, "roi": BENCHMARK_NAME.get(roi, roi),
                "standard": std, "baby": baby,
                "delta": baby - std,
                "delta_pct": 100.0 * (baby - std) / std if std and np.isfinite(std) else np.nan,
            })
    tbl = pd.DataFrame(rows)
    csv = os.path.join(out_dir, "baby_data_penalty.csv")
    tbl.to_csv(csv, index=False)

    # Wide view: Δ% per ROI, one column per family (the published table).
    wide = tbl.pivot(index="roi", columns="family", values="delta_pct")
    wide = wide.reindex([BENCHMARK_NAME.get(r, r) for r in PENALTY_ROIS])
    wide = wide[[f for f, _, _ in PENALTY_PAIRS]]

    # Headline: mean drop over higher-order areas.
    high_names = [BENCHMARK_NAME.get(r, r) for r in PENALTY_HIGH_ROIS]
    headline = {f: tbl[(tbl.family == f) & (tbl.roi.isin(high_names))]["delta_pct"].mean()
                for f, _, _ in PENALTY_PAIRS}

    with open(os.path.join(out_dir, "baby_data_penalty.md"), "w") as fh:
        fh.write("# Baby-data penalty (Δ% peak neural predictivity, baby − standard)\n\n")
        fh.write(wide.round(1).to_markdown())
        fh.write("\n\n**Mean over higher-order areas (V4, Mid/High ventral, TVSD V4/IT):**\n\n")
        for f in headline:
            fh.write(f"- {f}: {headline[f]:+.1f}%\n")
    with open(os.path.join(out_dir, "baby_data_penalty.tex"), "w") as fh:
        fh.write(wide.round(1).to_latex(float_format="%+.1f"))

    print("saved baby-data penalty ->", csv)
    print("  mean Δ% over higher-order areas:",
          {f: round(v, 1) for f, v in headline.items()})
    return tbl, wide, headline


def plot_baby_data_penalty(df, score_name, out_dir):
    """Bar plot of the baby-data penalty (Δ% peak predictivity) per ROI."""
    os.makedirs(out_dir, exist_ok=True)
    fam_color = {"V-JEPA 2 (300M)": "tab:brown", "ZWM (170M)": "tab:green", "ZWM (1B)": "tab:red"}
    fams = [f for f, _, _ in PENALTY_PAIRS]
    x = np.arange(len(PENALTY_ROIS))
    w = 0.8 / len(fams)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (family, std_m, baby_m) in enumerate(PENALTY_PAIRS):
        deltas = [100.0 * (peak_score(df, baby_m, r, score_name) - peak_score(df, std_m, r, score_name))
                  / peak_score(df, std_m, r, score_name) for r in PENALTY_ROIS]
        ax.bar(x + i * w, deltas, w, label=family, color=fam_color.get(family))
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x + w * (len(fams) - 1) / 2)
    ax.set_xticklabels([BENCHMARK_NAME.get(r, r) for r in PENALTY_ROIS], rotation=45, ha="right")
    ax.set_ylabel("Baby-data penalty\n(Δ% peak predictivity)")
    ax.set_title("ZWM is robust to a developmental data diet; V-JEPA2 is not")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    png = os.path.join(out_dir, "baby_data_penalty.png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved baby-data penalty plot ->", png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--score", default="ridge_final_pearson",
                    choices=["ridge_final_pearson", "ridge_final_unceiled_pearson"])
    args = ap.parse_args()

    setup_fonts()
    df = load_table(args.csv, args.score)

    plot_fig6B_developmental(df, args.score, os.path.join(args.out, "fig6B"))
    plot_fig6B_extended(df, args.score, os.path.join(args.out, "fig6B"))
    plot_fig6B_extended(df, args.score, os.path.join(args.out, "fig6B"), collapse_groups=True)
    plot_fig6C_layer_area(df, args.score, os.path.join(args.out, "fig6C"))
    plot_fig6D_layerwise(df, args.score, os.path.join(args.out, "fig6D"))

    # --- 6C for every model in the comparison set ---
    c_all = os.path.join(args.out, "fig6C_all_models")
    for model, st in MODEL_STYLE.items():
        ok = plot_fig6C_layer_area(
            df, args.score, c_all, model=model,
            title=f"First layer to reach noise ceiling - {st['name']}",
            fname=f"fig6C_{model}.png",
        )
        if ok is None:
            print("  (skipped 6C, no data):", model)

    # --- 6B (developmental) only exists for the BabyZWM 170M checkpoint series ---
    # No other model in the comparison set has intermediate training checkpoints
    # in the eval table, so a per-model developmental trajectory is not possible.
    have_ckpts = [m for m in MODEL_STYLE if m in DEVELOPMENTAL_MODELS.values()]
    print(f"6B (developmental) available only for: {have_ckpts} "
          f"(no checkpoint snapshots for the other models)")

    # --- baby-data penalty table + plot ---
    pen_dir = os.path.join(args.out, "baby_data_penalty")
    baby_data_penalty_table(df, args.score, pen_dir)
    plot_baby_data_penalty(df, args.score, pen_dir)


if __name__ == "__main__":
    main()
