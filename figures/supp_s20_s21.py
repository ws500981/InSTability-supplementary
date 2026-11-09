"""Reproduce Supplementary Figures S20 and S21 from saved DEG summaries."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import dgea_data
import argparse
from common import OUTPUT_ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = OUTPUT_ROOT / "supp_s20_s21"
SAVE_FORMATS = ["png", "pdf"]
N_RUNS = 50
METHODS = [
    "leiden", "louvain", "bayesspace", "graphst", "graphst_mclust",
    "sedr", "sedr_mclust", "stagate", "stagate_mclust", "spicemix",
]
METHOD_DISPLAY = {
    "leiden": "Leiden", "louvain": "Louvain", "bayesspace": "BayesSpace",
    "graphst": "GraphST", "graphst_mclust": "GraphST (mclust)",
    "sedr": "SEDR", "sedr_mclust": "SEDR (mclust)",
    "stagate": "STAGATE", "stagate_mclust": "STAGATE (mclust)",
    "spicemix": "SpiceMix",
}
DATASET_SAMPLES = [
    ("dlpfc", sample) for sample in (
        "151507", "151508", "151509", "151510", "151669", "151670",
        "151671", "151672", "151673", "151674", "151675", "151676",
    )
] + [
    ("mouse_brain", None), ("mouse_brain_cerebellum", None),
    ("human_breast_cancer", None),
    ("Visium_HD_Human_Colon_Cancer_cropped_square", None),
    ("ov_ffpe", None), ("coad_ffpe", None),
]
TOP_KS = [10, 20, 50, 200]
BIN_COLORS = {
    "[0.0,0.1)": "#08306b", "[0.1,0.2)": "#4292c6",
    "[0.2,0.3)": "#9ecae1", "[0.3,0.4)": "#deebf7",
    "[0.4,0.5)": "#fee391", "[0.5,0.6)": "#fec44f",
    "[0.6,0.7)": "#fdae6b", "[0.7,0.8)": "#fd8d3c",
    "[0.8,0.9)": "#fb6a4a", "[0.9,1.0]": "#cb181d",
}
FINE_BINS = list(BIN_COLORS)
FIGSIZE = (9.78, 18 * len(METHODS) / 7)
FIG_DPI = 600
SAVE_PDF = True
RANK_AXIS_DEFAULTS = {10: (6, 2), 20: (9, 3), 50: (25, 5), 200: (80, 20)}
METRICS = {
    "jaccard": {"column": "jaccard_mean", "ylabel": "Jaccard", "ylim": (0, 1)},
    "abs_rank_diff": {
        "column": "rank_abs_diff_shared_mean", "ylabel": "Abs rank diff", "ylim": None,
    },
}


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sample_id(dataset, sample):
    return dataset if sample is None else f"{dataset}_{sample}"




def rank_axis_limits(method_data):
    limits = {}
    for k, (old_max, step) in RANK_AXIS_DEFAULTS.items():
        values = np.concatenate([frame.loc[frame["top_k"].eq(k), "rank_abs_diff_shared_mean"].to_numpy(float)
                                 for frame in method_data.values()])
        values = values[np.isfinite(values)]
        data_max = float(values.max()) if len(values) else 0.0
        limits[k] = max(old_max, int(np.ceil(data_max / step)) * step)
        log(f"Top-{k} shared-rank axis: 0..{limits[k]} (data maximum {data_max:.6g})")
    return limits


def finite_values(df, fine_bin, column):
    values = df.loc[df["contrast_fine_bin"].eq(fine_bin), column].to_numpy(float)
    return values[np.isfinite(values)]


def draw_panel(mi, ax, pooled, top_k, metric_name, metric, show_ylabel, rank_limits):
    column = metric["column"]
    sub = pooled.loc[pooled["top_k"].eq(top_k)]
    populated_bins = [
        b for b in FINE_BINS
        if finite_values(sub, b, column).size
    ]

    if not populated_bins:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"No Top-{top_k} data",
            ha="center",
            va="center",
        )
        return

    max_bin_idx = max(FINE_BINS.index(b) for b in populated_bins)
    bins = FINE_BINS[:max_bin_idx + 1]

    positions = np.arange(1, len(bins) + 1)
    box_data = [finite_values(sub, b, column) for b in bins]
    colors = [BIN_COLORS[b] for b in bins]


    boxplot = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.64,
        patch_artist=True,
        showfliers=True,
        manage_ticks=False,
        whis=1.5,
    )

    for box, fine_bin in zip(boxplot["boxes"], bins):
        box.set(facecolor=BIN_COLORS[fine_bin], alpha=0.82,
                edgecolor="black", linewidth=1.1)

    for key in ["whiskers", "caps"]:
        for line in boxplot[key]:
            line.set(color="black", linewidth=1.1)

    for median in boxplot["medians"]:
        median.set(color="black", linewidth=1.5)

    for flier in boxplot["fliers"]:
        flier.set(color="black", markersize=4)


    ax.tick_params(axis='x', labelsize=12)
    ax.set_xticks(positions)

    if mi == len(METHODS) - 1:
        ax.set_xticklabels(bins,
            rotation=45,
            ha="right")
    else:
        ax.set_xticklabels([],
            rotation=45,
            ha="right")
    ax.set_xlim(0.25, len(bins) + 0.75)

    ax.tick_params(axis='y', labelsize=14)
    if metric_name == "jaccard":
        ax.set_ylim(0, 1.0)
        if show_ylabel:
            ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
        else:
            ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
            ax.set_yticklabels([])
    else:
        ymax_data = max(np.max(finite_values(sub, b, column)) for b in populated_bins)


        ymax = rank_limits[top_k]
        step = RANK_AXIS_DEFAULTS[top_k][1]
        ax.set_ylim(0, ymax)
        ax.set_yticks(np.arange(0, ymax + step / 2, step))
        require(ymax_data <= ymax,
                f"Top-{top_k}: shared-rank value {ymax_data} exceeds prepared limit {ymax}")


    if mi == 0:
        ax.set_title(f"Top-{top_k} DEGs", fontsize=18, fontweight="bold")

    if show_ylabel:
        ax.set_ylabel(metric["ylabel"], fontsize=14)

    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.35)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if metric_name == "jaccard":
        Ns = [str(len(box)) for box in box_data]
        text = "(" + r"$N=$" + ",".join(Ns) + ")"
        ax.text(0.75, 0.02, text, fontsize=8, ha="left", va="bottom", clip_on=False)

        if top_k == 10:
            text = f"{METHOD_DISPLAY[METHODS[mi]]}"
            ax.text(-3.5, 0.5, text, fontsize=19, fontweight="bold", ha="right", va="center", clip_on=False)
    else:
        if top_k == 10:
            text = f"{METHOD_DISPLAY[METHODS[mi]]}"
            ax.text(-3, rank_limits[top_k] / 2, text, fontsize=19, fontweight="bold", ha="right", va="center", clip_on=False)


def metric_counts(method_data):
    rows = []
    for method, frame in method_data.items():
        for k in TOP_KS:
            for fine_bin in FINE_BINS:
                subset = frame.loc[frame["top_k"].eq(k) & frame["contrast_fine_bin"].eq(fine_bin)]
                for metric_name, metric in METRICS.items():
                    denominator = ("jaccard_n_valid" if metric_name == "jaccard"
                                   else "rank_abs_diff_shared_mean_n_valid")
                    rows.append(dict(
                        method=method, top_k=k, example_fine_bin=fine_bin, metric=metric_name,
                        n_example_clusters_total=len(subset),
                        n_example_clusters_plotted=int(np.isfinite(subset[metric["column"]]).sum()),
                        n_comparisons_expected=int(subset["n_comparisons_expected"].sum()),
                        n_comparisons_available=int(subset["n_comparisons_available"].sum()),
                        n_metric_comparisons_valid=int(subset[denominator].sum()),
                    ))
    return pd.DataFrame(rows)


def save_figure(fig, metric_name):
    number = 's20' if metric_name == 'jaccard' else 's21'
    for extension in SAVE_FORMATS:
        path = OUTPUT_DIR / f'{number}_dgea_consistency_{metric_name}.{extension}'
        fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
        print(f'Saved {path}')


def plot_metric(metric_name, method_data, rank_limits):
    metric = METRICS[metric_name]
    fig, axes = plt.subplots(nrows=len(METHODS), ncols=len(TOP_KS),
                             figsize=FIGSIZE, constrained_layout=True)
    try:
        for mi, method in enumerate(METHODS):
            log(f"Plotting {metric_name}: {method}")
            for index, top_k in enumerate(TOP_KS):
                draw_panel(mi, axes[mi][index], method_data[method], top_k,
                           metric_name, metric, show_ylabel=index == 0, rank_limits=rank_limits)

        fig.tight_layout(rect=(0, 0.33, 1, 1))
        fig.subplots_adjust(wspace=0.2, hspace=0.4)
        save_figure(fig, metric_name)
    finally:
        plt.close(fig)


def main(argv=None):
    global OUTPUT_DIR, FIG_DPI, SAVE_FORMATS
    parser = argparse.ArgumentParser(description='Plot Supplementary Figures S20 and S21.')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--dpi', type=int, default=FIG_DPI)
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf'], default=SAVE_FORMATS)
    parser.add_argument('--metrics', nargs='+', choices=list(METRICS), default=list(METRICS))
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error('--dpi must be positive')
    OUTPUT_DIR, FIG_DPI, SAVE_FORMATS = args.output_dir, args.dpi, args.formats
    plt.rcParams['font.family'] = 'Arial'
    pooled = dgea_data.consistency().copy()
    pooled['contrast_fine_bin'] = pooled['example_fine_bin']
    require(set(pooled['method']) == set(METHODS), 'Expected all ten methods')
    method_data = {}
    expected_samples = {sample_id(*pair) for pair in DATASET_SAMPLES}
    for method in METHODS:
        frame = pooled.loc[pooled['method'].eq(method)].copy()
        require(set(frame['sample_id']) == expected_samples, f'{method}: expected all 18 samples')
        keys = ['sample_id', 'example_run_idx', 'example_cluster']
        require(not frame.duplicated(keys + ['top_k']).any(), f'{method}: duplicate cluster/Top-K rows')
        require(frame.groupby(keys)['top_k'].apply(lambda x: set(x) == set(TOP_KS)).all(),
                f'{method}: missing Top-K values')
        for metric, count in [('jaccard_mean', 'jaccard_n_valid'),
                              ('rank_abs_diff_shared_mean', 'rank_abs_diff_shared_mean_n_valid')]:
            require(np.isfinite(frame[metric]).eq(frame[count].gt(0)).all(),
                    f'{method}: missing values disagree with valid counts')
        method_data[method] = frame
    rank_limits = rank_axis_limits(method_data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_counts(method_data).to_csv(OUTPUT_DIR / 'counts.csv', index=False)
    for metric_name in args.metrics:
        plot_metric(metric_name, method_data, rank_limits)


if __name__ == "__main__":
    main()
