"""Reproduce Supplementary Figure S19 from saved spatial matches."""

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
from scipy.stats import pearsonr, spearmanr


OUTPUT_DIR = OUTPUT_ROOT / "supp_s19"
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
FINE_BINS = [f"[{i / 10:.1f},{(i + 1) / 10:.1f})" for i in range(9)] + ["[0.9,inf)"]
BIN_COLORS = dict(zip(FINE_BINS, [
    "#08306b", "#4292c6", "#9ecae1", "#deebf7", "#fee391",
    "#fec44f", "#fdae6b", "#fd8d3c", "#fb6a4a", "#cb181d",
]))
METHOD_GRID = (5, 2)
FIGSIZE = (9.78 * METHOD_GRID[1], 24 * METHOD_GRID[0] / 7)
FIG_DPI = 600
RANDOM_SEED = 42
FIGURE_STEM = "s19_cross-run-consistency-vs-instability"


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def sample_id(dataset, sample):
    return dataset if sample is None else f"{dataset}_{sample}"


def require(condition, message):
    if not condition:
        raise ValueError(message)




def correlation_results(method, matches, summary):
    x = summary["example_mean_instability"].to_numpy(float)
    y = summary["best_jaccard_mean"].to_numpy(float)
    result = dict(method=method, n_samples=summary["sample_id"].nunique(),
                  n_clusters=len(summary), n_matches=len(matches),
                  spearman_rho=np.nan, spearman_pvalue=np.nan,
                  pearson_r=np.nan, pearson_pvalue=np.nan, status="ok")
    if len(summary) < 3:
        result["status"] = "too_few_clusters"
    elif np.ptp(x) == 0 or np.ptp(y) == 0:
        result["status"] = "constant_values"
    else:
        sr, pr = spearmanr(x, y), pearsonr(x, y)
        result.update(spearman_rho=float(sr.statistic), spearman_pvalue=float(sr.pvalue),
                      pearson_r=float(pr.statistic), pearson_pvalue=float(pr.pvalue))
    return result


def bin_counts(method, matches, summary):
    return [dict(method=method, example_fine_bin=label,
                 n_clusters=int(summary["example_fine_bin"].eq(label).sum()),
                 n_matches=int(matches["example_fine_bin"].eq(label).sum()))
            for label in FINE_BINS]


def style_boxplot(boxplot, colors):
    for box, color in zip(boxplot["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.82)
        box.set_edgecolor("black")
        box.set_linewidth(1.1)

    for key in ["whiskers", "caps"]:
        for line in boxplot[key]:
            line.set_color("black")
            line.set_linewidth(1.1)

    for median in boxplot["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)


def plot_method(method, matches, cluster_summary, mi, ax_scatter, ax_box):
    populated_bins = [
        fine_bin
        for fine_bin in FINE_BINS
        if matches.loc[
            matches["example_fine_bin"].eq(fine_bin), "best_jaccard"
        ].notna().any()
    ]
    if not populated_bins:
        return


    box_data = [
        matches.loc[
            matches["example_fine_bin"].eq(fine_bin), "best_jaccard"
        ].dropna().to_numpy(float)
        for fine_bin in populated_bins
    ]
    positions = np.arange(1, len(populated_bins) + 1, dtype=float)
    box_positions = positions - 0.14
    violin_positions = positions + 0.10
    colors = [BIN_COLORS[b] for b in populated_bins]


    boxplot = ax_box.boxplot(
        box_data,
        positions=box_positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        manage_ticks=False,
    )
    style_boxplot(boxplot, colors)


    for values, color, violin_position in zip(
        box_data, colors, violin_positions
    ):


        if len(values) >= 2 and np.ptp(values) > 0:
            violin = ax_box.violinplot(
                [values],
                positions=[violin_position],
                widths=0.55,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method="scott",
            )
            body = violin["bodies"][0]
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = np.maximum(vertices[:, 0], violin_position)
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_linewidth(1.1)
            body.set_alpha(0.6)
            body.set_zorder(1)

    ax_box.tick_params(axis='x', labelsize=13.5)
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(populated_bins, rotation=45, ha="right")
    ax_box.set_xlim(0.45, len(populated_bins) + 0.65)
    ax_box.set_ylim(0, 1)


    ax_box.tick_params(axis='y', labelsize=13.5)
    ax_box.set_ylim(0, 1)
    ax_box.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax_box.set_yticklabels([])

    ax_box.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.4)
    ax_box.spines["top"].set_visible(False)
    ax_box.spines["right"].set_visible(False)


    Ns = []
    plotted_x_max = float(cluster_summary["example_mean_instability"].max())
    rng = np.random.default_rng(RANDOM_SEED)
    for fine_bin in populated_bins:
        sub = cluster_summary[
            cluster_summary["example_fine_bin"].eq(fine_bin)
        ]
        if sub.empty:
            continue
        jitter = rng.normal(0, 0.0025, size=len(sub))
        plotted_x_max = max(plotted_x_max, float(
            (sub["example_mean_instability"].to_numpy(float) + jitter).max()))

        Ns.append(str(len(sub["example_mean_instability"].to_numpy(float))))

        ax_scatter.scatter(
            sub["example_mean_instability"].to_numpy(float) + jitter,
            sub["best_jaccard_mean"],
            s=34,
            color=BIN_COLORS[fine_bin],
            edgecolor="black",
            linewidth=0.35,
            alpha=0.85,
        )


    fit_data = cluster_summary[
        ["example_mean_instability", "best_jaccard_mean"]
    ].dropna()
    if len(fit_data) >= 2 and fit_data["example_mean_instability"].nunique() >= 2:
        x_fit = fit_data["example_mean_instability"].to_numpy(float)
        y_fit = fit_data["best_jaccard_mean"].to_numpy(float)
        slope, intercept = np.polyfit(x_fit, y_fit, deg=1)
        x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
        y_line = slope * x_line + intercept
        ax_scatter.plot(
            x_line,
            y_line,
            color="black",
            linestyle="--",
            linewidth=1.75,

            zorder=4,
        )


    correlation = correlation_results(method, matches, cluster_summary)
    rho, spearman_p = correlation["spearman_rho"], correlation["spearman_pvalue"]
    pearson_r, pearson_p = correlation["pearson_r"], correlation["pearson_pvalue"]
    n_clusters = correlation["n_clusters"]

    correlation_text = (
        f"Spearman ρ = {rho:.3f}, p = {spearman_p:.3g}\n"
        f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.3g}\n"
        f"n = {n_clusters} clusters"
    )
    print(correlation_text)


    x_upper = max(0.1, float(np.ceil(plotted_x_max * 10) / 10))
    ax_scatter.set_xlim(0, x_upper)
    ax_scatter.set_xticks(np.arange(int(round(x_upper * 10)) + 1) / 10)

    ax_scatter.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])

    ax_scatter.tick_params(axis='x', labelsize=13.5)
    ax_scatter.tick_params(axis='y', labelsize=13.5)


    ax_scatter.set_xlabel("Cluster Instability - Example Run", fontsize=15)
    ax_scatter.set_ylabel("Jaccard w.r.t. Match", fontsize=15)
    ax_scatter.set_title(f"Cross-run Consistency of {METHOD_DISPLAY[method]}",
                         fontweight="bold",
                         fontsize=19, x=1.0)


    ax_scatter.grid(linestyle=":", linewidth=0.6, alpha=0.4)
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)


    text = "(" + r"$N=$" + ",".join(Ns) + ")"
    ax_scatter.text(x_upper / 2, -0.37, text, fontsize=16, ha="center", va="bottom", clip_on=False)
    return x_upper


def save_figure(fig):
    for extension in SAVE_FORMATS:
        path = OUTPUT_DIR / f'{FIGURE_STEM}.{extension}'
        fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight', facecolor='white')
        print(f'Saved {path}')


def main(argv=None):
    global OUTPUT_DIR, FIG_DPI, SAVE_FORMATS
    parser = argparse.ArgumentParser(description='Plot Supplementary Figure S19.')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--dpi', type=int, default=FIG_DPI)
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf'], default=SAVE_FORMATS)
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error('--dpi must be positive')
    OUTPUT_DIR, FIG_DPI, SAVE_FORMATS = args.output_dir, args.dpi, args.formats
    plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'pdf.compression': 9})
    matches = dgea_data.matches().copy()
    summary = dgea_data.cluster_summary().copy()
    for frame in (matches, summary):
        frame['example_fine_bin'] = frame['example_fine_bin'].replace({'[0.9,1.0]': '[0.9,inf)'})
    require(set(matches['method']) == set(METHODS), 'Expected all ten methods')
    method_data, correlation_rows, count_rows = {}, [], []
    expected_samples = {sample_id(*pair) for pair in DATASET_SAMPLES}
    for method in METHODS:
        mm = matches.loc[matches['method'].eq(method)].copy()
        ss = summary.loc[summary['method'].eq(method)].copy()
        require(set(mm['sample_id']) == set(ss['sample_id']) == expected_samples,
                f'{method}: expected all 18 samples')
        keys = ['sample_id', 'example_run_idx', 'example_cluster']
        grouped = mm.groupby(keys, sort=False)
        require(grouped.size().eq(49).all(), f'{method}: expected 49 matches per cluster')
        actual = grouped['best_jaccard'].mean()
        expected = ss.set_index(keys)['best_jaccard_mean'].reindex(actual.index)
        require(np.allclose(actual, expected, rtol=1e-12, atol=1e-14),
                f'{method}: cluster means disagree')
        method_data[method] = (mm, ss)
        correlation_rows.append(correlation_results(method, mm, ss))
        count_rows.extend(bin_counts(method, mm, ss))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(correlation_rows).to_csv(OUTPUT_DIR / 'correlations.csv', index=False)
    pd.DataFrame(count_rows).to_csv(OUTPUT_DIR / 'bin_counts.csv', index=False)
    fig, axes = plt.subplots(nrows=METHOD_GRID[0], ncols=2 * METHOD_GRID[1],
                             figsize=FIGSIZE, constrained_layout=True)
    try:
        for index, method in enumerate(METHODS):
            row, column = divmod(index, METHOD_GRID[1])
            plot_method(method, *method_data[method], index,
                        axes[row, 2 * column], axes[row, 2 * column + 1])
        fig.tight_layout(rect=(0, 0, 1, 1))
        save_figure(fig)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
