#!/usr/bin/env python3
"""Plot S27 convergence from saved per-sample mean absolute differences."""
from pathlib import Path
import argparse
import os

import numpy as np
import pandas as pd

from common import DATA_DIR as SHARED_DATA_DIR, OUTPUT_ROOT
DATA_DIR = SHARED_DATA_DIR / 's27'
OUTPUT_DIR = OUTPUT_ROOT / 's27'
DPI = 600

RUN_COUNTS = list(range(5, 51, 5))

METHODS = [
    "leiden", "louvain", "bayesspace", "graphst", "graphst_mclust",
    "stagate", "stagate_mclust", "spicemix", "sedr", "sedr_mclust",
]

METHOD_NAMES = {
    "leiden": "Leiden", "louvain": "Louvain", "bayesspace": "BayesSpace",
    "graphst": "GraphST", "graphst_mclust": "GraphST (mclust)",
    "stagate": "STAGATE", "stagate_mclust": "STAGATE (mclust)",
    "spicemix": "SpiceMix", "sedr": "SEDR", "sedr_mclust": "SEDR (mclust)",
}

SAMPLES = [("dlpfc", sample) for sample in [
    "151507", "151508", "151509", "151510", "151669", "151670",
    "151671", "151672", "151673", "151674", "151675", "151676",
]] + [
    ("mouse_brain", None), ("mouse_brain_cerebellum", None),
    ("human_breast_cancer", None),
    ("Visium_HD_Human_Colon_Cancer_cropped_square", None),
    ("ov_ffpe", None), ("coad_ffpe", None),
]

def sample_tag(dataset, sample):
    return f"{dataset}_{sample}" if sample else dataset


def calculate_convergence(mode):
    """Weight samples equally; SEM uses sample SD with ddof=1."""
    frame = pd.read_csv(DATA_DIR / f's27_convergence_{mode}_sample_values.csv')
    samples = [sample_tag(*sample) for sample in SAMPLES]
    keys = ['sample', 'method', 'iterations']
    expected = pd.MultiIndex.from_product([samples, METHODS, RUN_COUNTS], names=keys)
    actual = pd.MultiIndex.from_frame(frame[keys])
    if (actual.has_duplicates or len(frame) != len(expected)
            or len(expected.difference(actual)) or set(frame['mode']) != {mode}):
        raise ValueError(f'Incomplete or duplicate {mode} convergence data')
    if not np.isfinite(frame.mean_absolute_difference).all() or (frame.mean_absolute_difference < 0).any():
        raise ValueError('Invalid mean absolute differences')
    data, method_rows = {}, []
    for method in METHODS:
        values = (frame[frame.method.eq(method)]
                  .pivot(index='sample', columns='iterations', values='mean_absolute_difference')
                  .loc[samples, RUN_COUNTS].to_numpy())
        if not np.allclose(values[:, -1], 0, rtol=0, atol=1e-7):
            raise ValueError(f'{method}: 50-run differences must be zero')
        data[method] = values
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
        method_rows.extend(dict(mode=mode, method=method, iterations=runs,
                                n_samples=len(values), mean=float(avg), sem=float(se))
                           for runs, avg, se in zip(RUN_COUNTS, mean, sem))
    return data, method_rows

def plot_convergence(data, mode):

    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sampled = mode == "sampled"
    fig, axes = plt.subplots(2, 5, figsize=(10 * 5 / 4, 5), sharex=True, sharey=True)
    for ax, method in zip(axes.flat, METHODS):
        values = np.asarray(data[method], dtype=float)


        for y in values:
            ax.plot(RUN_COUNTS, y, color="gray", alpha=0.25, linewidth=0.7)


        mean = values.mean(axis=0)
        if len(values) > 1:
            sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))
        else:
            sem = np.zeros_like(mean)
        ax.fill_between(RUN_COUNTS, mean - sem, mean + sem, color="#2166ac", alpha=0.20)
        ax.plot(RUN_COUNTS, mean, color="#2166ac", linewidth=2, marker="o", markersize=3)
        ax.set_title(METHOD_NAMES[method], fontweight="bold")
        ax.set_xticks(range(10, 51, 10))
        ylim = -0.001 if sampled else -0.005
        ax.set_ylim(ylim, None)
        ax.grid(alpha=0.2, linewidth=0.5)

    for ax in axes[-1]:
        ax.set_xlabel("Number of runs")
    fig.supylabel("Mean absolute spotwise difference relative to 50 runs")
    fig.tight_layout()

    suffix = "_sampled" if sampled else ""
    stem = f"s27_continuous_score_convergence{suffix}"
    try:

        for extension in ("png", "pdf"):
            path = OUTPUT_DIR / f"{stem}.{extension}"
            temporary = path.with_name(f"{stem}.tmp.{extension}")
            fig.savefig(temporary, dpi=DPI, bbox_inches="tight")
            temporary.replace(path)
            print(f"[saved] {path}", flush=True)
    finally:
        plt.close(fig)

CAPTION_RUN_COUNTS = [5, 10, 20, 30, 40, 45]

def format_values(mode, method_rows):
    """Preserve the old means/SEMs and equal-weight across-method summaries."""
    lookup = {(row["method"], row["iterations"]): row for row in method_rows}
    lines = ["", "=" * 80, "SAMPLED" if mode == "sampled" else "NONSAMPLED", "=" * 80]
    lines.append(f"{'method':14s}" + "".join(f"{k:>15d}" for k in RUN_COUNTS))
    for method in METHODS:
        cells = []
        for runs in RUN_COUNTS:
            row = lookup[method, runs]
            cells.append(f"{row['mean']:.4f}±{row['sem']:.4f}".rjust(15))
        lines.append(f"{method:14s}" + "".join(cells))

    summaries = []
    for runs in RUN_COUNTS:
        values = np.asarray([lookup[method, runs]["mean"] for method in METHODS], dtype=float)
        summaries.append(dict(
            mode=mode, iterations=runs, n_methods=len(values),
            mean_across_methods=float(values.mean()),
            min_method_mean=float(values.min()), max_method_mean=float(values.max()),
            caption_selected=runs in CAPTION_RUN_COUNTS,
        ))
    lines.extend(["-" * 164, f"{'Across methods':14s}" + "".join(
        f"{row['mean_across_methods']:15.4f}" for row in summaries)])
    lines.append("\nCaption-ready values:")
    for row in summaries:
        if row["caption_selected"]:
            lines.append(
                f"K={row['iterations']:2d}: "
                f"mean across methods = {row['mean_across_methods']:.4f}, "
                f"range across method means = {row['min_method_mean']:.4f}-{row['max_method_mean']:.4f}"
            )
    return "\n".join(lines), summaries


def main():
    global DATA_DIR, OUTPUT_DIR, DPI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--mode', choices=['sampled', 'nested', 'both'], default='sampled')
    parser.add_argument('--dpi', type=int, default=DPI)
    parser.add_argument('--caption-only', action='store_true', help='Print caption values without plotting.')
    args = parser.parse_args()
    DATA_DIR, OUTPUT_DIR, DPI = args.data_dir, args.output_dir, args.dpi
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modes = ('nested', 'sampled') if args.mode == 'both' else (args.mode,)
    for mode in modes:
        data, method_rows = calculate_convergence(mode)
        text, summaries = format_values(mode, method_rows)
        (OUTPUT_DIR / f's27_caption_{mode}.txt').write_text(text + '\n')
        pd.DataFrame(method_rows).to_csv(OUTPUT_DIR / f's27_method_summary_{mode}.csv', index=False)
        pd.DataFrame(summaries).to_csv(OUTPUT_DIR / f's27_across_methods_{mode}.csv', index=False)
        if args.caption_only:
            print(text)
        else:
            plot_convergence(data, mode)


if __name__ == '__main__':
    main()
