#!/usr/bin/env python3
"""Plot S18 BayesSpace entropy and instability from minimal saved inputs."""
import argparse
import hashlib
import os
from pathlib import Path

from common import DATA_DIR, OUTPUT_ROOT, read_spatial_sample, histology_kwargs
os.environ.setdefault('MPLCONFIGDIR', str(OUTPUT_ROOT / '.matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
OUT_DIR = OUTPUT_ROOT / 's18'
ENTROPY_DIR = DATA_DIR / 's18'

FIGSIZE = (8, 4)

DPI = 600

FIG_LEFT = 0.08

FIG_RIGHT = 0.985

FIG_TOP = 0.96

FIG_BOTTOM = 0.055

SPATIAL_WSPACE = 0.025

FIG_HSPACE = 0.12

DISTRIBUTION_GAP_RATIO = 0.34

DISTRIBUTION_WIDTH_RATIO = 0.78

DISTRIBUTION_HEIGHT_FRACTION = 0.74

TITLE_FONTSIZE = 13

AXIS_LABEL_FONTSIZE = 9.5

TICK_FONTSIZE = 8

ROW_LABEL_FONTSIZE = 11.5

SUMMARY_FONTSIZE = 7.5

DISTRIBUTION_CATEGORY_FONTSIZE = 8

LEGEND_BIN_FONTSIZE = 6.3

LEGEND_GROUP_FONTSIZE = 7.5

CROP_PADDING = 0.035

SPATIAL_IMAGE_ALPHA = 0.45

SPATIAL_FRAME_LINEWIDTH = 0.35

OTHER_SPOTS_COLOR = "#d9d9d9"

ENTROPY_CMAP = "magma"

INSTABILITY_AXIS_COLOR = "#e6550d"

N_RUNS = 50

CLUSTER_COLORS = [
    "#3049ad", "#fe8011", "#1b7837", "#fa0000", "#ab43fc",
    "#8d574c", "#ff00d9", "#bcbd22", "#17becf",
    "#8baaf3", "#ffbb79", "#99df8b", "#fe7775", "#c6b1d4",
    "#c49d95", "#ff80c6", "#dcdb91", "#a7d1e6",
    "#393b79", "#8c6d31", "#0aac00", "#982109", "#7b4173",
    "#713230", "#ff008c", "#637939", "#e7cb94", "#ccefc5",
    "#efcece", "#f7b6d2", "#eeedc8",
]

INSTABILITY_BINS = [
    ("[0.0,0.1)", 0.0, 0.1, "#08306b"),
    ("[0.1,0.2)", 0.1, 0.2, "#4292c6"),
    ("[0.2,0.3)", 0.2, 0.3, "#9ecae1"),
    ("[0.3,0.4)", 0.3, 0.4, "#deebf7"),
    ("[0.4,0.5)", 0.4, 0.5, "#fee391"),
    ("[0.5,0.6)", 0.5, 0.6, "#fec44f"),
    ("[0.6,0.7)", 0.6, 0.7, "#fdae6b"),
    ("[0.7,0.8)", 0.7, 0.8, "#fd8d3c"),
    ("[0.8,0.9)", 0.8, 0.9, "#fb6a4a"),
    ("[0.9,∞)", 0.9, np.inf, "#cb181d"),
]

INSTABILITY_PALETTE = {item[0]: item[3] for item in INSTABILITY_BINS}

VISIUM_HD_DATASETS = {
    "Visium_HD_Human_Colon_Cancer_cropped_square",
    "ov_ffpe",
    "coad_ffpe",
}

def dataset_tag(dataset, sample):
    return f"{dataset}_{sample}" if sample is not None else dataset

def category_order(series):
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(value) for value in series.cat.categories]
    return [str(value) for value in pd.unique(series.dropna().astype(str))]

def reference_palette(annotation):
    categories = category_order(annotation)
    if len(categories) > len(CLUSTER_COLORS):
        raise ValueError(
            f"Need {len(categories)} reference colors, but only "
            f"{len(CLUSTER_COLORS)} are available."
        )
    return dict(zip(categories, CLUSTER_COLORS[:len(categories)]))

def aligned_prediction_palette(annotation, prediction, gt_palette):
    """Globally align predicted clusters to reference labels by Jaccard overlap."""
    pred = pd.Series(prediction, index=annotation.index).astype(str)
    gt = annotation.astype("string")
    valid = gt.notna()
    pred_categories = list(pd.unique(pred))
    gt_categories = list(gt_palette)

    if not valid.any():
        return {
            category: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            for i, category in enumerate(pred_categories)
        }

    scores = np.zeros((len(pred_categories), len(gt_categories)), dtype=float)
    for i, pred_label in enumerate(pred_categories):
        pred_mask = pred.eq(pred_label) & valid
        for j, gt_label in enumerate(gt_categories):
            gt_mask = gt.eq(gt_label) & valid
            intersection = np.sum(pred_mask & gt_mask)
            union = np.sum(pred_mask | gt_mask)
            if union:
                scores[i, j] = intersection / union

    row_ind, col_ind = linear_sum_assignment(-scores)
    palette = {}
    assigned = set()
    for i, j in zip(row_ind, col_ind):
        if scores[i, j] <= 0:
            continue
        pred_label = pred_categories[i]
        palette[pred_label] = gt_palette[gt_categories[j]]
        assigned.add(pred_label)

    unused_colors = [
        color for color in CLUSTER_COLORS
        if color not in set(gt_palette.values())
    ]
    if not unused_colors and len(assigned) < len(pred_categories):
        raise ValueError("No unused cluster colors remain for unmatched predictions.")

    extra_i = 0
    for pred_label in pred_categories:
        if pred_label in assigned:
            continue
        palette[pred_label] = unused_colors[extra_i % len(unused_colors)]
        extra_i += 1

    return palette

def grayscale(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    rgb = image[..., :3].astype(float)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if np.issubdtype(image.dtype, np.integer):
        gray = np.clip(gray, 0, np.iinfo(image.dtype).max)
    else:
        gray = np.clip(gray, 0.0, 1.0)
    return gray.astype(image.dtype)

def nearest_spacing(coords):
    unique = np.unique(np.asarray(coords, dtype=float), axis=0)
    if len(unique) < 2:
        return 1.0
    distances = cKDTree(unique).query(unique, k=2)[0][:, 1]
    distances = distances[np.isfinite(distances) & (distances > 0)]
    return float(np.median(distances)) if distances.size else 1.0

def make_spot_vertices(coords, dataset):
    """Use the same spot geometry conventions as the main spatial figures."""
    spacing = nearest_spacing(coords)

    if dataset in VISIUM_HD_DATASETS:
        half = spacing / 2.0
        offsets = np.array([
            [-half, -half], [half, -half],
            [half, half], [-half, half],
        ])
    elif dataset == "mouse_brain":
        radius = spacing / np.sqrt(3)
        angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        offsets = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    elif dataset in {"dlpfc", "human_breast_cancer"}:
        radius = spacing / np.sqrt(3)
        angles = np.deg2rad([30, 90, 150, 210, 270, 330])
        offsets = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    else:

        radius = 0.48 * spacing
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        offsets = radius * np.column_stack((np.cos(angles), np.sin(angles)))

    return coords[:, None, :] + offsets[None, :, :]

def square_limits(coords):
    min_x, min_y = np.min(coords, axis=0)
    max_x, max_y = np.max(coords, axis=0)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    span = max(max_x - min_x, max_y - min_y, 1.0)
    span *= 1 + 2 * CROP_PADDING
    half = span / 2
    return center_x - half, center_x + half, center_y - half, center_y + half

def style_spatial_axis(ax, coords):
    x0, x1, y0, y1 = square_limits(coords)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(SPATIAL_FRAME_LINEWIDTH)

def draw_histology(ax, context):
    if context["image"] is not None:
        ax.imshow(
            context["image"],
            cmap="gray",
            alpha=SPATIAL_IMAGE_ALPHA,
            origin="upper",
            interpolation="nearest",
            zorder=0,
            **context['image_kwargs'],
        )

def plot_spatial_categories(ax, context, labels, title, palette):
    draw_histology(ax, context)
    labels = pd.Series(labels, dtype="string")
    colors = [
        palette.get(str(value), OTHER_SPOTS_COLOR)
        if not pd.isna(value) else OTHER_SPOTS_COLOR
        for value in labels
    ]
    ax.add_collection(PolyCollection(
        context["vertices"],
        facecolors=colors,
        edgecolors="none",
        linewidths=0,
        rasterized=True,
        zorder=2,
    ))
    style_spatial_axis(ax, context["coords"])
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=8)

def instability_labels(values):
    values = np.asarray(values, dtype=float)
    labels = np.full(values.shape, pd.NA, dtype=object)
    for label, low, high, _ in INSTABILITY_BINS:
        mask = np.isfinite(values) & (values >= low)
        mask &= values < high
        labels[mask] = label
    return pd.Series(labels, dtype="string")

def mean_sd_text(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return "NA"
    if len(values) == 1:
        return f"{values[0]:.3f} ± NA"
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"

def plot_spatial_instability(ax, context, values, title):
    values = np.asarray(values, dtype=float)
    plot_spatial_categories(
        ax,
        context,
        instability_labels(values),
        title,
        INSTABILITY_PALETTE,
    )
    ax.set_xlabel(
        f"mean ± s.d. = {mean_sd_text(values)}",
        fontsize=SUMMARY_FONTSIZE,
        color="0.25",
        labelpad=3,
    )

def plot_spatial_entropy(ax, context, entropy, title, vmax):
    entropy = np.asarray(entropy, dtype=float)
    norm = Normalize(vmin=0.0, vmax=max(float(vmax), np.finfo(float).eps))
    collection = PolyCollection(
        context["vertices"],
        edgecolors="none",
        linewidths=0,
        cmap=ENTROPY_CMAP,
        norm=norm,
        rasterized=True,
        zorder=2,
    )
    collection.set_array(np.ma.masked_invalid(entropy))
    ax.add_collection(collection)
    style_spatial_axis(ax, context["coords"])
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=8)
    ax.set_xlabel(
        f"mean ± s.d. = {mean_sd_text(entropy)}",
        fontsize=SUMMARY_FONTSIZE,
        color="0.25",
        labelpad=3,
    )
    return collection

def correlation_stats(x, y, x_name, y_name):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = len(x)

    if n >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        rho = spearmanr(x, y).statistic
        r = pearsonr(x, y).statistic
    else:
        rho = r = np.nan

    return {
        "x": x_name,
        "y": y_name,
        "n": int(n),
        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        "pearson_r": float(r) if np.isfinite(r) else np.nan,
    }

def add_instability_legend(ax):
    """Figure 2-style horizontal grouped instability legend."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    groups = [
        ("Stable", 0, 2),
        ("Modestly Stable", 2, 4),
        ("Unstable", 4, 6),
        ("Highly Unstable", 6, 10),
    ]

    for group_name, start, stop in groups:
        x0 = start + 0.03
        width = stop - start - 0.06
        ax.add_patch(Rectangle(
            (x0, 0.50), width, 0.28,
            fill=False, edgecolor="0.35", linewidth=0.7,
        ))

        for i in range(start, stop):
            item_x = i + 0.12
            ax.add_patch(Rectangle(
                (item_x, 0.57), 0.15, 0.14,
                facecolor=INSTABILITY_BINS[i][3], edgecolor="none",
            ))
            ax.text(
                item_x + 0.19, 0.64, INSTABILITY_BINS[i][0],
                ha="left", va="center", fontsize=LEGEND_BIN_FONTSIZE, fontweight="bold",
            )

        ax.text(
            (start + stop) / 2, 0.22, group_name,
            ha="center", va="center", fontsize=LEGEND_GROUP_FONTSIZE, fontweight="bold",
        )

def style_boxplot(boxplot, facecolor):
    """Use restrained boxplot styling so the distribution panel stays lightweight."""
    for patch in boxplot["boxes"]:
        patch.set_facecolor(facecolor)
        patch.set_alpha(0.72)
        patch.set_edgecolor("0.25")
        patch.set_linewidth(0.75)
    for artist in boxplot["whiskers"] + boxplot["caps"]:
        artist.set_color("0.25")
        artist.set_linewidth(0.75)
    for artist in boxplot["medians"]:
        artist.set_color("black")
        artist.set_linewidth(1.15)

def plot_distribution_boxplot(ax, bayes_entropy, bayes_instability, leiden_instability):
    """Compact vertical distributions with separate entropy/instability y-axes."""
    entropy = np.asarray(bayes_entropy, dtype=float)
    entropy = entropy[np.isfinite(entropy)]
    bayes_instability = np.asarray(bayes_instability, dtype=float)
    bayes_instability = bayes_instability[np.isfinite(bayes_instability)]
    leiden_instability = np.asarray(leiden_instability, dtype=float)
    leiden_instability = leiden_instability[np.isfinite(leiden_instability)]

    entropy_x = 1.0
    bayes_instability_x = 2.45
    leiden_instability_x = 3.25
    separator_x = 1.72


    style_boxplot(
        ax.boxplot(
            [entropy],
            positions=[entropy_x],
            widths=0.46,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        ),
        "#c7c7c7",
    )


    instability_ax = ax.twinx()
    style_boxplot(
        instability_ax.boxplot(
            [bayes_instability, leiden_instability],
            positions=[bayes_instability_x, leiden_instability_x],
            widths=0.46,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        ),
        "#fdae6b",
    )

    entropy_max = float(np.nanmax(entropy)) if entropy.size else 1.0
    ax.set_ylim(0, entropy_max * 1.05 if entropy_max > 0 else 1.0)
    maximum = max(float(bayes_instability.max(initial=0)), float(leiden_instability.max(initial=0)))
    instability_max = max(1.0, float(np.ceil(maximum * 1.05 * 10) / 10))
    instability_ax.set_ylim(0, instability_max)
    instability_ax.set_yticks(np.linspace(0, instability_max, 3))

    ax.set_xlim(0.45, 3.80)
    instability_ax.set_xlim(ax.get_xlim())
    ax.axvline(
        separator_x,
        color="0.55",
        linestyle=(0, (2, 2)),
        linewidth=0.7,
        alpha=0.8,
        zorder=0,
    )

    ax.set_xticks([entropy_x, bayes_instability_x, leiden_instability_x])
    category_labels = ax.set_xticklabels(
        ["BayesSpace", "BayesSpace", "Leiden"],
        fontsize=DISTRIBUTION_CATEGORY_FONTSIZE,
        fontweight="bold",
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    category_labels[0].set_color("0.30")
    for label in category_labels[1:]:
        label.set_color(INSTABILITY_AXIS_COLOR)
    ax.tick_params(axis="x", length=0, pad=2)
    instability_ax.tick_params(axis="x", bottom=False, labelbottom=False)


    ax.set_ylabel(
        "Entropy (nats)",
        fontsize=AXIS_LABEL_FONTSIZE,
        color="0.30",
        labelpad=3,
    )
    ax.tick_params(
        axis="y",
        labelsize=TICK_FONTSIZE,
        length=2.5,
        width=0.75,
        color="0.30",
        labelcolor="0.30",
        pad=2,
    )
    ax.spines["left"].set_color("0.30")
    ax.spines["left"].set_linewidth(0.8)


    instability_ax.set_ylabel(
        "Instability",
        fontsize=AXIS_LABEL_FONTSIZE,
        color=INSTABILITY_AXIS_COLOR,
        labelpad=3,
    )
    instability_ax.tick_params(
        axis="y",
        labelsize=TICK_FONTSIZE,
        length=2.5,
        width=0.75,
        color=INSTABILITY_AXIS_COLOR,
        labelcolor=INSTABILITY_AXIS_COLOR,
        pad=2,
    )
    instability_ax.spines["right"].set_color(INSTABILITY_AXIS_COLOR)
    instability_ax.spines["right"].set_linewidth(0.8)


    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("0.45")
    ax.spines["bottom"].set_linewidth(0.7)
    for side in ("top", "left", "bottom"):
        instability_ax.spines[side].set_visible(False)
    ax.set_title(
        "Distribution",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=6,
    )
    ax.grid(False)
    instability_ax.grid(False)
    return instability_ax

def draw_layout(fig):
    'Resolve text/axes positions without redrawing large data artists.'
    artists = [artist for ax in fig.axes for artist in [*ax.images, *ax.collections]]
    visibility = [artist.get_visible() for artist in artists]
    try:
        for artist in artists:
            artist.set_visible(False)
        fig.canvas.draw()
    finally:
        for artist, visible in zip(artists, visibility):
            artist.set_visible(visible)


def spatial_context(dataset, adata):
    """Apply the original image scale and spot geometry."""
    coords = np.asarray(adata.obsm['spatial'], dtype=float).copy()
    image = None
    if adata.uns.get('spatial'):
        library = next(iter(adata.uns['spatial'].values()))
        for key in ('hires', 'lowres'):
            if key in library.get('images', {}):
                scale = float(library.get('scalefactors', {}).get(f'tissue_{key}_scalef', 1.0))
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError('Invalid histology scale')
                coords *= scale
                image = grayscale(library['images'][key])
                break
    if coords.shape != (len(adata.obs), 2) or not np.isfinite(coords).all():
        raise ValueError('Invalid spatial coordinates')
    return dict(image=image, coords=coords, vertices=make_spot_vertices(coords, dataset),
                image_kwargs=histology_kwargs(adata))


def load_entropy(tag, adata):
    """Natural-log entropy from the selected BayesSpace run, in nats."""
    with np.load(ENTROPY_DIR / f'{tag}.npz', allow_pickle=False) as saved:
        entropy = saved['entropy_nats']
        spot_digest = hashlib.sha256('\n'.join(adata.obs_names).encode()).hexdigest()
        if str(saved['spot_order_sha256']) != spot_digest:
            raise ValueError(f'{tag}: entropy spot order differs from spatial data')
        if int(saved['example_index']) != int(adata.uns['example_indices']['bayesspace']):
            raise ValueError(f'{tag}: entropy and example labels use different runs')
    if entropy.shape != (adata.n_obs,) or not np.isfinite(entropy).all() or (entropy < 0).any():
        raise ValueError(f'{tag}: invalid entropy')
    return entropy


def save_figure(fig, tag):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [OUT_DIR / f'{tag}_instability_entropy_bayesspace.{ext}' for ext in ('png', 'pdf')]
    try:
        for path in paths:
            fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.03, facecolor='white')
            print(f'[saved] {path}', flush=True)
    finally:
        plt.close(fig)
    return paths


def make_panel_for_dataset_sample(dataset, sample):
    tag = dataset_tag(dataset, sample)
    print(f"[START] {tag}", flush=True)

    adata = read_spatial_sample(tag)
    obs = adata.obs
    if 'annotation' not in obs:
        obs['annotation'] = pd.Series(pd.NA, index=obs.index, dtype='string')
    bayes_idx = int(adata.uns['example_indices']['bayesspace'])
    leiden_idx = int(adata.uns['example_indices']['leiden'])
    bayes_labels = np.asarray(adata.uns['example_labels']['bayesspace']).astype(str)
    leiden_labels = np.asarray(adata.uns['example_labels']['leiden']).astype(str)
    bayes_entropy = load_entropy(tag, adata)
    bayes_instability = obs['bayesspace_uncertainty'].to_numpy(dtype=float)
    leiden_instability = obs['leiden_uncertainty'].to_numpy(dtype=float)
    context = spatial_context(dataset, adata)
    gt = obs["annotation"]
    gt_palette = reference_palette(gt)
    bayes_palette = aligned_prediction_palette(gt, bayes_labels, gt_palette)
    leiden_palette = aligned_prediction_palette(gt, leiden_labels, gt_palette)


    reference_valid = gt.notna().to_numpy()
    reference_labels = gt[reference_valid].astype(str).to_numpy()
    bayes_ari = (adjusted_rand_score(reference_labels, bayes_labels[reference_valid])
                 if reference_valid.sum() >= 2 else np.nan)
    leiden_ari = (adjusted_rand_score(reference_labels, leiden_labels[reference_valid])
                  if reference_valid.sum() >= 2 else np.nan)

    corr_stats = [
        correlation_stats(
            bayes_entropy, leiden_instability,
            "BayesSpace entropy", "Leiden instability",
        ),
        correlation_stats(
            bayes_entropy, bayes_instability,
            "BayesSpace entropy", "BayesSpace instability",
        ),
    ]
    print(f"[CORR] {tag}", flush=True)
    for stat in corr_stats:
        print(
            f"  {stat['x']} vs {stat['y']}: "
            f"Spearman rho={stat['spearman_rho']:.4f}, "
            f"Pearson r={stat['pearson_r']:.4f}, n={stat['n']}",
            flush=True,
        )

    entropy_vmax = float(np.max(bayes_entropy))

    fig = plt.figure(figsize=FIGSIZE)


    outer_grid = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[2.0, 0.25],
        width_ratios=[3.0, DISTRIBUTION_GAP_RATIO, DISTRIBUTION_WIDTH_RATIO],
        left=FIG_LEFT,
        right=FIG_RIGHT,
        top=FIG_TOP,
        bottom=FIG_BOTTOM,
        hspace=FIG_HSPACE,
        wspace=0.0,
    )
    spatial_grid = outer_grid[0, 0].subgridspec(
        nrows=2,
        ncols=3,
        hspace=FIG_HSPACE,
        wspace=SPATIAL_WSPACE,
    )

    ax00 = fig.add_subplot(spatial_grid[0, 0])
    ax01 = fig.add_subplot(spatial_grid[0, 1])
    ax02 = fig.add_subplot(spatial_grid[0, 2])
    ax10 = fig.add_subplot(spatial_grid[1, 0])
    ax11 = fig.add_subplot(spatial_grid[1, 1])
    ax12 = fig.add_subplot(spatial_grid[1, 2])
    distribution_ax = fig.add_subplot(outer_grid[0, 2])
    legend_ax = fig.add_subplot(outer_grid[1, :])


    plot_spatial_categories(ax00, context, bayes_labels, "Example Run", bayes_palette)
    plot_spatial_instability(ax01, context, bayes_instability, "Instability Map")
    entropy_artist = plot_spatial_entropy(
        ax02,
        context,
        bayes_entropy,
        f"Entropy Ex. Run",
        entropy_vmax,
    )

    ax00.set_ylabel(
        "BayesSpace", fontsize=ROW_LABEL_FONTSIZE,
        fontweight="bold", labelpad=6,
    )
    ax00.set_xlabel(
        f"ARI = {bayes_ari:.3f}",
        fontsize=SUMMARY_FONTSIZE,
        color="0.25",
        labelpad=3,
    )


    plot_spatial_categories(ax10, context, leiden_labels, "", leiden_palette)
    plot_spatial_instability(ax11, context, leiden_instability, "")
    plot_spatial_categories(ax12, context, gt, "", gt_palette)

    ax10.set_ylabel(
        "Leiden", fontsize=ROW_LABEL_FONTSIZE,
        fontweight="bold", labelpad=6,
    )
    ax10.set_xlabel(
        f"ARI = {leiden_ari:.3f}",
        fontsize=SUMMARY_FONTSIZE,
        color="0.25",
        labelpad=3,
    )
    ax12.set_xlabel(
        "Reference",
        fontsize=ROW_LABEL_FONTSIZE,
        fontweight="bold",
        labelpad=6,
    )


    cbar_ax = ax02.inset_axes([1.025, 0.14, 0.025, 0.72])
    colorbar = fig.colorbar(entropy_artist, cax=cbar_ax)
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE, length=2, pad=2)
    colorbar.outline.set_linewidth(0.6)

    distribution_twin_ax = plot_distribution_boxplot(
        distribution_ax, bayes_entropy, bayes_instability, leiden_instability
    )


    draw_layout(fig)
    distribution_pos = distribution_ax.get_position()
    compact_height = distribution_pos.height * DISTRIBUTION_HEIGHT_FRACTION
    compact_bottom = distribution_pos.y0 + (distribution_pos.height - compact_height) / 2
    compact_distribution_pos = [
        distribution_pos.x0,
        compact_bottom,
        distribution_pos.width,
        compact_height,
    ]
    distribution_ax.set_position(compact_distribution_pos)
    distribution_twin_ax.set_position(compact_distribution_pos)

    add_instability_legend(legend_ax)

    png_path, pdf_path = save_figure(fig, tag)

    row = {
        "dataset": dataset,
        "sample": sample,
        "dataset_tag": tag,
        "bayesspace_example_run_idx": bayes_idx,
        "bayesspace_example_run_n_clusters": int(len(np.unique(bayes_labels))),
        "leiden_example_run_idx": leiden_idx,
        "leiden_example_run_n_clusters": int(len(np.unique(leiden_labels))),
        "bayesspace_example_run_ari": float(bayes_ari),
        "leiden_example_run_ari": float(leiden_ari),
        "n_spots": len(obs),
        "n_annotated_spots": int(reference_valid.sum()),
        "bayesspace_example_seed": 42 + bayes_idx,
        "leiden_example_seed": 42 + leiden_idx,
        "instability_runs": N_RUNS,
        "entropy_units": "nats",
        "bayesspace_entropy_mean": float(bayes_entropy.mean()),
        "bayesspace_entropy_sd": float(bayes_entropy.std(ddof=1)),
        "bayesspace_instability_mean": float(bayes_instability.mean()),
        "leiden_instability_mean": float(leiden_instability.mean()),
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
        "status": "ok",
        "error": "",
    }
    for stat in corr_stats:
        key = (
            stat["x"].lower().replace(" ", "_")
            + "_vs_"
            + stat["y"].lower().replace(" ", "_")
        )
        row[f"{key}_spearman_rho"] = stat["spearman_rho"]
        row[f"{key}_pearson_r"] = stat["pearson_r"]
        row[f"{key}_n"] = stat["n"]

    print(f"[DONE] {tag}", flush=True)
    return row


def main():
    global OUT_DIR, ENTROPY_DIR, DPI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', nargs='+', help='Sample tags; default: all 18 samples.')
    parser.add_argument('--output-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--data-dir', type=Path, default=ENTROPY_DIR, help='Directory of entropy arrays.')
    parser.add_argument('--dpi', type=int, default=DPI)
    args = parser.parse_args()
    OUT_DIR, ENTROPY_DIR, DPI = args.output_dir, args.data_dir, args.dpi
    samples = pd.read_csv(ENTROPY_DIR / 'samples.csv', dtype={'sample': str}).fillna('')
    if args.samples:
        unknown = set(args.samples) - set(samples.dataset_tag)
        if unknown:
            parser.error(f'Unknown sample tags: {sorted(unknown)}')
        samples = samples[samples.dataset_tag.isin(args.samples)]
    rows = [make_panel_for_dataset_sample(row.dataset, row.sample or None)
            for row in samples.itertuples()]
    pd.DataFrame(rows).to_csv(OUT_DIR / 's18_summary.csv', index=False)


if __name__ == '__main__':
    main()
