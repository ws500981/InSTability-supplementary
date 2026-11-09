#!/usr/bin/env python3
'Figure 2 and S1–S17: annotations, example clusters, and instability maps.'

from common import DATA_DIR, OUTPUT_ROOT, read_spatial_sample, histology_kwargs

import argparse
import gc
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
DATA_PATH = DATA_DIR / 'spatial'
PLOT_ROOT = OUTPUT_ROOT
OUT_DIR = PLOT_ROOT / 'fig2_s1_s17'
METHODS = [
    ('leiden', 'Leiden'), ('louvain', 'Louvain'), ('bayesspace', 'BayesSpace'),
    ('graphst', 'GraphST'), ('graphst_mclust', 'GraphST\n(mclust)'),
    ('sedr', 'SEDR'), ('sedr_mclust', 'SEDR\n(mclust)'),
    ('stagate', 'STAGATE'), ('stagate_mclust', 'STAGATE\n(mclust)'),
    ('spicemix', 'SpiceMix'),
]
FULL_DATASETS = {'Visium_HD_Human_Colon_Cancer', 'coad_ffpe_full', 'ov_ffpe_full'}
DATASETS = ['dlpfc', 'mouse_brain', 'mouse_brain_cerebellum', 'human_breast_cancer', 'Visium_HD_Human_Colon_Cancer_cropped_square', 'ov_ffpe', 'coad_ffpe'] # , 'Visium_HD_Human_Colon_Cancer', 'coad_ffpe_full', 'ov_ffpe_full']
DLPFC_SAMPLES = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
DATASET_DISPLAY = {'dlpfc': 'DLPFC', 'mouse_brain': 'Mouse Brain', 'mouse_brain_cerebellum': 'Mouse Brain Cerebellum', 'human_breast_cancer': 'Human Breast Cancer', 'Visium_HD_Human_Colon_Cancer_cropped_square': 'CRC', 'Visium_HD_Human_Colon_Cancer': 'CRC (full sample)', 'coad_ffpe': 'COAD', 'ov_ffpe': 'OV'} # , 'coad_ffpe_full': 'COAD (full sample)', 'ov_ffpe_full': 'OV (full sample)'}
CLUSTER_COLORS = ['#3049ad', '#fe8011', '#1b7837', '#fa0000', '#ab43fc', '#8d574c', '#ff00d9', '#bcbd22', '#17becf', '#8baaf3', '#ffbb79', '#99df8b', '#fe7775', '#c6b1d4', '#c49d95', '#ff80c6', '#dcdb91', '#a7d1e6', '#393b79', '#8c6d31', '#0aac00', '#982109', '#7b4173', '#713230', '#ff008c', '#637939', '#e7cb94', '#ccefc5', '#efcece', '#f7b6d2', '#eeedc8']

BINS = [('[0.0,0.1)', 0.0, 0.1, '#08306b'), ('[0.1,0.2)', 0.1, 0.2, '#4292c6'), ('[0.2,0.3)', 0.2, 0.3, '#9ecae1'), ('[0.3,0.4)', 0.3, 0.4, '#deebf7'), ('[0.4,0.5)', 0.4, 0.5, '#fee391'), ('[0.5,0.6)', 0.5, 0.6, '#fec44f'), ('[0.6,0.7)', 0.6, 0.7, '#fdae6b'), ('[0.7,0.8)', 0.7, 0.8, '#fd8d3c'), ('[0.8,0.9)', 0.8, 0.9, '#fb6a4a'), ('[0.9,1.0]', 0.9, 1.0, '#cb181d')]
BIN_COLORS = [x[3] for x in BINS]
HD_DATASETS = {'Visium_HD_Human_Colon_Cancer_cropped_square', 'Visium_HD_Human_Colon_Cancer', 'ov_ffpe', 'coad_ffpe', 'ov_ffpe_full', 'coad_ffpe_full'}
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 9, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'pdf.fonttype': 42, 'ps.fonttype': 42})

def pretty_dataset(ds):
    return DATASET_DISPLAY.get(ds, ds.replace('_', ' ').title())

def make_runs():
    runs = []
    for ds in DATASETS:
        if ds == 'dlpfc':
            runs.extend(((ds, sample) for sample in DLPFC_SAMPLES))
        else:
            runs.append((ds, None))
    return runs

def bin_indices(values):
    'Return 0..9 instability-bin indices.'
    values = np.asarray(values, dtype=float)
    idx = np.digitize(values, bins=np.arange(0.1, 1.0, 0.1), right=False)
    idx = np.clip(idx, 0, len(BINS) - 1)
    idx[~np.isfinite(values)] = -1
    return idx

def bin_counts(values):
    idx = bin_indices(values)
    return np.array([(idx == i).sum() for i in range(len(BINS))], dtype=int)

def category_order(series):
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(x) for x in series.cat.categories]
    return [str(x) for x in pd.unique(series.dropna().astype(str))]

def reference_palette(annotation):
    cats = category_order(annotation)
    return {cat: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cat in enumerate(cats)}

def aligned_prediction_palette(annotation, prediction, gt_palette):
    'Globally match predicted clusters to reference labels using'
    pred = pd.Series(prediction, index=annotation.index).astype(str)
    gt = annotation.astype('string')
    valid = gt.notna()
    pred_cats = list(pd.unique(pred))
    gt_cats = list(gt_palette.keys())
    palette = {}
    if valid.sum() == 0:
        return {cat: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cat in enumerate(pred_cats)}

    scores = np.zeros((len(pred_cats), len(gt_cats)))
    for i, p in enumerate(pred_cats):
        p_mask = (pred == p) & valid
        for j, g in enumerate(gt_cats):
            g_mask = (gt == g) & valid
            intersection = np.sum(p_mask & g_mask)
            union = np.sum(p_mask | g_mask)
            if union > 0:
                scores[i, j] = intersection / union

    row_ind, col_ind = linear_sum_assignment(-scores)
    assigned_pred = set()
    for i, j in zip(row_ind, col_ind):
        if scores[i, j] <= 0:
            continue
        pred_label = pred_cats[i]
        gt_label = gt_cats[j]
        color = gt_palette[gt_label]
        palette[pred_label] = color
        assigned_pred.add(pred_label)

    reference_colors = set(gt_palette.values())
    unused_colors = [c for c in CLUSTER_COLORS if c not in reference_colors] or CLUSTER_COLORS
    extra_i = 0
    for pred_label in pred_cats:
        if pred_label in assigned_pred:
            continue
        palette[pred_label] = unused_colors[extra_i % len(unused_colors)]
        extra_i += 1
    return palette

def get_spatial_context(adata, dataset):
    coords = np.asarray(adata.obsm['spatial'], dtype=float)
    image = None
    spatial = adata.uns.get('spatial', {})
    if isinstance(spatial, dict) and spatial:
        library = spatial[next(iter(spatial))]
        images = library.get('images', {})
        scales = library.get('scalefactors', {})
        for key in ('hires', 'lowres'):
            if key in images:
                image = np.asarray(images[key])
                scale = float(scales.get(f'tissue_{key}_scalef', 1.0))
                coords = coords * scale
                break
    unique = np.unique(coords, axis=0)
    if len(unique) > 1:
        distances = cKDTree(unique).query(unique, k=2)[0][:, 1]
        distances = distances[np.isfinite(distances) & (distances > 0)]
        spacing = float(np.median(distances))
    else:
        spacing = 1.0

    if dataset in HD_DATASETS:
        half = spacing / 2
        offsets = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    elif dataset == 'mouse_brain':
        radius = spacing / np.sqrt(3)
        angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        offsets = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    elif dataset in {'dlpfc', 'human_breast_cancer'}:
        radius = spacing / np.sqrt(3)
        angles = np.deg2rad([30, 90, 150, 210, 270, 330])
        offsets = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        radius = 0.48 * spacing
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        offsets = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    vertices = coords[:, None, :] + offsets[None, :, :]
    return {'coords': coords, 'image': image, 'vertices': vertices, 'spacing': spacing, 'image_kwargs': histology_kwargs(adata)}

def grayscale(image):
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    rgb = image[..., :3].astype(float)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

def draw_spatial(ax, ctx, labels, palette):
    image = ctx['image']
    coords = ctx['coords']
    if image is not None:
        ax.imshow(grayscale(image), cmap='gray', alpha=0.5, origin='upper', **ctx['image_kwargs'])
    labels = pd.Series(labels).astype('string')
    colors = [palette.get(str(x), '#d9d9d9') if not pd.isna(x) else '#d9d9d9' for x in labels]
    collection = PolyCollection(ctx['vertices'], facecolors=colors, edgecolors='none', linewidths=0, rasterized=True)
    ax.add_collection(collection)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    px = 0.035 * max(x1 - x0, 1)
    py = 0.035 * max(y1 - y0, 1)
    ax.set_xlim(x0 - px, x1 + px)
    ax.set_ylim(y1 + py, y0 - py)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color('black')

def blank_panel(ax, text=''):
    ax.set_xticks([])
    ax.set_yticks([])
    if text:
        ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center', fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)

def draw_instability_legend(fig, map_axes):

    fig.canvas.draw()
    positions = [ax.get_position() for ax in map_axes]
    left, right = positions[0].x0, positions[-1].x1
    top = min(pos.y0 for pos in positions) - 0.08 / fig.get_figheight()
    groups = [('Stable', 0, 2), ('Modestly Stable', 2, 4), ('Unstable', 4, 6), ('Highly Unstable', 6, 10)]

    stacked = (right - left) * fig.get_figwidth() < 6.5
    rows = len(groups) if stacked else 1
    row_height = 0.48 / fig.get_figheight()
    ax = fig.add_axes([left, top - (rows - 0.16) * row_height,
                       right - left, rows * row_height])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, rows)
    ax.axis('off')

    x_per_inch = 10 / ((right - left) * fig.get_figwidth())
    tile_width, tile_height = 0.16 * x_per_inch, 0.08 / 0.48
    for group_i, (group_name, start, stop) in enumerate(groups):
        y = rows - 1 - group_i if stacked else 0
        x0 = 0.03 if stacked else start + 0.03
        width = 9.94 if stacked else stop - start - 0.06
        ax.add_patch(Rectangle((x0, y + 0.44), width, 0.40, fill=False, edgecolor='0.35', linewidth=0.7))
        for i in range(start, stop):
            item_x = 10 * (i - start) / (stop - start) + 0.12 if stacked else i + 0.12
            ax.add_patch(Rectangle((item_x, y + 0.64 - tile_height / 2), tile_width, tile_height, facecolor=BINS[i][3], edgecolor='none'))
            ax.text(item_x + tile_width + 0.04 * x_per_inch, y + 0.64, BINS[i][0], ha='left', va='center', fontsize=8, fontweight='bold')
        center = 5 if stacked else (start + stop) / 2
        ax.text(center, y + 0.20, group_name, ha='center', va='center', fontsize=10.5, fontweight='bold')

def plot_one(dataset, sample=None, dpi=300):
    suffix = f'_{sample}' if sample else ''
    infile = DATA_PATH / f'{dataset}{suffix}.h5ad'
    print(f'Reading {infile}', flush=True)
    adata = read_spatial_sample(f'{dataset}{suffix}')
    provenance = adata.uns.get('instability_provenance', {})
    if provenance.get('mode') != 'nested' or provenance.get('iterations') != 50:
        raise ValueError(f'{infile}: expected packaged nested 50-run instability')
    if not adata.n_obs or not adata.obs_names.is_unique:
        raise ValueError(f'{infile}: missing or duplicated spot IDs')
    ctx = get_spatial_context(adata, dataset)
    has_reference = 'annotation' in adata.obs and adata.obs['annotation'].notna().any()
    if has_reference:
        gt_palette = reference_palette(adata.obs['annotation'])
    else:
        gt_palette = {}

    plot_methods = [(m, label) for m, label in METHODS
                    if dataset not in FULL_DATASETS or m in ('leiden', 'louvain')]
    for method, _ in plot_methods:
        if f'{method}_uncertainty' not in adata.obs or method not in adata.uns['example_labels']:
            raise ValueError(f'{infile}: missing {method} results or instability')
    valid = (adata.obs['annotation'].notna().to_numpy() if has_reference
             else np.zeros(adata.n_obs, dtype=bool))
    reference = adata.obs.loc[valid, 'annotation'].astype(str).to_numpy() if has_reference else None
    caption_rows = []
    n_methods = len(plot_methods)
    spot_counts = np.zeros((n_methods, len(BINS)), dtype=int)
    cluster_counts = np.zeros((n_methods, len(BINS)), dtype=int)
    fig_width = max(9.0, 5.0 + 1.25 * n_methods)
    fig = plt.figure(figsize=(fig_width, 4.6))
    ncols = n_methods + 4
    gs = fig.add_gridspec(2, ncols, width_ratios=[1.0, 0.3, *[1.0] * n_methods, 0.4, 1.18], hspace=0.025, wspace=0.08)
    fig.subplots_adjust(left=0.025, right=0.995, top=0.90, bottom=0.28)
    ax_ref = fig.add_subplot(gs[0, 0])
    if has_reference:
        draw_spatial(ax_ref, ctx, adata.obs['annotation'], gt_palette)
    else:
        blank_panel(ax_ref, 'No reference')
    ax_ref.set_title('Reference', fontsize=11, pad=4, fontweight='bold')
    ax_ref.text(-0.19, 1.08, 'A', transform=ax_ref.transAxes, fontsize=13, fontweight='bold', ha='left', va='top', clip_on=False)
    ax_unused = fig.add_subplot(gs[1, 0])
    ax_unused.axis('off')
    top_axes = []
    bottom_axes = []
    for method_i, (method, display) in enumerate(plot_methods):
        col = method_i + 2
        ax_top = fig.add_subplot(gs[0, col])
        ax_bottom = fig.add_subplot(gs[1, col])
        top_axes.append(ax_top)
        bottom_axes.append(ax_bottom)
        ax_top.set_title(display, fontsize=10, pad=4, fontweight='bold')
        uncertainty_key = f'{method}_uncertainty'
        results_key = f'{method}_results'
        u = adata.obs[uncertainty_key].to_numpy(dtype=float)
        if not np.isfinite(u).all():
            raise ValueError(f'{infile}: nonfinite {method} instability')
        u_bin = bin_indices(u)
        instability_labels = pd.Series([BINS[i][0] if i >= 0 else pd.NA for i in u_bin], index=adata.obs_names, dtype='string')
        instability_palette = {name: color for name, _, _, color in BINS}
        draw_spatial(ax_bottom, ctx, instability_labels, instability_palette)
        spot_counts[method_i] = bin_counts(u)
        example_i = int(adata.uns['example_indices'][method])
        example_run = np.asarray(adata.uns['example_labels'][method]).astype(str)
        if example_run.shape != (adata.n_obs,):
            raise ValueError(f'{infile}: {method} labels do not match spot count')
        if has_reference:
            pred_palette = aligned_prediction_palette(adata.obs['annotation'], example_run, gt_palette)
        else:
            cats = list(pd.unique(example_run))
            pred_palette = {cat: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cat in enumerate(cats)}
        draw_spatial(ax_top, ctx, example_run, pred_palette)
        cluster_means = []
        for cluster in np.unique(example_run):
            mask = example_run == cluster
            if mask.any():
                cluster_means.append(np.nanmean(u[mask]))
        cluster_counts[method_i] = bin_counts(np.asarray(cluster_means))
        ari = adjusted_rand_score(reference, example_run[valid]) if has_reference else np.nan
        mean_instability = float(u.mean())
        caption_rows.append(dict(
            sample=f'{dataset}{suffix}', dataset=dataset, method=method,
            example_run=example_i + 1, seed=int(provenance['seeds'][example_i]),
            n_clusters=len(cluster_means), n_spots=adata.n_obs,
            n_annotated_spots=int(valid.sum()), ari=ari, mean_instability=mean_instability,
        ))
        print(f'  {method}: example run {example_i + 1}, ARI={ari:.4f}, '
              f'mean instability={mean_instability:.4f}', flush=True)
    top_axes[0].set_ylabel('Example Run', fontsize=11, fontweight='bold', labelpad=3)
    bottom_axes[0].set_ylabel('Instability Map', fontsize=11, fontweight='bold', labelpad=3)
    top_axes[0].text(-0.22, 1.08, 'B', transform=top_axes[0].transAxes, fontsize=13, fontweight='bold', ha='left', va='top', clip_on=False)
    bottom_axes[0].text(-0.22, 1.08, 'C', transform=bottom_axes[0].transAxes, fontsize=13, fontweight='bold', ha='left', va='top', clip_on=False)
    d_col = n_methods + 3
    ax_spots = fig.add_subplot(gs[0, d_col])
    ax_clusters = fig.add_subplot(gs[1, d_col])

    pos = ax_clusters.get_position()
    ax_clusters.set_position([pos.x0, pos.y0, pos.width, pos.height - 0.025])
    x = np.arange(n_methods)
    totals = spot_counts.sum(axis=1)
    percent = np.divide(spot_counts, totals[:, None], out=np.zeros_like(spot_counts, dtype=float), where=totals[:, None] > 0) * 100
    bottoms = np.zeros(n_methods)
    for bin_i, color in enumerate(BIN_COLORS):
        ax_spots.bar(x, percent[:, bin_i], bottom=bottoms, width=0.72, color=color, edgecolor='white', linewidth=0.4)
        bottoms += percent[:, bin_i]
    ax_spots.set_ylim(0, 100)
    ax_spots.set_yticks(np.arange(0, 101, 20))
    ax_spots.set_ylabel('Percent spots', fontsize=10, fontweight='bold')
    ax_spots.set_xticks([])
    bottoms = np.zeros(n_methods)
    for bin_i, color in enumerate(BIN_COLORS):
        ax_clusters.bar(x, cluster_counts[:, bin_i], bottom=bottoms, width=0.72, color=color, edgecolor='white', linewidth=0.4)
        bottoms += cluster_counts[:, bin_i]
    ax_clusters.set_ylabel('Number of clusters', fontsize=10, fontweight='bold')
    ax_clusters.set_xticks([])
    ax_clusters.set_xlabel('Clustering Methods\n(same order as maps)', fontsize=10, fontweight='bold', labelpad=3)
    ax_clusters.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for ax in (ax_spots, ax_clusters):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7, width=0.7, length=2)
    ax_spots.text(-0.35, 1.08, 'D', transform=ax_spots.transAxes, fontsize=13, fontweight='bold', ha='left', va='top', clip_on=False)
    draw_instability_legend(fig, bottom_axes)
    out_base = OUT_DIR / f'fig2_s1_s17_{dataset}{suffix}_methods_uncertainty'
    fig.savefig(f'{out_base}.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(f'{out_base}.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_base}.png and .pdf', flush=True)
    return caption_rows

def main():
    run_lookup = {f"{ds}{'_' + sample if sample else ''}": (ds, sample)
                  for ds, sample in make_runs()}
    parser = argparse.ArgumentParser(description='Plot Figure 2 and S1–S17 using packaged nested 50-run results.')
    parser.add_argument('--samples', nargs='+', choices=list(run_lookup),
                        help='Plot only these samples; default: all 18. Example: dlpfc_151674')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    selected = list(dict.fromkeys(args.samples)) if args.samples else list(run_lookup)
    runs = [run_lookup[name] for name in selected]
    required = [DATA_PATH / f"{ds}{'_' + sample if sample else ''}.h5ad"
                for ds, sample in runs]
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError('Bundled spatial files are missing:\n'
                                + '\n'.join(missing))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    caption_rows = []
    for dataset, sample in runs:
        caption_rows.extend(plot_one(dataset, sample, dpi=args.dpi))
        gc.collect()

    suffix = ('_' + selected[0] if len(selected) == 1 else '_selected') if args.samples else ''
    summary = OUT_DIR / f'fig2_s1_s17_caption_statistics{suffix}.csv'
    pd.DataFrame(caption_rows).to_csv(summary, index=False, na_rep='NA')
    print(f'Saved {summary}', flush=True)

if __name__ == '__main__':
    main()
