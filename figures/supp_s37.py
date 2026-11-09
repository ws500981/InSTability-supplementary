#!/usr/bin/env python3
'Figure S37: spatial instability and agreement with reference annotations.'

from common import DATA_DIR as INPUT_DATA_DIR, OUTPUT_ROOT, read_spatial_sample, histology_kwargs
import argparse
import gc
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from sample_metrics import load_sample_metrics
from scipy import stats
from scipy.spatial import cKDTree

ROOT = OUTPUT_ROOT
DATA_DIR = INPUT_DATA_DIR / 'spatial'
FIG_DIR = OUTPUT_ROOT / 'supp_s37'
METRICS_CSV = INPUT_DATA_DIR / 'metrics/sample_metrics.csv'
DPI, NRUNS = 300, 50
FIGSIZE = (26.25, 23.0)
LEGEND_GAP_INCHES = 0.20
HIGHLIGHT_METHODS = ('bayesspace', 'louvain')
METHODS = ['leiden', 'louvain', 'bayesspace', 'graphst', 'graphst_mclust',
           'sedr', 'sedr_mclust', 'stagate', 'stagate_mclust', 'spicemix']
METHOD_DISPLAY = dict(zip(METHODS, ['Leiden', 'Louvain', 'BayesSpace', 'GraphST',
    'GraphST\n(mclust)', 'SEDR', 'SEDR\n(mclust)', 'STAGATE', 'STAGATE\n(mclust)', 'SpiceMix']))
HD_DATASETS = {'Visium_HD_Human_Colon_Cancer_cropped_square', 'coad_ffpe', 'ov_ffpe'}
DATASET_DISPLAY = {'dlpfc': 'DLPFC', 'mouse_brain': 'MB', 'mouse_brain_cerebellum': 'MBC',
    'human_breast_cancer': 'HBC', 'Visium_HD_Human_Colon_Cancer_cropped_square': 'CRC',
    'ov_ffpe': 'OV', 'coad_ffpe': 'COAD'}
DLPFC_SAMPLES = ['151507', '151508', '151509', '151510', '151669', '151670',
                 '151671', '151672', '151673', '151674', '151675', '151676']
SAMPLES = [('dlpfc', sample) for sample in DLPFC_SAMPLES] + [
    (dataset, None) for dataset in DATASET_DISPLAY if dataset != 'dlpfc']
SCATTER_PANELS = [(method, metric) for method in METHODS for metric in
                  (('instability', 'entropy') if method == 'bayesspace' else ('instability',))]
LABELS = [f'[{i / 10:.1f},{(i + 1) / 10:.1f})' for i in range(9)] + ['[0.9,1.0]']
BIN_COLORS = ['#08306b', '#4292c6', '#9ecae1', '#deebf7', '#fee391',
              '#fec44f', '#fdae6b', '#fd8d3c', '#fb6a4a', '#cb181d']
COLOR_INDEX = {'dlpfc': 14, 'mouse_brain': 8, 'mouse_brain_cerebellum': 0,
    'human_breast_cancer': 12, 'Visium_HD_Human_Colon_Cancer_cropped_square': 2,
    'ov_ffpe': 18, 'coad_ffpe': 16}
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

def sample_tag(dataset, sample):
    return f'{dataset}_{sample}' if sample else dataset

def read_metrics():
    df = load_sample_metrics()
    df = df.loc[df['iterations'] == NRUNS].copy()
    df['sample'] = df['sample'].fillna('')
    keys = list(zip(df['dataset'], df['sample']))
    expected = {(dataset, sample or '') for dataset, sample in SAMPLES}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Metrics CSV must contain exactly one 50-run row per sample')
    for method, metric in SCATTER_PANELS:
        columns = [f'{method}_mean_ari_all_runs', metric_column(method, metric)]
        for column in columns:
            df[column] = pd.to_numeric(df[column], errors='raise')
            if np.isinf(df[column]).any():
                raise ValueError(f'Infinite values in {column}')
    for _, row in df.iterrows():
        for method in METHODS:
            value = row[f'{method}_mean_instability']
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Missing/invalid instability for {row["dataset"]}/{method}')
        entropy = row['bayesspace_mean_entropy_all_runs']
        if not np.isfinite(entropy) or entropy < 0:
            raise ValueError(f'Missing/invalid BayesSpace entropy for {row["dataset"]}')
    return df

def spatial_context(path, dataset):
    'Read only obs, spatial coordinates/images, and provenance; build shapes once.'
    data = read_spatial_sample(path.stem)
    obs = data.obs
    coords = np.asarray(data.obsm['spatial'], dtype=float)
    spatial = data.uns.get('spatial', {})
    provenance = data.uns['instability_provenance']
    if (provenance.get('mode') != 'nested' or provenance.get('iterations') != NRUNS
            or not np.array_equal(provenance['seeds'], np.arange(42, 92))):
        raise ValueError(f'{path}: expected nested 50-run results, seeds 42..91')
    if coords.shape != (len(obs), 2) or not len(obs) or not np.isfinite(coords).all():
        raise ValueError(f'{path}: invalid spatial coordinates')
    image = None
    if spatial:
        library = spatial[next(iter(spatial))]
        for key in ('hires', 'lowres'):
            if key in library.get('images', {}):
                image = np.asarray(library['images'][key])
                coords *= float(library.get('scalefactors', {}).get(f'tissue_{key}_scalef', 1.0))
                break
    if image is not None and image.ndim == 3:
        image = image[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])
    unique = np.unique(coords, axis=0)
    spacing = float(np.median(cKDTree(unique).query(unique, k=2)[0][:, 1])) if len(unique) > 1 else 1.0
    if dataset in HD_DATASETS:
        offsets = spacing / 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    elif dataset == 'mouse_brain_cerebellum':
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        offsets = 0.48 * spacing * np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        angles = np.deg2rad(np.arange(0, 360, 60) + (0 if dataset == 'mouse_brain' else 30))
        offsets = spacing / np.sqrt(3) * np.column_stack([np.cos(angles), np.sin(angles)])
    vertices = coords[:, None, :] + offsets[None, :, :]
    low, high = vertices.min(axis=(0, 1)), vertices.max(axis=(0, 1))
    center, half = (low + high) / 2, (high - low).max() * 1.07 / 2
    bounds = (center[0] - half, center[0] + half, center[1] + half, center[1] - half)
    return obs, image, vertices, bounds, histology_kwargs(data)

def plot_spatial_panel(axes, metrics, dry_run=False):
    rows = metrics.set_index(['dataset', 'sample'])
    for col, (dataset, sample) in enumerate(SAMPLES):
        tag = sample_tag(dataset, sample)
        print(f'[spatial {col + 1:02d}/{len(SAMPLES)}] {tag}', flush=True)
        if dry_run:
            bounds = (0, 1, 1, 0)
        else:
            obs, image, vertices, bounds, image_kwargs = spatial_context(DATA_DIR / f'{tag}.h5ad', dataset)
        for row, method in enumerate(METHODS):
            ax = axes[row, col]
            if not dry_run:
                values = obs[f'{method}_uncertainty'].to_numpy(dtype=float)
                if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                    raise ValueError(f'{tag}/{method}: invalid spot instability')
                expected = rows.loc[(dataset, sample or ''), f'{method}_mean_instability']
                if not np.isclose(values.mean(), expected, rtol=1e-6, atol=1e-8):
                    raise ValueError(f'{tag}/{method}: H5AD instability differs from metrics CSV')
                bins = np.digitize(values, np.arange(0.1, 1.0, 0.1), right=False)
                if image is not None:
                    ax.imshow(image, cmap='gray', alpha=0.45, origin='upper', **image_kwargs)
                ax.add_collection(PolyCollection(vertices, facecolors=np.asarray(BIN_COLORS)[bins],
                    edgecolors='none', linewidths=0, rasterized=True))
            ax.set_xlim(bounds[:2])
            ax.set_ylim(bounds[2:])
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.35)
        if not dry_run:
            del obs, image, vertices
            gc.collect()
    for col, (dataset, sample) in enumerate(SAMPLES):
        title = DATASET_DISPLAY[dataset] + (f'\n{sample}' if sample else '')
        axes[0, col].text(0.5, 1.3, title, ha='center', va='center', fontsize=20,
            fontweight='bold', linespacing=0.95, transform=axes[0, col].transAxes, clip_on=False)
    for row, method in enumerate(METHODS):
        ax = axes[row, 0]
        if method in HIGHLIGHT_METHODS:
            ax.add_patch(FancyBboxPatch((-1.25, 0.18), 1.2, 0.68, boxstyle='round,pad=0.02',
                facecolor='#a5cbe7', edgecolor='black', linewidth=1.5, linestyle='--',
                transform=ax.transAxes, clip_on=False))
        label = METHOD_DISPLAY[method].replace('BayesSpace', 'Bayes\nSpace')
        ax.text(-0.045, 0.5, label, ha='right', va='center', fontsize=22,
                fontweight='bold', transform=ax.transAxes, clip_on=False)

def add_instability_legend(ax):
    ax.set_axis_off()

    handles = [Patch(facecolor='none', edgecolor='none', label='Instability Score Bins')]
    handles += [Patch(facecolor=color, edgecolor='none', label=label)
               for label, color in zip(LABELS, BIN_COLORS)]
    legend = ax.legend(handles=handles, loc='center', ncol=11,
        frameon=False, fontsize=14, handlelength=1.3, handleheight=0.8,
        handletextpad=0.5, columnspacing=1.5, labelspacing=0.6)
    legend.get_texts()[0].set_fontweight('bold')
    return legend

def metric_column(method, metric):
    return f'{method}_mean_entropy_all_runs' if metric == 'entropy' else f'{method}_mean_instability'

def plot_scatter_panel(axes, df, dry_run=False):
    instability = df[[f'{m}_mean_instability' for m in METHODS]].to_numpy(dtype=float)
    aris = df[[f'{m}_mean_ari_all_runs' for m in METHODS]].to_numpy(dtype=float)
    finite_aris = aris[np.isfinite(aris)]
    if not len(finite_aris) or (np.abs(finite_aris) > 1).any():
        raise ValueError('No valid reference ARIs, or ARIs outside [-1, 1]')
    xmax = max(0.5, np.ceil((np.nanmax(instability) + 0.02) * 10) / 10)
    ymin = min(0, np.floor(finite_aris.min() * 10) / 10)
    records = []
    for i, (ax, (method, metric)) in enumerate(zip(axes, SCATTER_PANELS)):
        x = df[metric_column(method, metric)].to_numpy(dtype=float)
        y = df[f'{method}_mean_ari_all_runs'].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        xx, yy = x[valid], y[valid]
        limit = max(0.5, np.ceil((np.nanmax(x) + 0.02) * 10) / 10) if metric == 'entropy' else xmax
        ax.set(xlim=(0, limit), ylim=(ymin - 0.02, 1.02))
        ax.set_title(METHOD_DISPLAY[method].replace('\n', ' '), fontsize=20, fontweight='bold')
        ax.set_xlabel('Sample Entropy' if metric == 'entropy' else 'Sample Instability', fontsize=16)
        if i == 0:
            ax.set_ylabel('ARI w.r.t. Ref.', fontsize=16)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=14)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        r = p = slope = intercept = np.nan
        if len(xx) >= 3 and np.ptp(xx) > 0 and np.ptp(yy) > 0:
            r, p = stats.pearsonr(xx, yy)
            slope, intercept = np.polyfit(xx, yy, 1)
            fit_x = np.array([0, limit])
            if not dry_run:
                ax.plot(fit_x, slope * fit_x + intercept, 'k--', linewidth=2)
        info = f'r = {r:.2f}\nr² = {r ** 2:.2f}\np = {p:.3g}' if np.isfinite(r) else 'r = NA\nr² = NA\np = NA'

        ax.annotate(info, xy=(0.5, 0), xycoords='axes fraction',
                    xytext=(0, -0.77 * 72), textcoords='offset points',
                    ha='center', va='top', fontsize=14,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='gray'))
        if not dry_run:
            for dataset in DATASET_DISPLAY:
                keep = valid & (df['dataset'].to_numpy() == dataset)
                ax.scatter(x[keep], y[keep], color=plt.cm.tab20.colors[COLOR_INDEX[dataset]],
                           s=140, alpha=0.3 if dataset == 'dlpfc' else 1.0)
        ax.spines[['top', 'right']].set_visible(False)
        records.append(dict(method=method, metric=metric, n=len(xx), r=r, r_squared=r ** 2,
                            p=p, slope=slope, intercept=intercept))
        print(f'{method}/{metric}: n={len(xx)}, r={r:.4f}, r²={r ** 2:.4f}, p={p:.4g}', flush=True)
    return pd.DataFrame(records)

def add_scatter_boxes(fig, axes):
    for method in HIGHLIGHT_METHODS:
        selected = [ax for ax, (m, _) in zip(axes, SCATTER_PANELS) if m == method]
        first, last = selected[0].get_position(), selected[-1].get_position()

        x0, y0 = first.x0 - 0.007, first.y0 - 1.64 / fig.get_figheight()
        x1, y1 = last.x1 + 0.003, first.y1 + 0.36 / fig.get_figheight()
        fig.add_artist(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0.002',
            edgecolor='black', facecolor='none', linestyle='--', linewidth=1.5,
            transform=fig.transFigure, clip_on=False, zorder=10))

def add_sample_legend(fig, axes):
    handles = [Line2D([], [], linestyle='none', marker='o', markersize=12,
        markerfacecolor=plt.cm.tab20.colors[COLOR_INDEX[dataset]], markeredgecolor='none',
        label=label.replace('\n', ' ')) for dataset, label in DATASET_DISPLAY.items()]
    p0, p1 = axes[0].get_position(), axes[-1].get_position()
    y = p0.y0 - 1.95 / fig.get_figheight()
    legend = fig.legend(handles=handles, loc='center', bbox_to_anchor=((p0.x0 + p1.x1) / 2, y),
        ncol=len(handles), frameon=False, fontsize=16, handletextpad=0.4, columnspacing=1.2)
    fig.canvas.draw()
    bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(bbox.x0 - 0.015, y, 'Sample', fontsize=16, fontweight='bold', ha='right', va='center')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='Save a layout-only preview without drawing spatial or scatter data.')
    parser.add_argument('--dpi', type=int, default=DPI)
    args = parser.parse_args()
    required = [METRICS_CSV]
    if not args.dry_run:
        required += [DATA_DIR / f'{sample_tag(d, s)}.h5ad' for d, s in SAMPLES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing bundled plotting inputs:\n' + '\n'.join(missing))
    metrics = read_metrics()
    if args.dry_run:
        print('[preview] Layout only: no spatial or scatter graphics.', flush=True)
    fig = plt.figure(figsize=FIGSIZE)
    outer = fig.add_gridspec(3, 1, height_ratios=[len(METHODS), 0.6, 1.8], hspace=0.12,
                            left=0.08, right=0.995, top=0.95, bottom=0.17)
    top = outer[0].subgridspec(len(METHODS), len(SAMPLES), wspace=0.02, hspace=0.02)
    spatial_axes = np.array([[fig.add_subplot(top[i, j]) for j in range(len(SAMPLES))]
                            for i in range(len(METHODS))])

    bounds = outer[2].get_position(fig)

    scatter_width = ((bounds.x1 - 0.04) * fig.get_figwidth()
                     / (len(SCATTER_PANELS) + 0.28 * (len(SCATTER_PANELS) - 1)))
    bottom = fig.add_gridspec(1, len(SCATTER_PANELS), left=0.04, right=bounds.x1,
                             bottom=bounds.y1 - scatter_width / fig.get_figheight(),
                             top=bounds.y1, wspace=0.28)
    scatter_axes = [fig.add_subplot(bottom[0, j]) for j in range(len(SCATTER_PANELS))]
    plot_spatial_panel(spatial_axes, metrics, dry_run=args.dry_run)
    legend = add_instability_legend(fig.add_subplot(outer[1]))
    correlations = plot_scatter_panel(scatter_axes, metrics, dry_run=args.dry_run)
    fig.canvas.draw()

    legend_top = legend.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted()).y1
    panel_a_bottom = min(ax.get_position().y0 for ax in spatial_axes[-1])
    shift = panel_a_bottom - LEGEND_GAP_INCHES / fig.get_figheight() - legend_top
    for ax in [legend.axes, *scatter_axes]:
        pos = ax.get_position()

        extra = 0 if ax is legend.axes else 0.20 / fig.get_figheight()
        ax.set_position([pos.x0, pos.y0 + shift + extra, pos.width, pos.height])
    add_scatter_boxes(fig, scatter_axes)
    add_sample_legend(fig, scatter_axes)
    for letter, ax in [('A', spatial_axes[0, 0]), ('B', scatter_axes[0])]:
        fig.text(0.03, ax.get_position().y1 + 0.035, letter,
                 fontsize=28, fontweight='bold', ha='left', va='top')
    if args.dry_run:
        renderer = fig.canvas.get_renderer()
        legend_box = legend.get_window_extent(renderer)
        panel_b_top = max(ax.get_tightbbox(renderer).y1 for ax in scatter_axes)
        panel_b_top = max(panel_b_top, *(artist.get_window_extent(renderer).y1
                                       for artist in fig.artists))
        if legend_box.y0 <= panel_b_top:
            raise ValueError('Instability legend overlaps Panel B; increase the legend spacing.')
        p0, px, py = [ax.get_position() for ax in
                      (spatial_axes[0, 0], spatial_axes[0, 1], spatial_axes[1, 0])]
        pb = scatter_axes[0].get_position()
        print(f'[layout] Column gap: {(px.x0 - p0.x1) * FIGSIZE[0]:.3f} in; '
              f'row gap: {(p0.y0 - py.y1) * FIGSIZE[1]:.3f} in; '
              f'A-to-legend gap: {panel_a_bottom * FIGSIZE[1] - legend_box.y1 / fig.dpi:.3f} in; '
              f'legend clearance above B: {(legend_box.y0 - panel_b_top) / fig.dpi:.3f} in; '
              f'B plot size: {pb.width * FIGSIZE[0]:.3f} x {pb.height * FIGSIZE[1]:.3f} in',
              flush=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    basename = 's37_layout_preview' if args.dry_run else 's37'
    for suffix in ('png', 'pdf'):
        path = FIG_DIR / f'{basename}.{suffix}'
        fig.savefig(path, dpi=args.dpi, bbox_inches='tight', pad_inches=0.03)
        print(f'[saved] {path}', flush=True)
    plt.close(fig)
    if args.dry_run:
        return
    path = FIG_DIR / 's37_correlations.csv'
    correlations.to_csv(path, index=False, float_format='%.8g', na_rep='')
    print(f'[saved] {path}', flush=True)

if __name__ == '__main__':
    main()
