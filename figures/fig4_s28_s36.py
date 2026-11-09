#!/usr/bin/env python3
'Figures 4 and S28–S36: spatial and aggregate instability at 5, 10, 20, and 40 runs.'
import argparse
from collections import deque
import gc
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from common import DATA_DIR, OUTPUT_ROOT, read_spatial_sample, histology_kwargs
from cluster_metrics import load_cluster_metrics
from sample_metrics import load_sample_metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.spatial import cKDTree

plt.rcParams.update({'font.family': 'sans-serif',
                    'font.sans-serif': ['Arial', 'DejaVu Sans'],
                    'pdf.fonttype': 42, 'ps.fonttype': 42})

FIGSIZE = (22, 13)
FIG4_LEFT = 0.08
FIG4_RIGHT = 0.995
FIG4_TOP = 0.96
FIG4_BOTTOM = 0.08
FIG4_HEIGHT_RATIOS = [1.7, 0.13, 1.2, 0.25, 0.75]
FIG4_OUTER_WSPACE = 0.16
FIG4_BC_WSPACE = 0.22
ROOT = Path(__file__).resolve().parent
OUT_DIR = OUTPUT_ROOT / 'fig4_s28_s36'
METRICS_CSV = DATA_DIR / 'metrics' / 'sample_metrics.csv'
CLUSTERWISE_METRICS_CSV = DATA_DIR / 'metrics' / 'clusterwise_metrics.csv'
RUN_COUNTS = [5, 10, 20, 40]
D_RUN_COUNTS = RUN_COUNTS.copy()
E_RUN_COUNTS = RUN_COUNTS.copy()

REFERENCE_RUNS = 50
HEATMAP_RUN_COUNTS = RUN_COUNTS.copy()
HEXBIN_RUN_COUNTS = RUN_COUNTS.copy()
MAX_HEXBIN_SPOTS_PER_SAMPLE = 10000
HEXBIN_GRIDSIZE = 45

TRANSITION_BIN_LABELS = ['[0.0,0.1)', '[0.1,0.2)', '[0.2,0.3)', '[0.3,0.4)', '[0.4,∞)']
TRANSITION_BINS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, np.inf])
HEATMAP_ANNOTATION_THRESHOLD = 0.01
HEATMAP_ANNOTATION_FONTSIZE = 7
DPI = 300
TABLEAU_20 = plt.cm.tab20.colors
D_DATASET_COLOR_INDEX = {'mouse_brain_cerebellum': 0, 'mouse_brain': 8, 'Visium_HD_Human_Colon_Cancer_cropped_square': 2, 'human_breast_cancer': 12, 'coad_ffpe': 16, 'ov_ffpe': 18, 'dlpfc': 14}
D_SCATTER_SIZE = 140
E_SCATTER_SIZE = 20
E_NON_DLPFC_ALPHA = 0.8
E_DLPFC_ALPHA = 0.3
E_DEFAULT_XMAX = 0.7
E_XMAX_STEP = 0.1
E_XCOL = 'clusterwise mean instability'
E_METRIC = 'clusterwise jaccard'
E_SINGLETON_SAMPLE_TAGS = {'mouse_brain_cerebellum', 'mouse_brain', 'Visium_HD_Human_Colon_Cancer_cropped_square', 'human_breast_cancer', 'coad_ffpe', 'ov_ffpe'}
COLUMN_WSPACE = 0.06
ROW_HSPACE = 0.01
CROP_PADDING = 0.035
SAMPLE_FONTSIZE = 20
ROW_LABEL_FONTSIZE = 22
LEGEND_FONTSIZE = 14
LEGEND_TITLE_FONTSIZE = 16
PANEL_LETTER_FONTSIZE = 28
METHODS = ['leiden', 'louvain', 'bayesspace', 'graphst', 'graphst_mclust',
           'stagate', 'stagate_mclust', 'sedr', 'sedr_mclust', 'spicemix']
METHOD_DISPLAY = {'leiden': 'Leiden', 'louvain': 'Louvain', 'bayesspace': 'BayesSpace',
    'graphst': 'GraphST', 'graphst_mclust': 'GraphST (mclust)',
    'stagate': 'STAGATE', 'stagate_mclust': 'STAGATE (mclust)',
    'spicemix': 'SpiceMix', 'sedr': 'SEDR', 'sedr_mclust': 'SEDR (mclust)'}
DATASET_DISPLAY = {'dlpfc': 'DLPFC', 'mouse_brain': 'MB', 'mouse_brain_cerebellum': 'MBC', 'human_breast_cancer': 'HBC', 'Visium_HD_Human_Colon_Cancer_cropped_square': 'CRC', 'ov_ffpe': 'OV', 'coad_ffpe': 'COAD'}
SAMPLES = [('dlpfc', sample) for sample in (
    '151507', '151508', '151509', '151510', '151669', '151670',
    '151671', '151672', '151673', '151674', '151675', '151676')]
SAMPLES += [(dataset, None) for dataset in (
    'mouse_brain', 'mouse_brain_cerebellum', 'human_breast_cancer',
    'Visium_HD_Human_Colon_Cancer_cropped_square', 'ov_ffpe', 'coad_ffpe')]
SAMPLE_TAGS = {f'{dataset}_{sample}' if sample else dataset for dataset, sample in SAMPLES}
HD_DATASETS = {'Visium_HD_Human_Colon_Cancer_cropped_square', 'coad_ffpe', 'ov_ffpe'}
LABELS = [f'[{i / 10:.1f},{(i + 1) / 10:.1f})' for i in range(9)] + ['[0.9,∞)']
BINS = np.r_[np.arange(0.0, 1.0, 0.1), np.inf]
PALETTE = dict(zip(LABELS, ['#08306b', '#4292c6', '#9ecae1', '#deebf7', '#fee391', '#fec44f', '#fdae6b', '#fd8d3c', '#fb6a4a', '#cb181d']))


def dataset_tag(dataset, sample):
    return f'{dataset}_{sample}' if sample else dataset


def sample_name(dataset, sample):
    name = DATASET_DISPLAY[dataset]
    return f'{name}\n{sample}' if sample is not None else name


def read_metrics(selected_methods):
    'Require one current metrics row for every included sample/run count.'
    df = load_sample_metrics()
    required = {'dataset', 'sample', 'iterations', 'example_mode'}
    required |= {f'{method}_{metric}' for method in selected_methods
                 for metric in ('mean_instability', 'mean_ari_all_runs')}
    missing = sorted(required - set(df))
    if missing:
        raise ValueError(f'{METRICS_CSV}: missing columns {missing}')
    df['sample'] = df['sample'].fillna('')
    df['iterations'] = pd.to_numeric(df['iterations'], errors='raise')
    cohort = {(dataset, sample or '') for dataset, sample in SAMPLES}
    keep = [(dataset, sample) in cohort for dataset, sample in zip(df['dataset'], df['sample'])]
    df = df.loc[np.asarray(keep, dtype=bool) & df['iterations'].isin(RUN_COUNTS)].copy()
    keys = list(zip(df['dataset'], df['sample'], df['iterations']))
    expected = {(dataset, sample, k) for dataset, sample in cohort for k in RUN_COUNTS}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError(f'{METRICS_CSV}: require exactly one row per included sample/run count; '
                         f'missing={sorted(expected - set(keys))}, duplicate_rows={df.duplicated(["dataset", "sample", "iterations"]).sum()}')
    if not df['example_mode'].eq('example_50').all():
        raise ValueError(f'{METRICS_CSV}: expected example_50 results')
    for method in selected_methods:
        for suffix, low, high, allow_missing in [('mean_instability', 0, np.inf, False),
                                                 ('mean_ari_all_runs', -1, 1, True)]:
            col = f'{method}_{suffix}'
            df[col] = pd.to_numeric(df[col], errors='raise')
            values = df[col].to_numpy(dtype=float)
            if (np.isinf(values).any() or ((values < low) | (values > high)).any()
                    or (not allow_missing and not np.isfinite(values).all())):
                raise ValueError(f'{METRICS_CSV}: invalid values in {col}')
    return df


def read_clusterwise_metrics(selected_methods):
    'Enforce complete fixed-example cluster coverage without admitting full HD.'
    df = load_cluster_metrics(iterations=RUN_COUNTS, methods=selected_methods)
    required = {'dataset-sample', 'method', 'iterations', 'reference_mode',
                'reference_run', 'cluster', 'n_spots', E_XCOL, E_METRIC}
    missing = sorted(required - set(df))
    if missing:
        raise ValueError(f'{CLUSTERWISE_METRICS_CSV}: missing columns {missing}')
    df['iterations'] = pd.to_numeric(df['iterations'], errors='raise')
    df = df.loc[df['dataset-sample'].isin(SAMPLE_TAGS) & df['method'].isin(selected_methods)
                & df['iterations'].isin(RUN_COUNTS)].copy()
    if df[list(required - {E_METRIC})].isna().any().any():
        raise ValueError(f'{CLUSTERWISE_METRICS_CSV}: missing required cluster values')
    if not df['reference_mode'].eq('example_50').all():
        raise ValueError(f'{CLUSTERWISE_METRICS_CSV}: expected fixed example_50 clusters')
    keys = ['dataset-sample', 'method', 'iterations', 'cluster']
    if df.duplicated(keys).any():
        raise ValueError(f'{CLUSTERWISE_METRICS_CSV}: duplicate sample/method/run/cluster rows')
    expected = {(tag, method, k) for tag in SAMPLE_TAGS
                for method in selected_methods for k in RUN_COUNTS}
    actual = set(df.groupby(keys[:3], observed=True).groups)
    if actual != expected:
        raise ValueError(f'{CLUSTERWISE_METRICS_CSV}: missing sample/method/run groups: '
                         f'{sorted(expected - actual)}')
    for column, low, high, allow_missing in [(E_XCOL, 0, np.inf, False), (E_METRIC, 0, 1, True),
                                             ('reference_run', 0, 49, False),
                                             ('n_spots', 1, np.inf, False)]:
        df[column] = pd.to_numeric(df[column], errors='raise')
        values = df[column].to_numpy(dtype=float)
        if (np.isinf(values).any() or ((values < low) | (values > high)).any()
                or (not allow_missing and not np.isfinite(values).all())):
            raise ValueError(f'{CLUSTERWISE_METRICS_CSV}: invalid {column}')
        if column in {'reference_run', 'n_spots'} and not np.equal(values, np.floor(values)).all():
            raise ValueError(f'{CLUSTERWISE_METRICS_CSV}: noninteger {column}')
    for (tag, method), group in df.groupby(['dataset-sample', 'method'], observed=True):
        if group['reference_run'].nunique() != 1:
            raise ValueError(f'{tag}/{method}: example run changes across run counts')
        cluster_sets = group.groupby('iterations', observed=True)['cluster'].agg(frozenset)
        if len(set(cluster_sets)) != 1:
            raise ValueError(f'{tag}/{method}: example cluster membership changes across run counts')
        for cluster, rows in group.groupby('cluster', observed=True):
            if rows['n_spots'].nunique() != 1 or rows[E_METRIC].nunique(dropna=False) != 1:
                raise ValueError(f'{tag}/{method}/{cluster}: fixed-cluster metrics change across run counts')
    return df


def load_sample_scores(method, metrics_df, cluster_df):
    'Load once per method; share validated, paired scores between A, B, and C.'
    metrics = metrics_df.set_index(['dataset', 'sample', 'iterations'])
    clusters = cluster_df.loc[cluster_df['method'] == method].groupby(
        ['dataset-sample', 'iterations'], observed=True)
    data = {}
    for dataset, sample in SAMPLES:
        tag = dataset_tag(dataset, sample)
        adata = read_spatial_sample(tag)
        obs = adata.obs
        path = DATA_DIR / 'spot_instability' / f'{tag}__{method}.npz'
        with np.load(path, allow_pickle=False) as bundle:
            scores = {k: np.asarray(bundle[f'scores_{k}'], dtype=float) for k in RUN_COUNTS}
        scores[REFERENCE_RUNS] = obs[f'{method}_uncertainty'].to_numpy(dtype=float)
        for k, values in scores.items():
            if values.shape != (len(obs),) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'{path}: invalid scores at {k} runs')
        for k in RUN_COUNTS:
            mean_score = float(scores[k].mean())
            expected = metrics.loc[(dataset, sample or '', k), f'{method}_mean_instability']
            if not np.isclose(mean_score, expected, rtol=1e-6, atol=1e-8):
                raise ValueError(f'{tag}/{method}/{k}: Figure S37 mean instability differs from spot scores')
            rows = clusters.get_group((tag, k))
            if rows['n_spots'].sum() != len(obs):
                raise ValueError(f'{tag}/{method}/{k}: cluster sizes do not cover all spots')
            weighted_mean = np.average(rows[E_XCOL], weights=rows['n_spots'])
            if not np.isclose(mean_score, weighted_mean, rtol=1e-6, atol=1e-8):
                raise ValueError(f'{tag}/{method}/{k}: cluster means differ from spot scores')
        data[tag] = {'scores': scores, 'obs_names': obs.index.astype(str)}
        print(f'[validated] {tag}/{method}: {len(obs)} spots', flush=True)
    return data


def spatial_context(dataset, sample, expected_names):
    'Read spatial metadata only; use the geometry conventions of current Figure S37.'
    path = DATA_DIR / 'spatial' / f'{dataset_tag(dataset, sample)}.h5ad'
    adata = read_spatial_sample(dataset_tag(dataset, sample))
    obs = adata.obs
    coords = np.asarray(adata.obsm['spatial'], dtype=float).copy()
    spatial = adata.uns.get('spatial', {})
    image_kwargs = histology_kwargs(adata)
    if not obs.index.astype(str).equals(expected_names):
        raise ValueError(f'{path}: spatial spot order differs from validated instability')
    if coords.shape != (len(obs), 2) or not len(obs) or not np.isfinite(coords).all():
        raise ValueError(f'{path}: invalid spatial coordinates')
    image = None
    if spatial:
        library = spatial[next(iter(spatial))]
        for key in ('hires', 'lowres'):
            if key in library.get('images', {}):
                image = np.asarray(library['images'][key])
                scale = float(library.get('scalefactors', {}).get(f'tissue_{key}_scalef', 1.0))
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError(f'{path}: invalid image scale {scale}')
                coords *= scale
                break
    if image is not None and image.ndim == 3:
        image = image[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])
    unique = np.unique(coords, axis=0)
    distances = cKDTree(unique).query(unique, k=2)[0][:, 1] if len(unique) > 1 else np.array([1.0])
    spacing = float(np.median(distances[np.isfinite(distances) & (distances > 0)]))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f'{path}: cannot determine spatial spot spacing')
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
    bounds = (low[0], low[1], high[0], high[1])
    return image, vertices, bounds, image_kwargs


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


def square_limits(bounds):
    min_x, min_y, max_x, max_y = bounds
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    span = max(max_x - min_x, max_y - min_y)
    span *= 1 + 2 * CROP_PADDING
    half = span / 2
    return (center_x - half, center_x + half, center_y - half, center_y + half)

def blank_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_subplot(ax, image, vertices, bounds, scores, image_kwargs):
    if image is not None:
        ax.imshow(image, cmap='gray', alpha=0.45, origin='upper', **image_kwargs)
    bins = np.digitize(scores, BINS[1:-1], right=False)
    colors = np.asarray([PALETTE[label] for label in LABELS])[bins]
    ax.add_collection(PolyCollection(vertices, facecolors=colors, edgecolors='none',
                                    linewidths=0, rasterized=True))
    x0, x1, y0, y1 = square_limits(bounds)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_anchor('C')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.35)


def add_spatial_headers(axes):
    "Use Figure S37's explicit axes-coordinate text, not set_title/pad."
    for j, (dataset, sample) in enumerate(SAMPLES):
        axes[0, j].text(0.5, 1.3, sample_name(dataset, sample), ha='center', va='center', fontsize=SAMPLE_FONTSIZE, fontweight='bold', linespacing=0.95, transform=axes[0, j].transAxes, clip_on=False)
    for i, k in enumerate(RUN_COUNTS):
        axes[i, 0].text(-0.045, 0.5, f'{k}\nruns', ha='right', va='center', fontsize=ROW_LABEL_FONTSIZE, fontweight='bold', transform=axes[i, 0].transAxes, clip_on=False)

def add_instability_legend(fig, axes):
    'Draw the instability-bin key as one horizontal row below panel A.'
    handles = [Patch(facecolor=PALETTE[label], edgecolor='none', label=label) for label in LABELS]
    draw_layout(fig)
    left_pos = axes[-1, 0].get_position()
    right_pos = axes[-1, -1].get_position()
    group_center_x = (left_pos.x0 + right_pos.x1) / 2
    legend_y = left_pos.y0 - 0.032
    legend = fig.legend(handles=handles, labels=LABELS, loc='center', bbox_to_anchor=(group_center_x, legend_y), bbox_transform=fig.transFigure, ncol=len(LABELS), frameon=False, fontsize=LEGEND_FONTSIZE, handlelength=1.3, handleheight=0.8, handletextpad=0.5, columnspacing=1.5, labelspacing=0.9, borderaxespad=0.0)
    draw_layout(fig)
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    legend_width = legend_bbox.width
    temp_title = fig.text(0.0, 0.0, 'Instability Score Bins', fontsize=LEGEND_TITLE_FONTSIZE, fontweight='bold', ha='left', va='center')
    draw_layout(fig)
    title_bbox = temp_title.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    title_width = title_bbox.width
    temp_title.remove()
    title_legend_gap = 0.01
    total_width = title_width + title_legend_gap + legend_width
    group_left = group_center_x - total_width / 2
    legend_center_x = group_left + title_width + title_legend_gap + legend_width / 2
    legend.set_bbox_to_anchor((legend_center_x, legend_y), transform=fig.transFigure)
    fig.text(group_left, legend_y, 'Instability Score Bins', fontsize=LEGEND_TITLE_FONTSIZE, fontweight='bold', ha='left', va='center')
    return legend

def plot_fig4a(fig, parent_spec, method, sample_scores):
    'Draw all four spatial rows for every included sample.'
    top_grid = parent_spec.subgridspec(nrows=len(RUN_COUNTS), ncols=len(SAMPLES),
                                      wspace=COLUMN_WSPACE, hspace=ROW_HSPACE)
    axes = np.array([[fig.add_subplot(top_grid[i, j]) for j in range(len(SAMPLES))]
                     for i in range(len(RUN_COUNTS))])
    for j, (dataset, sample) in enumerate(SAMPLES):
        tag = dataset_tag(dataset, sample)
        data = sample_scores[tag]
        print(f'[spatial] {method}/{tag}', flush=True)
        image, vertices, bounds, image_kwargs = spatial_context(dataset, sample, data['obs_names'])
        for i, k in enumerate(RUN_COUNTS):
            plot_subplot(axes[i, j], image, vertices, bounds, data['scores'][k], image_kwargs)
        del image, vertices
        gc.collect()
    add_spatial_headers(axes)
    legend = add_instability_legend(fig, axes)
    return axes, legend


def transition_category(values):
    'Map instability values into the five Figure 4B transition bins.'
    return np.digitize(values, TRANSITION_BINS[1:-1], right=False)

def build_transition_dataframe(method, sample_scores):
    'Pool exact paired spot counts; retain every sample and every finite score.'
    rows = []
    n_categories = len(TRANSITION_BIN_LABELS)
    for dataset, sample in SAMPLES:
        scores = sample_scores[dataset_tag(dataset, sample)]['scores']
        reference_category = transition_category(scores[REFERENCE_RUNS])
        for runs in HEATMAP_RUN_COUNTS:
            current_category = transition_category(scores[runs])
            counts = np.zeros((n_categories, n_categories), dtype=np.int64)
            np.add.at(counts, (reference_category, current_category), 1)
            for reference_code in range(n_categories):
                for current_code in range(n_categories):
                    rows.append({'dataset': dataset, 'sample': sample, 'method': method,
                        'runs': runs, 'reference_category': reference_code,
                        'current_category': current_code,
                        'count': counts[reference_code, current_code]})
    return pd.DataFrame(rows)

def match_colorbar_height(fig, cbar_ax, plot_axes):
    'Make the colorbar start and end at exactly the same vertical'
    draw_layout(fig)
    plot_pos = plot_axes[0].get_position()
    cbar_pos = cbar_ax.get_position()
    cbar_ax.set_position([cbar_pos.x0, plot_pos.y0, cbar_pos.width, plot_pos.height])

def plot_fig4b(fig, parent_spec, transition_df, selected_methods):
    'Draw Figure 4B: four 50-vs-run transition heatmaps plus one colorbar.'
    heatmap_df = transition_df[transition_df['method'].isin(selected_methods)]
    if heatmap_df.empty:
        raise ValueError('No transition data for selected Figure 4B methods.')
    grid = parent_spec.subgridspec(nrows=1, ncols=len(HEATMAP_RUN_COUNTS) + 1, width_ratios=[1.0] * len(HEATMAP_RUN_COUNTS) + [0.09], wspace=0.2)
    axes = np.array([fig.add_subplot(grid[0, j]) for j in range(len(HEATMAP_RUN_COUNTS))])
    cbar_ax = fig.add_subplot(grid[0, -1])
    image = None
    n_categories = len(TRANSITION_BIN_LABELS)
    for axi, (ax, runs) in enumerate(zip(axes, HEATMAP_RUN_COUNTS)):
        subset = heatmap_df[heatmap_df['runs'] == runs]
        counts = subset.groupby(['reference_category', 'current_category'])['count'].sum().unstack(fill_value=0).reindex(index=range(n_categories), columns=range(n_categories), fill_value=0).astype(float)
        matrix = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
        image = ax.imshow(matrix.to_numpy(), cmap='Blues', vmin=0, vmax=1, interpolation='nearest', aspect='equal')
        for i in range(n_categories):
            for j in range(n_categories):
                value = matrix.iloc[i, j]
                if np.isfinite(value) and value >= HEATMAP_ANNOTATION_THRESHOLD:
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=HEATMAP_ANNOTATION_FONTSIZE, color='white' if value >= 0.5 else 'black')
        ax.set_title(f'{REFERENCE_RUNS} vs. {runs}', fontsize=20, fontweight='bold', pad=8)
        ax.set_xticks(range(n_categories))
        ax.set_xticklabels(TRANSITION_BIN_LABELS, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_yticks(range(n_categories))
        if axi == 0:
            ax.set_yticklabels(TRANSITION_BIN_LABELS)
            ax.set_ylabel(f'Bins {REFERENCE_RUNS} runs', fontsize=16)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', labelsize=14)
    colorbar = fig.colorbar(image, cax=cbar_ax)
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.ax.tick_params(labelsize=14)
    colorbar.set_label('Fraction of spots', fontsize=16, labelpad=10)
    match_colorbar_height(fig, cbar_ax, axes)
    return (axes, colorbar)


def build_hexbin_dataframe(method, sample_scores):
    'Use identical sampled spots at each run count, with at most 10k per sample.'
    rng = np.random.default_rng(42)
    frames = []
    for dataset, sample in SAMPLES:
        tag = dataset_tag(dataset, sample)
        scores = sample_scores[tag]['scores']
        keep = np.arange(len(scores[REFERENCE_RUNS]))
        if keep.size > MAX_HEXBIN_SPOTS_PER_SAMPLE:
            keep = rng.choice(keep, size=MAX_HEXBIN_SPOTS_PER_SAMPLE, replace=False)
        frame = pd.DataFrame({'dataset-sample': np.repeat(tag, len(keep))})
        for runs in [REFERENCE_RUNS] + HEXBIN_RUN_COUNTS:
            frame[f'instability_{runs}runs'] = scores[runs][keep]
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)

    maximum = max(float(values.max()) for data in sample_scores.values()
                  for values in data['scores'].values())
    result.attrs['axis_max'] = max(1.0, float(np.ceil(maximum * 10) / 10))
    return result

def plot_fig4c(fig, parent_spec, spot_df, method):
    'Draw Figure 4C: four direct instability comparisons plus one colorbar.'
    grid = parent_spec.subgridspec(nrows=1, ncols=len(HEXBIN_RUN_COUNTS) + 1, width_ratios=[1.0] * len(HEXBIN_RUN_COUNTS) + [0.09], wspace=0.2)
    axes = np.array([fig.add_subplot(grid[0, j]) for j in range(len(HEXBIN_RUN_COUNTS))])
    cbar_ax = fig.add_subplot(grid[0, -1])
    images = []
    score_columns = [f'instability_{runs}runs' for runs in [REFERENCE_RUNS] + HEXBIN_RUN_COUNTS]
    observed_max = float(spot_df[score_columns].to_numpy(dtype=float).max())
    axis_max = max(1.0, float(np.ceil(observed_max * 10) / 10),
                   float(spot_df.attrs.get('axis_max', 1.0)))
    for axi, (ax, runs) in enumerate(zip(axes, HEXBIN_RUN_COUNTS)):
        x = spot_df[f'instability_{runs}runs'].to_numpy(dtype=float)
        y = spot_df[f'instability_{REFERENCE_RUNS}runs'].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        image = ax.hexbin(
                    x[valid],
                    y[valid],
                    gridsize=HEXBIN_GRIDSIZE,
                    extent=(0, axis_max, 0, axis_max),
                    mincnt=1,
                    cmap='viridis',
                    edgecolors='white',
                    linewidths=0.1,
                )
        image.set_rasterized(True)
        images.append(image)
        ax.plot([0, axis_max], [0, axis_max], color='red', linestyle='--', linewidth=1)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{REFERENCE_RUNS} vs. {runs}', fontsize=20, fontweight='bold', pad=8)
        ax.set_xlabel(f'Score {runs} runs', fontsize=16)
        ticks = np.linspace(0.0, axis_max, 5)
        tick_labels = [f'{value:g}' for value in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_yticks(ticks)
        if axi == 0:
            ax.set_yticklabels(tick_labels)
            ax.set_ylabel(f'Score {REFERENCE_RUNS} runs', fontsize=16)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', labelsize=14)
    global_vmax = max((float(np.max(image.get_array())) for image in images if image.get_array().size))
    global_vmax = max(2.0, global_vmax)
    shared_norm = LogNorm(vmin=1.0, vmax=global_vmax)
    for image in images:
        image.set_norm(shared_norm)
    colorbar = fig.colorbar(images[-1], cax=cbar_ax)
    max_exponent = int(np.floor(np.log10(global_vmax)))
    colorbar_ticks = 10.0 ** np.arange(max_exponent + 1)
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels([str(exponent) for exponent in range(max_exponent + 1)])
    colorbar.ax.tick_params(labelsize=14)
    colorbar.set_label('Spot density (log10)', fontsize=16, labelpad=10)
    match_colorbar_height(fig, cbar_ax, axes)
    return (axes, colorbar, shared_norm)


def plot_fig4d(fig, parent_spec, metrics_df, method):
    'Draw Figure 4D for one method and four run counts: 5, 10, 20, and 40.'
    x_col = f'{method}_mean_instability'
    y_col = f'{method}_mean_ari_all_runs'
    required_columns = {'iterations', 'dataset', x_col, y_col}
    missing_columns = sorted(required_columns - set(metrics_df.columns))
    if missing_columns:
        raise ValueError(f'Missing Figure 4D metrics columns: {missing_columns}')
    finite_x = metrics_df[x_col].to_numpy(dtype=float)
    finite_y = metrics_df[y_col].to_numpy(dtype=float)
    finite_y = finite_y[np.isfinite(finite_y)]
    if not len(finite_y):
        raise ValueError(f'{method}: no finite annotation ARI values for panel D')
    x_max = max(0.5, np.ceil((finite_x.max() + 0.02) * 10) / 10)
    y_min = min(0.0, np.floor(finite_y.min() * 10) / 10) - 0.02
    grid = parent_spec.subgridspec(nrows=1, ncols=len(D_RUN_COUNTS), wspace=0.3)
    axes = np.array([fig.add_subplot(grid[0, j]) for j in range(len(D_RUN_COUNTS))])
    for axi, (ax, runs) in enumerate(zip(axes, D_RUN_COUNTS)):
        tmp = metrics_df[metrics_df['iterations'] == runs]
        xs = tmp[x_col].to_numpy(dtype=float)
        ys = tmp[y_col].to_numpy(dtype=float)
        datasets = tmp['dataset'].to_numpy()
        valid = np.isfinite(xs) & np.isfinite(ys)
        xx = xs[valid]
        yy = ys[valid]
        plot_datasets = datasets[valid]
        ax.set_title(f'{runs} runs', fontsize=20, fontweight='bold', pad=8)
        ax.set_xlabel('Sample Instability', fontsize=16)
        if axi == 0:
            ax.set_ylabel('ARI w.r.t. Ref.', fontsize=16)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, 1.02)
        can_correlate = len(xx) >= 2 and np.ptp(xx) > 0 and (np.ptp(yy) > 0)
        if can_correlate:
            r, p = stats.pearsonr(xx, yy)
            r_sq = r ** 2
            print(f'[panel D] {method}, {runs} runs: r={r:.4f}, r2={r_sq:.4f}, p={p:.4g}')
            slope, intercept = np.polyfit(xx, yy, 1)
            fit_x = np.linspace(0, x_max, 100)
            fit_y = slope * fit_x + intercept
            ax.plot(fit_x, fit_y, color='black', linestyle='--', linewidth=2)
            info = f'r = {r:.2f}\nr² = {r_sq:.2f}\np = {p:.3g}'
            ax.text(0.5, -0.4, info, transform=ax.transAxes, ha='center', va='top', fontsize=14, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='gray'))
        else:
            print(f'[panel D] {method}, {runs} runs: insufficient/non-varying finite data for Pearson correlation')
        for x, y, dataset in zip(xx, yy, plot_datasets):
            color_index = D_DATASET_COLOR_INDEX.get(dataset, 14)
            alpha = 0.3 if dataset == 'dlpfc' else 1.0
            ax.scatter(x, y, color=TABLEAU_20[color_index], s=D_SCATTER_SIZE, alpha=alpha)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return axes


def prepare_clusterwise_metrics(cluster_df, method):
    'Prepare the cluster-level table used by Figure 4E for one Figure 4 method.'
    required_columns = {'dataset-sample', 'method', 'iterations', E_XCOL, E_METRIC}
    missing_columns = sorted(required_columns - set(cluster_df.columns))
    if missing_columns:
        raise ValueError(f'Missing Figure 4E clusterwise metrics columns: {missing_columns}')
    out = cluster_df.loc[cluster_df['dataset-sample'].isin(SAMPLE_TAGS)
                         & (cluster_df['method'] == method)
                         & cluster_df['iterations'].isin(E_RUN_COUNTS)].copy()
    if out.empty:
        raise ValueError(f'No Figure 4E data for method {method}.')
    return out

def figure4e_xmax(cluster_df):
    'Choose one shared x-axis maximum for all four Figure 4E panels.'
    xs = cluster_df[E_XCOL].to_numpy(dtype=float)
    ys = cluster_df[E_METRIC].to_numpy(dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(valid):
        raise ValueError('Figure 4E has no finite cluster instability/Jaccard pairs.')
    observed_max = float(np.max(xs[valid]))
    if observed_max <= E_DEFAULT_XMAX:
        x_max = E_DEFAULT_XMAX
    else:
        x_max = np.ceil((observed_max - 1e-12) / E_XMAX_STEP) * E_XMAX_STEP
        x_max = float(x_max)
    print(f'[panel E] maximum cluster instability={observed_max:.4f}; shared x-axis maximum={x_max:.2f}')
    return x_max

def dataset_mask_for_clusterwise(ydf, dataset):
    'Return rows for one dataset without defining DLPFC by exclusion.'
    sample_tags = ydf['dataset-sample'].astype(str)
    if dataset == 'dlpfc':
        return sample_tags.str.startswith('dlpfc_', na=False)
    return sample_tags == dataset

def plot_fig4e(fig, parent_spec, cluster_df, method):
    'Draw Figure 4E for one method and 5, 10, 20, and 40 runs.'
    xdf = prepare_clusterwise_metrics(cluster_df, method)
    x_max = figure4e_xmax(xdf)
    grid = parent_spec.subgridspec(nrows=1, ncols=len(E_RUN_COUNTS), wspace=0.3)
    axes = np.array([fig.add_subplot(grid[0, j]) for j in range(len(E_RUN_COUNTS))])
    for axi, (ax, runs) in enumerate(zip(axes, E_RUN_COUNTS)):
        ydf = xdf[xdf['iterations'] == runs]
        xs = ydf[E_XCOL].to_numpy(dtype=float)
        ys = ydf[E_METRIC].to_numpy(dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys)
        xx = xs[valid]
        yy = ys[valid]
        ax.set_title(f'{runs} runs', fontsize=20, fontweight='bold', pad=8)
        ax.set_xlabel('Cluster Instability', fontsize=16)
        if axi == 0:
            ax.set_ylabel('Jaccard w.r.t. Ref.', fontsize=16)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        x_ticks = np.arange(0.0, x_max + 1e-09, 0.2)
        ax.set_xticks(x_ticks)
        if axi == 0:
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        can_correlate = len(xx) >= 2 and np.ptp(xx) > 0 and (np.ptp(yy) > 0)
        if can_correlate:
            r, p = stats.pearsonr(xx, yy)
            r_sq = r ** 2
            print(f'[panel E] {method}, {runs} runs: r={r:.4f}, r2={r_sq:.4f}, p={p:.4g}')
            slope, intercept = np.polyfit(xx, yy, 1)
            fit_x = np.linspace(0, x_max, 100)
            fit_y = slope * fit_x + intercept
            ax.plot(fit_x, fit_y, color='black', linestyle='--', linewidth=2)
            info = f'r = {r:.2f}\nr² = {r_sq:.2f}\np = {p:.3g}'
            ax.text(0.5, -0.4, info, transform=ax.transAxes, ha='center', va='top', fontsize=14, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='gray'))
        else:
            print(f'[panel E] {method}, {runs} runs: insufficient/non-varying finite data for Pearson correlation')
        datasets_to_plot = ['dlpfc', 'mouse_brain_cerebellum', 'Visium_HD_Human_Colon_Cancer_cropped_square', 'human_breast_cancer', 'mouse_brain', 'coad_ffpe', 'ov_ffpe']
        for dataset in datasets_to_plot:
            zdf = ydf[dataset_mask_for_clusterwise(ydf, dataset)]
            zx = zdf[E_XCOL].to_numpy(dtype=float)
            zy = zdf[E_METRIC].to_numpy(dtype=float)
            point_valid = np.isfinite(zx) & np.isfinite(zy)
            zx = zx[point_valid]
            zy = zy[point_valid]
            if zx.size == 0:
                continue
            color_index = D_DATASET_COLOR_INDEX[dataset]
            alpha = E_DLPFC_ALPHA if dataset == 'dlpfc' else E_NON_DLPFC_ALPHA
            ax.scatter(zx, zy, color=TABLEAU_20[color_index], s=E_SCATTER_SIZE, alpha=alpha)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return (axes, x_max)

def add_sample_legend(fig, scatter_axes):
    'Shared dataset/sample legend centered below panels D and E.'
    labels = ['DLPFC', 'MB', 'MBC', 'HBC', 'CRC', 'OV', 'COAD']
    datasets = ['dlpfc', 'mouse_brain', 'mouse_brain_cerebellum', 'human_breast_cancer', 'Visium_HD_Human_Colon_Cancer_cropped_square', 'ov_ffpe', 'coad_ffpe']
    colors = [TABLEAU_20[D_DATASET_COLOR_INDEX[dataset]] for dataset in datasets]
    legend_marker_size = np.sqrt(D_SCATTER_SIZE)
    handles = [Line2D([], [], linestyle='none', marker='o', markersize=legend_marker_size, markerfacecolor=color, markeredgecolor='none', label=label) for label, color in zip(labels, colors)]
    draw_layout(fig)
    left = min((ax.get_position().x0 for ax in scatter_axes))
    right = max((ax.get_position().x1 for ax in scatter_axes))
    bottom = min((ax.get_position().y0 for ax in scatter_axes))
    x_center = (left + right) / 2
    y = bottom - 0.14
    legend = fig.legend(handles=handles, labels=labels, loc='center', bbox_to_anchor=(x_center, y), bbox_transform=fig.transFigure, ncol=len(labels), frameon=False, fontsize=16, handletextpad=0.4, columnspacing=1.7, borderaxespad=0.0)
    draw_layout(fig)
    bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(bbox.x0 - 0.015, y, 'Sample', fontsize=16, fontweight='bold', ha='right', va='center')
    return legend


def add_panel_letters(fig, spatial_axes, heatmap_axes, hexbin_axes, d_axes, e_axes):
    'Align panel letters as:'
    draw_layout(fig)
    x_left = 0.035
    x_right = e_axes[0].get_position().x0 - 0.035
    y_a = spatial_axes[0, 0].get_position().y1 + 0.035
    y_bc = max(max((ax.get_position().y1 for ax in heatmap_axes)), max((ax.get_position().y1 for ax in hexbin_axes))) + 0.03
    y_de = max(max((ax.get_position().y1 for ax in d_axes)), max((ax.get_position().y1 for ax in e_axes))) + 0.033
    positions = {'A': (x_left, y_a), 'B': (x_left, y_bc), 'C': (x_right, y_bc), 'D': (x_left, y_de), 'E': (x_right, y_de)}
    for letter, (x, y) in positions.items():
        fig.text(x, y, letter, fontsize=PANEL_LETTER_FONTSIZE, fontweight='bold', ha='left', va='top')

def add_placeholder_panel(fig, spec, letter):
    'Create an empty panel placeholder and place its panel letter.'
    ax = fig.add_subplot(spec)
    ax.set_axis_off()
    draw_layout(fig)
    pos = ax.get_position()
    fig.text(pos.x0 - 0.025, pos.y1 + 0.012, letter, fontsize=PANEL_LETTER_FONTSIZE, fontweight='bold', ha='left', va='top')
    return ax

def create_figure4(method, transition_df, hexbin_df, metrics_df, clusterwise_df, sample_scores):
    'Draw Figure 4 on the actual final-size canvas.'
    fig = plt.figure(figsize=FIGSIZE)
    outer = fig.add_gridspec(nrows=5, ncols=2, height_ratios=FIG4_HEIGHT_RATIOS, hspace=0.08, wspace=FIG4_OUTER_WSPACE, left=FIG4_LEFT, right=FIG4_RIGHT, top=FIG4_TOP, bottom=FIG4_BOTTOM)
    bc_grid = outer[2, :].subgridspec(nrows=1, ncols=2, wspace=FIG4_BC_WSPACE)
    spatial_axes, legend = plot_fig4a(fig, outer[0, :], method, sample_scores)
    heatmap_axes, heatmap_colorbar = plot_fig4b(fig, bc_grid[0, 0], transition_df, [method])
    hexbin_axes, hexbin_colorbar, hexbin_norm = plot_fig4c(fig, bc_grid[0, 1], hexbin_df, method)
    d_axes = plot_fig4d(fig, outer[4, 0], metrics_df, method)
    e_axes, e_xmax = plot_fig4e(fig, outer[4, 1], clusterwise_df, method)
    add_panel_letters(fig, spatial_axes, heatmap_axes, hexbin_axes, d_axes, e_axes)
    add_sample_legend(fig, np.concatenate([d_axes, e_axes]))
    return (fig, spatial_axes, legend, heatmap_axes, heatmap_colorbar, hexbin_axes, hexbin_colorbar, hexbin_norm, d_axes, e_axes, e_xmax)

def save_figure4(fig, method):
    'Regenerate both formats, publishing only complete validated files.'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = []
    started = time.monotonic()
    try:
        for suffix in ('png', 'pdf'):
            descriptor, name = tempfile.mkstemp(prefix=f'.fig4_s28_s36_{method}_',
                                                 suffix=f'.tmp.{suffix}', dir=OUT_DIR)
            os.close(descriptor)
            path = Path(name)
            temporary.append(path)
            print(f'[save start] {method}: {suffix.upper()}', flush=True)
            fig.savefig(path, format=suffix, dpi=DPI, bbox_inches='tight', pad_inches=0.03)
            if not valid_figure_file(path, suffix):
                raise RuntimeError(f'{method}: generated {suffix.upper()} did not pass completion checks')

        for path, suffix in zip(temporary, ('png', 'pdf')):
            destination = OUT_DIR / f'fig4_s28_s36_{method}.{suffix}'
            os.replace(path, destination)
            print(f'[saved] {destination}', flush=True)
        print(f'[saved pair] {method}: {time.monotonic() - started:.1f}s', flush=True)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
        plt.close(fig)


def valid_figure_file(path, suffix):
    'Reject empty/truncated images and unfinished Matplotlib PDF files.'
    try:
        if not path.is_file() or path.stat().st_size < 64:
            return False
        if suffix == 'png':
            with Image.open(path) as image:
                if image.format != 'PNG' or min(image.size) <= 0:
                    return False
                image.verify()
            return True
        if suffix == 'pdf':
            with path.open('rb') as handle:
                if not handle.read(8).startswith(b'%PDF-'):
                    return False
                size = path.stat().st_size
                handle.seek(max(0, size - 4096))
                tail = handle.read()
                match = re.search(rb'startxref\s+(\d+)\s+%%EOF\s*\Z', tail)
                if match is None:
                    return False
                offset = int(match.group(1))
                if not 0 < offset < size:
                    return False
                handle.seek(offset)
                return handle.read(4) == b'xref'
    except (OSError, ValueError, SyntaxError):
        return False
    return False


def render_method(method, metrics_df, clusterwise_df):
    'Each worker owns its Matplotlib state and exits after exactly one method.'
    started = time.monotonic()
    print(f'[method start] {method}, pid={os.getpid()}', flush=True)
    sample_scores = load_sample_scores(method, metrics_df, clusterwise_df)
    transition_df = build_transition_dataframe(method, sample_scores)
    hexbin_df = build_hexbin_dataframe(method, sample_scores)
    print(f'[render start] {method}: elapsed={time.monotonic() - started:.1f}s', flush=True)
    fig = create_figure4(method, transition_df, hexbin_df, metrics_df,
                         clusterwise_df, sample_scores)[0]
    print(f'[layout done] {method}: elapsed={time.monotonic() - started:.1f}s', flush=True)
    save_figure4(fig, method)
    print(f'[method done] {method}: elapsed={time.monotonic() - started:.1f}s', flush=True)


def run_parallel(methods, workers, script_path=None):
    'Launch fresh interpreters, cap concurrency, and report every worker failure.'
    script_path = Path(script_path) if script_path is not None else Path(__file__).resolve()
    log_dir = OUT_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f'{os.environ.get("SLURM_JOB_ID", "local")}-{time.time_ns()}'
    pending, active, completed, failures = deque(methods), {}, [], []
    started = time.monotonic()
    print(f'[parallel] {len(methods)} methods, up to {workers} workers; regenerating all selected plots', flush=True)
    try:
        while pending or active:
            while pending and len(active) < workers:
                method = pending.popleft()
                log_path = log_dir / f'fig4_s28_s36-{run_tag}-{method}.log'
                env = os.environ.copy()
                for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                            'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMBA_NUM_THREADS',
                            'VECLIB_MAXIMUM_THREADS'):
                    env[key] = '1'
                env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
                           MPLCONFIGDIR=str(log_dir / '.matplotlib' / method))
                handle = log_path.open('w', buffering=1)
                try:
                    process = subprocess.Popen([sys.executable, '-u', '-B', str(script_path),
                                                '--worker-method', method],
                                               stdout=handle, stderr=subprocess.STDOUT, env=env)
                except BaseException:
                    handle.close()
                    raise
                active[method] = (process, handle, log_path, time.monotonic())
                print(f'[launched] {method}: pid={process.pid}, log={log_path}', flush=True)
            for method, (process, handle, log_path, launch_time) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del active[method]
                elapsed = time.monotonic() - launch_time
                if code:
                    failures.append((method, code, log_path))
                    print(f'[failed] {method}: exit={code}, elapsed={elapsed:.1f}s, log={log_path}', flush=True)
                else:
                    completed.append(method)
                    print(f'[completed {len(completed)}/{len(methods)}] {method}: {elapsed:.1f}s', flush=True)
            if active:
                time.sleep(1)
    finally:

        for process, _, _, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, handle, _, _ in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            handle.close()
    if failures:
        raise RuntimeError('Figure 4 workers failed: ' + '; '.join(
            f'{method} (exit {code}; {path})' for method, code, path in failures))
    print(f'[all methods done] {len(completed)} figures in {time.monotonic() - started:.1f}s', flush=True)
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--workers', type=int, default=1,
                        help='Maximum simultaneous method processes (default: 1)')
    parser.add_argument('--worker-method', choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--check-inputs', action='store_true',
                        help='Validate all figure inputs without writing outputs')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    selected_methods = [args.worker_method] if args.worker_method else list(dict.fromkeys(args.methods))
    metrics_df = read_metrics(selected_methods)
    clusterwise_df = read_clusterwise_metrics(selected_methods)
    if args.check_inputs:
        for method in selected_methods:
            sample_scores = load_sample_scores(method, metrics_df, clusterwise_df)

            for dataset, sample in SAMPLES:
                spatial_context(dataset, sample,
                    sample_scores[dataset_tag(dataset, sample)]['obs_names'])
            print(f'[inputs valid] {method}: {len(SAMPLES)} samples', flush=True)
            del sample_scores
            gc.collect()
    elif args.worker_method:
        render_method(args.worker_method, metrics_df, clusterwise_df)
    else:
        workers = min(args.workers, len(selected_methods))
        if os.environ.get('SLURM_JOB_ID') and workers > int(os.environ.get('SLURM_CPUS_PER_TASK', workers)):
            parser.error('Worker count exceeds allocated Slurm CPUs')
        run_parallel(selected_methods, workers)


if __name__ == '__main__':
    main()
