"""Reproduce Figure 3 from saved spatial and DEG summaries."""

from pathlib import Path
import dgea_data
from common import OUTPUT_ROOT, read_spatial_sample
import argparse
import anndata as ad
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch, FancyBboxPatch
import numpy as np
import pandas as pd
import spatialdata as sd

import spatialdata_plot  # Registers the spatial plotting accessor.
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from shapely import box as shapely_box
from shapely.geometry import Polygon
from spatialdata.models import Image2DModel, ShapesModel, TableModel


plt.rcParams['font.family'] = 'Arial'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
FIGSIZE = (22, 13.5)
DPI = 300


FIG3_LEFT = 0.08
FIG3_RIGHT = 0.995
FIG3_TOP = 0.96
FIG3_BOTTOM = 0.08
FIG3_OUTER_HSPACE = 0.08
FIG3_OUTER_WSPACE = 0.16


FIG3_HEIGHT_RATIOS = [0.8, 0.8, 1, 1, 1.1]
PANEL_LETTER_XPAD = 0.035
PANEL_LETTER_RIGHT_XPAD = 0.05
PANEL_LETTER_YPAD = 0.037
PANEL_LETTER_A_YPAD = 0.015
PANEL_B_XSHIFT = -0.025


SHOW_PANEL_GUIDES = False


FIG3_METHOD = 'sedr_mclust'
FIG3C_COMPARISON = 'cross_run'
METHODS = ['leiden', 'louvain', 'bayesspace', 'graphst', 'graphst_mclust',
           'sedr', 'sedr_mclust', 'stagate', 'stagate_mclust', 'spicemix']
FIG3_ANALYSIS_METHODS = METHODS.copy()
METHOD_LABELS = {
    'leiden': 'Leiden', 'louvain': 'Louvain', 'bayesspace': 'BayesSpace',
    'graphst': 'GraphST', 'graphst_mclust': 'GraphST-mclust',
    'sedr': 'SEDR', 'sedr_mclust': 'SEDR-mclust',
    'stagate': 'STAGATE', 'stagate_mclust': 'STAGATE-mclust',
    'spicemix': 'SpiceMix',
}
FIG3_ANALYSIS_SCOPE = 'all_methods'
OUT_DIR = OUTPUT_ROOT / 'fig3' / FIG3_ANALYSIS_SCOPE
SAVE_FORMATS = ['png', 'pdf']
OUT_STEM = OUT_DIR / 'fig3'
EXAMPLE_DATASET, EXAMPLE_SAMPLE = 'human_breast_cancer', None


N_RUNS = 50
SELECTED_EXAMPLE_CLUSTERS = [5, 7, 15, 18, 20]

SELECTED_EXAMPLE_COLORS = {
    '5': '#fe8011', '7': '#1b7837', '15': '#fa0000',
    '18': '#ab43fc', '20': '#8d574c',
}
SELECTED_REFERENCE_LABELS = [
    'IDC_1', 'Healthy_1', 'DCIS/LCIS_3', 'IDC_4', 'DCIS/LCIS_4',
]
MAX_DISPLAY_CLUSTERS = 6
RUN_GRID_COLUMNS = 17


FIG3B_DATASET_SAMPLES = [('dlpfc', '151507'), ('dlpfc', '151508'), ('dlpfc', '151509'), ('dlpfc', '151510'), ('dlpfc', '151669'), ('dlpfc', '151670'), ('dlpfc', '151671'), ('dlpfc', '151672'), ('dlpfc', '151673'), ('dlpfc', '151674'), ('dlpfc', '151675'), ('dlpfc', '151676'), ('mouse_brain', None), ('mouse_brain_cerebellum', None), ('human_breast_cancer', None), ('Visium_HD_Human_Colon_Cancer_cropped_square', None), ('ov_ffpe', None), ('coad_ffpe', None)]
FIG3B_RANDOM_SEED = 42
FIG3B_INNER_WSPACE = 0.16
FIG3B_TITLE_FONTSIZE = 19
FIG3B_AXIS_LABEL_FONTSIZE = 15
FIG3B_TICK_FONTSIZE = 13.5
FIG3B_SCATTER_SIZE = 34
FIG3B_SCATTER_XMAX = 0.7
FIG3B_HEIGHT_RATIOS = [1.0, 0.32]


FIG3C_DATASET_SAMPLES = FIG3B_DATASET_SAMPLES.copy()
FIG3C_TOP_KS = [10, 20, 50, 200]
FIG3C_METRICS = {'jaccard': {'column': 'jaccard_mean', 'ylabel': 'Jaccard'}, 'abs_rank_diff': {'column': 'rank_abs_diff_shared_mean', 'ylabel': 'Abs rank diff (shared)'}}
FIG3C_INNER_WSPACE = 0.3
FIG3C_INNER_HSPACE = 0.05
FIG3C_TITLE_FONTSIZE = 20
FIG3C_AXIS_LABEL_FONTSIZE = 16
FIG3C_TICK_FONTSIZE = 14
FIG3C_SHARED_XLABEL_FONTSIZE = 16
FIG3C_HEIGHT_RATIOS = [1.0, 0.18, 1.0, 0.42, 0.2]


FIG3D_METRICS = [('clusterwise jaccard', 'Jaccard'), ('clusterwise recall', 'Recall'), ('clusterwise purity (precision)', 'Purity')]
FIG3D_XCOL = 'clusterwise mean instability'
FIG3D_ITERATIONS = 50
TABLEAU_20 = plt.cm.tab20.colors
FIG3D_DATASET_COLOR_INDEX = {'mouse_brain_cerebellum': 0, 'mouse_brain': 8, 'Visium_HD_Human_Colon_Cancer_cropped_square': 2, 'human_breast_cancer': 12, 'coad_ffpe': 16, 'ov_ffpe': 18, 'dlpfc': 14}
FIG3D_SCATTER_SIZE = 20
FIG3D_XMAX = 0.7
FIG3D_NON_DLPFC_ALPHA = 0.8
FIG3D_DLPFC_ALPHA = 0.3
FIG3D_TITLE_FONTSIZE = 20
FIG3D_AXIS_LABEL_FONTSIZE = 16
FIG3D_TICK_FONTSIZE = 14
FIG3D_STATS_FONTSIZE = 14
FIG3D_INNER_WSPACE = 0.18
FIG3D_INNER_HSPACE = 0.04
FIG3D_HEIGHT_RATIOS = [1.0, 0.26, 0.3]


COLUMN_WSPACE = 0.06
ROW_HSPACE = 0.02
CROP_PADDING = 0.035
SPATIAL_IMAGE_ALPHA = 0.45
SPATIAL_CIRCLE_SIZE = 0.48
SPATIAL_SQUARE_SIZE = 1.0
SPATIAL_FLAT_HEX_SIZE = 1.0
SPATIAL_POINTY_HEX_SIZE = 1.0
SPATIAL_FRAME_LINEWIDTH = 0.35
ROW_LABEL_FONTSIZE = 18
CONTEXT_TITLE_FONTSIZE = 16
LEGEND_FONTSIZE = 14
REFERENCE_LEGEND_FONTSIZE = 12
EXAMPLE_LABEL_FONTSIZE = 15
CONTEXT_FRAME_LINEWIDTH = 3.0
CONTEXT_WORKFLOW_FONTSIZE = 13
CONTEXT_BINNED_CLUSTER_BINS = ['[0.0,0.1)', '[0.2,0.3)', '[0.4,0.5)']


ANNOTATION_COLUMN = 'annotation'
OTHER_SPOTS_COLOR = '#d9d9d9'
COLLISION_COLOR = '#3049ad'
INSTABILITY_LABELS = [f'[{i / 10:.1f},{(i + 1) / 10:.1f})' for i in range(9)] + ['[0.9,1.0]']
INSTABILITY_BINS = np.r_[np.arange(0.0, 1.0, 0.1), np.nextafter(1.0, np.inf)]
BIN_COLORS = {'[0.0,0.1)': '#08306b', '[0.1,0.2)': '#4292c6', '[0.2,0.3)': '#9ecae1', '[0.3,0.4)': '#deebf7', '[0.4,0.5)': '#fee391', '[0.5,0.6)': '#fec44f', '[0.6,0.7)': '#fdae6b', '[0.7,0.8)': '#fd8d3c', '[0.8,0.9)': '#fb6a4a', '[0.9,1.0]': '#cb181d'}
COLOR_LIST = ['#3049ad', '#fe8011', '#1b7837', '#fa0000', '#ab43fc', '#8d574c', '#ff00d9', '#bcbd22', '#17becf', '#8baaf3', '#ffbb79', '#99df8b', '#fe7775', '#c6b1d4', '#c49d95', '#ff80c6', '#dcdb91', '#a7d1e6', '#393b79', '#8c6d31', '#0aac00', '#982109', '#7b4173', '#713230', '#ff008c', '#637939', '#e7cb94', '#ccefc5', '#efcece', '#f7b6d2', '#eeedc8']
VISIUM_HD_DATASETS = {'Visium_HD_Human_Colon_Cancer_cropped_square', 'ov_ffpe', 'coad_ffpe'}


def safe_name(value):
    return str(value).replace('/', '_').replace('\\', '_').replace(' ', '_')

def dataset_tag(dataset, sample):
    if sample is None:
        return dataset
    return f'{dataset}_{sample}'



def read_result_table(dataset, sample, method, filename, required):
    if (dataset, sample, method) != (EXAMPLE_DATASET, EXAMPLE_SAMPLE, FIG3_METHOD):
        raise ValueError('Packaged spatial showcase is human breast cancer / SEDR-mclust')
    frame = dgea_data.showcase_table(filename)
    if frame.empty or not set(required).issubset(frame.columns):
        raise ValueError(f'Invalid showcase input: {filename}')
    return frame


def reference_labels(annotation):
    text = annotation.astype('string')
    missing = text.isna() | text.str.strip().str.lower().isin(
        ['', 'na', 'n/a', 'nan', 'none', '<na>'])
    return text.mask(missing, 'NA').astype(str)

def load_showcase_input(dataset, sample, method):
    adata = read_spatial_sample(dataset_tag(dataset, sample))
    results = np.asarray(adata.uns[f'{method}_results'])
    if not adata.obs_names.is_unique or results.shape != (N_RUNS, adata.n_obs):
        raise ValueError('Invalid showcase spot IDs or clustering dimensions')
    if pd.isna(results).any():
        raise ValueError('Missing showcase cluster labels')
    if np.issubdtype(results.dtype, np.number):
        if not np.isfinite(results).all():
            raise ValueError('Nonfinite showcase cluster labels')
        if np.equal(results, np.rint(results)).all():
            results = results.astype(np.int64)
    return adata, results.astype(str)


def cluster_string(value):
    if pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def load_alignment(dataset, sample, method):
    required = {'example_cluster', 'target_run', 'target_cluster', 'best_jaccard',
                'example_mean_instability', 'example_fine_bin', 'example_n_cells'}
    matches = read_result_table(dataset, sample, method, 'cross_run_matches.csv', required)
    matches = matches.rename(columns={'target_run': 'run', 'target_cluster': 'matched_run_cluster',
                                      'target_cluster_size': 'run_cluster_size'})
    matches['run'] = pd.to_numeric(matches['run'], errors='raise').astype(int)
    example_run = int(matches['example_run_idx'].iloc[0])
    expected = set(range(N_RUNS)) - {example_run}
    if matches.duplicated(['example_cluster', 'run']).any():
        raise ValueError('Duplicate example/target run matches')
    if not all(set(group['run']) == expected for _, group in matches.groupby('example_cluster')):
        raise ValueError('Each example cluster must have the other 49 runs')
    meta = read_result_table(dataset, sample, method, 'example_clusters.csv',
                             {'example_cluster', 'example_n_cells'})
    if (meta['example_cluster'].duplicated().any()
            or set(meta['example_cluster']) != set(matches['example_cluster'])
            or not meta['example_run_idx'].eq(example_run).all()):
        raise ValueError('Example-cluster metadata disagrees with alignment')
    identity = meta.copy()
    identity['run'] = example_run
    identity['matched_run_cluster'] = identity['example_cluster']
    identity['best_jaccard'] = 1.0
    identity['overlap'] = identity['example_n_cells']
    identity['example_cluster_size'] = identity['example_n_cells']
    identity['run_cluster_size'] = identity['example_n_cells']
    identity['is_example_run'] = True
    matches['is_example_run'] = False
    return pd.concat([matches, identity], ignore_index=True)


def get_image_and_coords(adata, preferred_key='hires'):
    coords = np.asarray(adata.obsm['spatial'], dtype=float)
    spatial = adata.uns.get('spatial', {})
    if not isinstance(spatial, dict) or not spatial:
        return (None, coords)
    library = spatial[next(iter(spatial))]
    images = library.get('images', {})
    scales = library.get('scalefactors', {})
    keys = ('hires', 'lowres') if preferred_key == 'hires' else ('lowres', 'hires')
    for key in keys:
        if key in images:
            scale = float(scales.get(f'tissue_{key}_scalef', 1.0))
            return (np.asarray(images[key]), coords * scale)
    return (None, coords)

def to_grayscale(image):
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
    unique_coords = np.unique(np.asarray(coords, dtype=float), axis=0)
    if len(unique_coords) < 2:
        return 1.0
    distances = cKDTree(unique_coords).query(unique_coords, k=2)[0][:, 1]
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if distances.size:
        return float(np.median(distances))
    return 1.0

def make_hexagons(coords, radius, angles):
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    return [Polygon(np.column_stack((x + radius * cos_angles, y + radius * sin_angles))) for x, y in coords]

def make_shapes(coords, index, dataset):
    spacing = nearest_spacing(coords)
    if dataset == 'mouse_brain_cerebellum':
        gdf = gpd.GeoDataFrame({'radius': np.full(len(coords), SPATIAL_CIRCLE_SIZE * spacing)}, geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]), index=index)
    elif dataset in VISIUM_HD_DATASETS:
        half_size = SPATIAL_SQUARE_SIZE * spacing / 2
        gdf = gpd.GeoDataFrame(geometry=shapely_box(coords[:, 0] - half_size, coords[:, 1] - half_size, coords[:, 0] + half_size, coords[:, 1] + half_size), index=index)
    elif dataset == 'mouse_brain':
        radius = SPATIAL_FLAT_HEX_SIZE * spacing / np.sqrt(3)
        angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        gdf = gpd.GeoDataFrame(geometry=make_hexagons(coords, radius, angles), index=index)
    elif dataset in {'dlpfc', 'human_breast_cancer'}:
        radius = SPATIAL_POINTY_HEX_SIZE * spacing / np.sqrt(3)
        angles = np.deg2rad([30, 90, 150, 210, 270, 330])
        gdf = gpd.GeoDataFrame(geometry=make_hexagons(coords, radius, angles), index=index)
    else:
        gdf = gpd.GeoDataFrame({'radius': np.full(len(coords), SPATIAL_CIRCLE_SIZE * spacing)}, geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]), index=index)
    return ShapesModel.parse(gdf)

def make_spatialdata(adata, dataset):
    image, coords = get_image_and_coords(adata)
    index = pd.Index(adata.obs_names.astype(str), name='instance_id')
    shapes = make_shapes(coords, index, dataset)
    table_obs = pd.DataFrame(index=index)
    table_obs['region'] = pd.Categorical(['spots'] * adata.n_obs)
    table_obs['instance_id'] = index
    table = TableModel.parse(ad.AnnData(obs=table_obs), region='spots', region_key='region', instance_key='instance_id')
    images = {}
    if image is not None:
        images['histology'] = Image2DModel.parse(to_grayscale(image)[None, ...], dims=('c', 'y', 'x'), c_coords=['gray'])
    sdata = sd.SpatialData(images=images, shapes={'spots': shapes}, tables={'table': table})
    return (sdata, (coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()))


def square_limits(bounds):
    min_x, min_y, max_x, max_y = bounds
    x_span = max(max_x - min_x, 1)
    y_span = max(max_y - min_y, 1)
    px = CROP_PADDING * x_span
    py = CROP_PADDING * y_span
    return (min_x - px, max_x + px, min_y - py, max_y + py)

def apply_spatial_view(ax, bounds):
    x0, x1, y0, y1 = square_limits(bounds)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_anchor('C')

def keep_black_frame(ax, linewidth=SPATIAL_FRAME_LINEWIDTH):
    ax.set_axis_on()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(linewidth)
        spine.set_linestyle('-')

def style_example_frame(ax):
    style_rounded_spatial_frame(ax, linewidth=3.0, linestyle=(0, (3, 2)), rounding=0.025)

def style_rounded_spatial_frame(ax, linewidth=3.0, linestyle='-', rounding=0.025):
    for spine in ax.spines.values():
        spine.set_visible(False)
    old_frame = getattr(ax, '_rounded_spatial_frame', None)
    if old_frame is not None:
        old_frame.remove()
    frame = FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes, boxstyle=f'round,pad=0,rounding_size={rounding}', fill=False, edgecolor='black', linewidth=linewidth, linestyle=linestyle, clip_on=False, zorder=20)
    ax.add_patch(frame)
    ax._rounded_spatial_frame = frame
    return frame


def render_categorical_map(ax, sdata, bounds, color, palette, groups=None, fill_alpha=1.0, show_histology=True):
    plot = sdata

    if show_histology and 'histology' in sdata.images:
        plot = plot.pl.render_images(element='histology', cmap='gray', alpha=SPATIAL_IMAGE_ALPHA)
    render_kwargs = dict(element='spots', color=color, palette=palette, fill_alpha=fill_alpha, outline_width=0, table_name='table', method='matplotlib')
    if groups is not None:
        render_kwargs['groups'] = groups
        render_kwargs['na_color'] = '#00000000'
    plot.pl.render_shapes(**render_kwargs).pl.show(ax=ax, title='', frameon=True, legend_loc=None, colorbar=False, show=False)

    for collection in ax.collections:
        collection.set_rasterized(True)

    apply_spatial_view(ax, bounds)
    keep_black_frame(ax)


def load_spot_instability(base_adata, dataset, sample, method):
    values = base_adata.obs[f'{method}_uncertainty'].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f'{dataset_tag(dataset, sample)}/{method}: invalid spot instability')
    return values

def add_spot_instability_to_table(sdata, values):
    sdata.tables['table'].obs['spot_instability_bin'] = pd.cut(values, bins=INSTABILITY_BINS, labels=INSTABILITY_LABELS, include_lowest=True, right=False, ordered=True)


def prepare_figure3_data(dataset, sample, method, selected_example_clusters):
    adata, results = load_showcase_input(dataset, sample, method)
    matches = load_alignment(dataset, sample, method)
    meta = read_result_table(dataset, sample, method, 'example_clusters.csv',
                             {'example_cluster', 'example_mean_instability', 'example_fine_bin'})
    example_run = int(meta['example_run_idx'].iloc[0])
    expected_example = min(range(N_RUNS), key=lambda run: (len(pd.unique(results[run])), run))
    if example_run != expected_example:
        raise ValueError('Saved example run is not the first run with the fewest clusters')
    if set(meta['example_cluster']) != set(results[example_run]):
        raise ValueError('Saved example clusters differ from the packaged labels')
    instability = load_spot_instability(adata, dataset, sample, method)
    for row in meta.itertuples():
        actual = float(instability[results[example_run] == row.example_cluster].mean())
        if not np.isclose(actual, row.example_mean_instability):
            raise ValueError('Saved cluster instability differs from the packaged input')
    if selected_example_clusters is None:
        ordered = meta.sort_values(['example_mean_instability', 'example_cluster'])
        positions = np.linspace(0, len(ordered) - 1, min(MAX_DISPLAY_CLUSTERS, len(ordered)), dtype=int)
        selected_example_clusters = ordered.iloc[positions]['example_cluster'].tolist()
    else:
        selected_example_clusters = [cluster_string(value) for value in selected_example_clusters]
    if (not selected_example_clusters or len(set(selected_example_clusters)) != len(selected_example_clusters)
            or not set(selected_example_clusters).issubset(set(meta['example_cluster']))
            or len(selected_example_clusters) > len(COLOR_LIST)):
        raise ValueError('Invalid SELECTED_EXAMPLE_CLUSTERS for the new example run')
    selected_matches = matches.loc[matches['example_cluster'].isin(selected_example_clusters)].copy()
    annotation = reference_labels(adata.obs[ANNOTATION_COLUMN])
    reference = read_result_table(dataset, sample, method, 'reference_matches.csv',
                                  {'example_cluster', 'target_cluster', 'best_jaccard'})
    if reference['example_cluster'].duplicated().any() or set(reference['example_cluster']) != set(meta['example_cluster']):
        raise ValueError('Expected one reference match per example cluster')
    if not reference['example_run_idx'].eq(example_run).all():
        raise ValueError('Reference alignment uses a different example run')
    lookup = reference.set_index('example_cluster')
    cluster_colors = dict(zip(selected_example_clusters, COLOR_LIST))
    cluster_colors.update({cluster: color for cluster, color in SELECTED_EXAMPLE_COLORS.items()
                           if cluster in cluster_colors})
    annotation_categories = sorted(pd.unique(annotation))
    reference_palette = {label: OTHER_SPOTS_COLOR for label in annotation_categories}
    reference_examples = {}
    for cluster in selected_example_clusters:
        label = lookup.loc[cluster, 'target_cluster']
        reference_examples.setdefault(label, []).append(cluster)


    for label, clusters in reference_examples.items():
        reference_palette[label] = cluster_colors[clusters[0]] if len(clusters) == 1 else COLLISION_COLOR
    if SELECTED_REFERENCE_LABELS is None:
        selected_reference_labels = list(reference_examples)
    else:
        selected_reference_labels = [str(label) for label in SELECTED_REFERENCE_LABELS]
        if (not selected_reference_labels
                or len(set(selected_reference_labels)) != len(selected_reference_labels)):
            raise ValueError('SELECTED_REFERENCE_LABELS must contain unique display labels')
    missing_reference_labels = set(selected_reference_labels) - set(annotation_categories)
    if missing_reference_labels:
        raise ValueError(f'Selected reference labels are absent from the input annotation: {sorted(missing_reference_labels)}')
    selected_meta = meta.set_index('example_cluster').loc[selected_example_clusters].reset_index()
    present_bins = [label for label in INSTABILITY_LABELS if label in set(selected_meta['example_fine_bin'])]
    if CONTEXT_BINNED_CLUSTER_BINS is None:
        context_bins = present_bins
    else:
        context_bins = list(CONTEXT_BINNED_CLUSTER_BINS)
        if not context_bins or not set(context_bins).issubset(present_bins):
            raise ValueError('Configured context bins must contain selected example clusters')
    sdata, bounds = make_spatialdata(adata, dataset)
    add_spot_instability_to_table(sdata, instability)
    display_selection = selected_meta.merge(
        reference[['example_cluster', 'target_cluster', 'best_jaccard']], on='example_cluster', validate='one_to_one')
    display_selection['cluster_color'] = display_selection['example_cluster'].map(cluster_colors)
    display_selection['reference_display_color'] = display_selection['target_cluster'].map(reference_palette)
    display_selection['reference_match_displayed'] = display_selection['target_cluster'].isin(selected_reference_labels)

    display_selection['display_reference_label'] = display_selection['target_cluster'].where(
        display_selection['reference_match_displayed'])
    display_selection['display_reference_color'] = display_selection['display_reference_label'].map(reference_palette)
    display_selection['reference_selection_mode'] = ('matched' if SELECTED_REFERENCE_LABELS is None else 'explicit')
    display_selection['reference_color_source'] = 'example_max_jaccard_match'
    return dict(dataset=dataset, sample=sample, method=method, adata=adata, results=results,
                matches=matches, selected_matches=selected_matches,
                selected_example_clusters=selected_example_clusters, example_run=example_run,
                example_labels=results[example_run], annotation=annotation,
                annotation_categories=annotation_categories, selected_reference_labels=selected_reference_labels,
                cluster_colors=cluster_colors, reference_palette=reference_palette,
                reference_examples=reference_examples, context_bins=context_bins,
                display_selection=display_selection, sdata=sdata, bounds=bounds)


def build_aligned_display(run_index, labels, selected_matches, selected_example_clusters, cluster_colors):
    run_matches = selected_matches.loc[selected_matches['run'].eq(run_index)].copy()
    target_to_examples = run_matches.groupby('matched_run_cluster')['example_cluster'].agg(list).to_dict()
    labels = np.asarray(labels).astype(str)
    labels_present = set(labels)
    display = np.full(len(labels), 'Other', dtype=object)
    run_collision_rows = []
    for matched_cluster, example_clusters in target_to_examples.items():
        if matched_cluster not in labels_present:
            raise ValueError(f'Matched cluster {matched_cluster} is absent from run {run_index}.')
        if len(example_clusters) == 1:
            display[labels == matched_cluster] = example_clusters[0]
        else:
            display[labels == matched_cluster] = 'Collision'
            run_collision_rows.append({'run': run_index, 'displayed_run': run_index + 1, 'matched_run_cluster': matched_cluster, 'example_clusters': ','.join(example_clusters)})
    categories = ['Other'] + list(selected_example_clusters)
    palette = {'Other': OTHER_SPOTS_COLOR, **cluster_colors}
    visible_groups = list(selected_example_clusters)
    if run_collision_rows:
        categories.append('Collision')
        palette['Collision'] = COLLISION_COLOR
        visible_groups.append('Collision')
    return (display, categories, palette, visible_groups, run_collision_rows)

def plot_aligned_run(ax, data, run_index, record_collisions=True):
    display, categories, palette, visible_groups, run_collision_rows = build_aligned_display(run_index=run_index, labels=data['results'][run_index], selected_matches=data['selected_matches'], selected_example_clusters=data['selected_example_clusters'], cluster_colors=data['cluster_colors'])
    data['sdata'].tables['table'].obs['aligned_cluster'] = pd.Categorical(display, categories=categories, ordered=True)
    render_categorical_map(ax=ax, sdata=data['sdata'], bounds=data['bounds'], color='aligned_cluster', palette=palette, groups=visible_groups, fill_alpha=0.98)
    if run_index == data['example_run']:
        style_example_frame(ax)
    if record_collisions:
        return run_collision_rows
    return []

def match_axes_size_to_reference(fig, reference_ax, axes_to_resize, scale=1.2, allow_enlarge=True):
    fig.canvas.draw()
    ref_pos = reference_ax.get_position()
    target_width = ref_pos.width * scale
    target_height = ref_pos.height * scale
    if not allow_enlarge and axes_to_resize:
        fit_scale = min(1.0, *(ax.get_position().width / target_width for ax in axes_to_resize),
                        *(ax.get_position().height / target_height for ax in axes_to_resize))
        target_width *= fit_scale
        target_height *= fit_scale
    for ax in axes_to_resize:
        pos = ax.get_position()
        center_x = (pos.x0 + pos.x1) / 2
        center_y = (pos.y0 + pos.y1) / 2
        ax.set_position([center_x - target_width / 2, center_y - target_height / 2, target_width, target_height])

def add_run_row_label(ax, text):
    ax.text(-0.12, 0.5, text, ha='right', va='center', fontsize=ROW_LABEL_FONTSIZE, fontweight='bold', linespacing=0.95, transform=ax.transAxes, clip_on=False)

def draw_example_label_cell(ax, example_run, n_selected):
    ax.set_axis_off()
    ax.annotate('', xy=(0.04, 0.52), xytext=(0.28, 0.52),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', linewidth=1.1, color='black'))
    ax.text(0.58, 0.72, 'Example\nRun', ha='center', va='center',
            fontsize=EXAMPLE_LABEL_FONTSIZE, fontweight='bold', linespacing=0.95,
            transform=ax.transAxes)
    ax.text(0.58, 0.25, f'1st with\nmin. clusters\n({n_selected} shown)',
            ha='center', va='center', fontsize=11, transform=ax.transAxes)


def plot_fig3a(fig, parent_spec, data):
    run_slots = list(range(N_RUNS))
    run_slots.insert(data['example_run'] + 1, None)
    if len(run_slots) != 3 * RUN_GRID_COLUMNS:
        raise ValueError('Figure 3A requires 50 run cells plus one annotation cell.')
    panel_grid = parent_spec.subgridspec(nrows=3, ncols=1, hspace=ROW_HSPACE)
    run_axes, row_first_axes, collision_rows = {}, [], []
    example_label_ax = None
    for row in range(3):
        items = run_slots[row * RUN_GRID_COLUMNS:(row + 1) * RUN_GRID_COLUMNS]
        row_grid = panel_grid[row, 0].subgridspec(
            nrows=1, ncols=RUN_GRID_COLUMNS, wspace=COLUMN_WSPACE)
        first_run_ax = None
        for column, run in enumerate(items):
            ax = fig.add_subplot(row_grid[0, column])
            if run is None:
                example_label_ax = ax
                draw_example_label_cell(ax, data['example_run'],
                                        len(data['selected_example_clusters']))
                continue
            if first_run_ax is None:
                first_run_ax = ax
            run_axes[run] = ax
            collision_rows.extend(plot_aligned_run(ax, data, run, record_collisions=True))
        row_runs = [run for run in items if run is not None]
        add_run_row_label(first_run_ax, f'runs\n{row_runs[0] + 1}-{row_runs[-1] + 1}')
        row_first_axes.append(first_run_ax)
    return run_axes, row_first_axes, example_label_ax, collision_rows


def plot_spot_instability_map(ax, data):
    render_categorical_map(ax=ax, sdata=data['sdata'], bounds=data['bounds'], color='spot_instability_bin', palette=BIN_COLORS, groups=None, fill_alpha=1.0)
    ax.set_title('Instability Map', fontsize=CONTEXT_TITLE_FONTSIZE, fontweight='bold', pad=8)

def add_instability_legend_5x2(ax):
    ax.set_axis_off()
    handles = [Patch(facecolor=BIN_COLORS[label], edgecolor='none', label=label) for label in INSTABILITY_LABELS]
    order = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
    handles = [handles[i] for i in order]
    labels = [INSTABILITY_LABELS[i] for i in order]
    legend = ax.legend(handles=handles, labels=labels, title='', loc='center left', ncol=2, frameon=False, fontsize=LEGEND_FONTSIZE, handlelength=0.8, handleheight=0.8, handletextpad=0.5, columnspacing=1, labelspacing=0.4, borderaxespad=0.0)
    legend.get_title().set_fontweight('bold')
    try:
        legend._legend_box.align = 'left'
    except AttributeError:
        pass
    return legend

def plot_reference_annotation(ax, data):
    data['sdata'].tables['table'].obs['reference_annotation'] = pd.Categorical(data['annotation'].to_numpy(), categories=data['annotation_categories'], ordered=True)
    render_categorical_map(ax=ax, sdata=data['sdata'], bounds=data['bounds'], color='reference_annotation', palette=data['reference_palette'], groups=data['selected_reference_labels'], fill_alpha=1.0)
    ax.set_title('Reference\nAnnotation', fontsize=CONTEXT_TITLE_FONTSIZE, fontweight='bold', pad=8)

def add_reference_legend(ax, data):
    ax.set_axis_off()
    handles = []
    for label in data['selected_reference_labels']:
        clusters = data['reference_examples'].get(label, [])
        suffix = f" (clusters {','.join(clusters)})" if len(clusters) > 1 else ''
        handles.append(Line2D([], [], linestyle='none', marker='h', markersize=11,
                              markerfacecolor=data['reference_palette'][label],
                              markeredgecolor='none', label=label + suffix))
    return ax.legend(handles=handles, loc='center left', frameon=False,
                     fontsize=REFERENCE_LEGEND_FONTSIZE, handletextpad=0.5,
                     labelspacing=0.55, borderaxespad=0.0)

def plot_context_example_run(ax, data):
    plot_aligned_run(ax=ax, data=data, run_index=data['example_run'], record_collisions=False)
    ax.set_title('Example Run', fontsize=CONTEXT_TITLE_FONTSIZE, fontweight='bold', linespacing=0.95, pad=8)

def add_context_bottom_title(ax, text, y=-0.11):
    ax.text(0.5, y, text, transform=ax.transAxes, ha='center', va='top', fontsize=CONTEXT_TITLE_FONTSIZE, fontweight='bold', linespacing=0.95, clip_on=False)

def add_context_workflow_box(ax, text):
    ax.set_axis_off()
    ax.text(0.03, 0.95, text, transform=ax.transAxes, ha='left', va='top', fontsize=CONTEXT_WORKFLOW_FONTSIZE, linespacing=1.2, bbox=dict(boxstyle='round,pad=0.38', facecolor='white', edgecolor='black', linewidth=1.2, linestyle='--'))

def plot_context_binned_clusters(ax, data, fine_bin):
    cluster_bins = data['selected_matches'][['example_cluster', 'example_fine_bin']].drop_duplicates()
    selected_clusters = cluster_bins.loc[cluster_bins['example_fine_bin'].eq(fine_bin), 'example_cluster'].astype(str).tolist()
    example_labels = np.asarray(data['example_labels']).astype(str)
    display = np.where(np.isin(example_labels, selected_clusters), example_labels, 'Other')
    categories = ['Other'] + selected_clusters
    data['sdata'].tables['table'].obs['context_binned_cluster'] = pd.Categorical(display, categories=categories, ordered=True)
    palette = {'Other': OTHER_SPOTS_COLOR}
    for cluster in selected_clusters:
        palette[cluster] = data['cluster_colors'][cluster]
    render_categorical_map(ax=ax, sdata=data['sdata'], bounds=data['bounds'], color='context_binned_cluster', palette=palette, groups=selected_clusters, fill_alpha=0.98, show_histology=False)
    style_rounded_spatial_frame(ax, linewidth=CONTEXT_FRAME_LINEWIDTH, linestyle=(0, (3, 2)), rounding=0.025)
    ax.set_title('')
    handle = Patch(facecolor=BIN_COLORS[fine_bin], edgecolor='none')
    bin_fontsize = 9 if len(data['context_bins']) >= 4 else 10
    ax.legend(handles=[handle], labels=[fine_bin], loc='upper center', bbox_to_anchor=(0.5, -0.035), frameon=False, fontsize=bin_fontsize, handlelength=0.8, handletextpad=0.35, borderaxespad=0.0)


def plot_fig3_context(fig, parent_spec, data):
    wrapper = parent_spec.subgridspec(3, 3, height_ratios=[0.04, 0.92, 0.04],
                                     width_ratios=[0.03, 0.94, 0.03], hspace=0, wspace=0)
    grid = wrapper[1, 1].subgridspec(2, 5, width_ratios=[1, 1.35, 1.45, 1.45, 1.35],
                                    height_ratios=[0.82, 1.18], wspace=0.18, hspace=0.28)
    instability_ax = fig.add_subplot(grid[0, 0])
    plot_spot_instability_map(instability_ax, data)
    instability_ax.set_title('')
    style_rounded_spatial_frame(instability_ax, linewidth=CONTEXT_FRAME_LINEWIDTH)
    add_context_bottom_title(instability_ax, 'Instability\nMap')
    instability_legend_ax = fig.add_subplot(grid[0, 1])
    add_instability_legend_5x2(instability_legend_ax)
    example_ax = fig.add_subplot(grid[0, 2])
    plot_context_example_run(example_ax, data)
    example_ax.set_title('')
    style_rounded_spatial_frame(example_ax, linewidth=CONTEXT_FRAME_LINEWIDTH, linestyle=(0, (3, 2)))
    add_context_bottom_title(example_ax, 'Example\nRun')
    top_box_grid = grid[0, 3:5].subgridspec(1, 2, width_ratios=[0.45, 0.55], wspace=0)
    workflow_top_ax = fig.add_subplot(top_box_grid[0, 1])
    add_context_workflow_box(workflow_top_ax,
        'Bin cluster in example run\n'
        'by mean instability score\n\n'
        'Find best match in ref.\n'
        'for accuracy metrics.\n\n'
        'Find best match in\n'
        'other runs for DGEA')
    reference_ax = fig.add_subplot(grid[1, 0])
    plot_reference_annotation(reference_ax, data)
    reference_ax.set_title('')
    style_rounded_spatial_frame(reference_ax, linewidth=CONTEXT_FRAME_LINEWIDTH)
    add_context_bottom_title(reference_ax, 'Reference\nAnnotation')
    reference_legend_ax = fig.add_subplot(grid[1, 1])
    add_reference_legend(reference_legend_ax, data)
    bins = data['context_bins']
    binned_grid = grid[1, 2:4].subgridspec(2, len(bins), height_ratios=[1, 0.18],
                                         wspace=0.22 if len(bins) >= 4 else 0.14, hspace=0)
    binned_axes = []
    for index, fine_bin in enumerate(bins):
        ax = fig.add_subplot(binned_grid[0, index])
        plot_context_binned_clusters(ax, data, fine_bin)
        binned_axes.append(ax)
    binned_title_ax = fig.add_subplot(binned_grid[1, :])
    binned_title_ax.set_axis_off()
    binned_title_ax.text(0.5, 0, 'Binned Clusters', transform=binned_title_ax.transAxes,
                         ha='center', va='bottom', fontsize=CONTEXT_TITLE_FONTSIZE, fontweight='bold')
    workflow_bottom_ax = fig.add_subplot(grid[1, 4])
    target_text = 'other runs' if FIG3C_COMPARISON == 'cross_run' else 'the reference'
    add_context_workflow_box(workflow_bottom_ax,
        'Perform DGEA for all\n'
        'clusters vs rest in the\n'
        'example run\n\n'
        'Repeat for matched clusters\n'
        'in ' + target_text + ' & compare\n'
        'results to example run')
    return dict(instability_ax=instability_ax, instability_legend_ax=instability_legend_ax,
                example_ax=example_ax, reference_ax=reference_ax,
                reference_legend_ax=reference_legend_ax, binned_axes=binned_axes,
                binned_title_ax=binned_title_ax, workflow_top_ax=workflow_top_ax,
                workflow_bottom_ax=workflow_bottom_ax)


def load_fig3b_method_matches(method):
    frame = dgea_data.matches().copy()
    return frame.loc[frame['method'].eq(method)].copy()

def summarize_fig3b_clusters(matches):
    methods = matches['method'].unique()
    if len(methods) != 1:
        raise ValueError('Expected one method')
    frame = dgea_data.cluster_summary().copy()
    summary = frame.loc[frame['method'].eq(methods[0])].copy()
    keys = ['sample_id', 'method', 'example_run_idx', 'example_cluster']
    grouped = matches.groupby(keys, sort=False)
    if not grouped.size().eq(N_RUNS - 1).all():
        raise ValueError('Each example cluster must have 49 other-run matches')
    actual = grouped['best_jaccard'].mean()
    expected = summary.set_index(keys)['best_jaccard_mean'].reindex(actual.index)
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-14):
        raise ValueError('Cluster means disagree with run-level matches')
    return summary

def fig3b_correlation_results(cluster_summary):
    valid = cluster_summary[['example_mean_instability', 'best_jaccard_mean']].dropna()
    if len(valid) < 3 or (valid.nunique() < 2).any():
        return np.nan, np.nan, np.nan, np.nan, len(valid)
    x, y = valid['example_mean_instability'], valid['best_jaccard_mean']
    sr, pr = spearmanr(x, y), pearsonr(x, y)
    return float(sr.statistic), float(sr.pvalue), float(pr.statistic), float(pr.pvalue), len(valid)

def style_fig3b_boxplot(boxplot, colors):
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
        box.set_alpha(0.82)
        box.set_edgecolor('black')
        box.set_linewidth(1.1)
    for key in ['whiskers', 'caps']:
        for line in boxplot[key]:
            line.set_color('black')
            line.set_linewidth(1.1)
    for median in boxplot['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)

def plot_fig3b(fig, parent_spec, matches, cluster_summary):
    populated_bins = [fine_bin for fine_bin in INSTABILITY_LABELS if matches.loc[matches['example_fine_bin'].eq(fine_bin), 'best_jaccard'].notna().any()]
    if not populated_bins:
        raise ValueError('No populated instability bins are available for Figure 3D.')
    wrapper = parent_spec.subgridspec(nrows=2, ncols=1, height_ratios=[0.18, 0.82], hspace=0.0)
    grid = wrapper[1, 0].subgridspec(nrows=2, ncols=2, height_ratios=FIG3B_HEIGHT_RATIOS, width_ratios=[1.0, 1.0], hspace=0.0, wspace=FIG3B_INNER_WSPACE)
    ax_scatter = fig.add_subplot(grid[0, 0])
    ax_box = fig.add_subplot(grid[0, 1])
    box_data = [matches.loc[matches['example_fine_bin'].eq(fine_bin), 'best_jaccard'].dropna().to_numpy(float) for fine_bin in populated_bins]
    positions = np.arange(1, len(populated_bins) + 1, dtype=float)
    box_positions = positions - 0.14
    violin_positions = positions + 0.1
    colors = [BIN_COLORS[bin_label] for bin_label in populated_bins]
    boxplot = ax_box.boxplot(box_data, positions=box_positions, widths=0.28, patch_artist=True, showfliers=False, whis=1.5, manage_ticks=False)
    style_fig3b_boxplot(boxplot, colors)
    for values, color, violin_position in zip(box_data, colors, violin_positions):
        if len(values) >= 2 and np.ptp(values) > 0:
            violin = ax_box.violinplot([values], positions=[violin_position], widths=0.55, showmeans=False, showmedians=False, showextrema=False, bw_method='scott')
            body = violin['bodies'][0]
            vertices = body.get_paths()[0].vertices

            vertices[:, 0] = np.maximum(vertices[:, 0], violin_position)
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_linewidth(1.1)
            body.set_alpha(0.6)
            body.set_zorder(1)
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(populated_bins, rotation=45, ha='right', rotation_mode='anchor')
    ax_box.set_xlim(0.45, len(populated_bins) + 0.65)
    ax_box.set_ylim(0, 1)
    ax_box.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_box.set_yticklabels([])
    ax_box.tick_params(axis='both', labelsize=FIG3B_TICK_FONTSIZE)
    ax_box.set_title('(distribution)', fontweight='bold', fontsize=FIG3B_TITLE_FONTSIZE, pad=6)
    ax_box.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.4)
    ax_box.spines['top'].set_visible(False)
    ax_box.spines['right'].set_visible(False)
    cluster_counts = []
    rng = np.random.default_rng(FIG3B_RANDOM_SEED)
    for fine_bin in populated_bins:
        sub = cluster_summary[cluster_summary['example_fine_bin'].eq(fine_bin)]
        cluster_counts.append(int(len(sub)))
        if sub.empty:
            continue
        x_values = sub['example_mean_instability'].to_numpy(float)
        y_values = sub['best_jaccard_mean'].to_numpy(float)
        jitter = rng.normal(0, 0.0025, size=len(sub))
        ax_scatter.scatter(np.clip(x_values + jitter, 0, FIG3B_SCATTER_XMAX), y_values, s=FIG3B_SCATTER_SIZE, color=BIN_COLORS[fine_bin], edgecolor='black', linewidth=0.35, alpha=0.85)
    fit_data = cluster_summary[['example_mean_instability', 'best_jaccard_mean']].dropna()
    if len(fit_data) >= 2 and fit_data['example_mean_instability'].nunique() >= 2:
        x_fit = fit_data['example_mean_instability'].to_numpy(float)
        y_fit = fit_data['best_jaccard_mean'].to_numpy(float)
        slope, intercept = np.polyfit(x_fit, y_fit, deg=1)
        x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
        y_line = slope * x_line + intercept
        ax_scatter.plot(x_line, y_line, color='black', linestyle='--', linewidth=1.75, zorder=4)
    rho, spearman_p, pearson_r, pearson_p, n_clusters = fig3b_correlation_results(cluster_summary)
    print(f'[Figure 3D] Spearman rho={rho:.3f}, p={spearman_p:.3g}; Pearson r={pearson_r:.3f}, p={pearson_p:.3g}; n={n_clusters} clusters')
    ax_scatter.set_xlim(0, FIG3B_SCATTER_XMAX)
    ax_scatter.set_xticks(np.arange(0, FIG3B_SCATTER_XMAX + 1e-9, 0.1))
    ax_scatter.set_ylim(0, 1)
    ax_scatter.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_scatter.tick_params(axis='both', labelsize=FIG3B_TICK_FONTSIZE)
    n_text = ','.join((str(value) for value in cluster_counts))
    ax_scatter.set_xlabel(f'Cluster Instability - Example Run\n($N$ = {n_text})', fontsize=FIG3B_AXIS_LABEL_FONTSIZE, linespacing=1.05, labelpad=4)
    ax_scatter.set_ylabel('Jaccard w.r.t. Match', fontsize=FIG3B_AXIS_LABEL_FONTSIZE)
    ax_scatter.set_title('Cross-run Consistency (mean)', fontweight='bold', fontsize=FIG3B_TITLE_FONTSIZE, pad=6)
    ax_scatter.grid(linestyle=':', linewidth=0.6, alpha=0.4)
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['right'].set_visible(False)
    stats = {'method': FIG3_ANALYSIS_SCOPE, 'methods': ','.join(FIG3_ANALYSIS_METHODS),
             'n_methods': cluster_summary['method'].nunique(),
             'n_samples': cluster_summary['sample_id'].nunique(),
             'n_sample_method_pairs': len(cluster_summary[['sample_id', 'method']].drop_duplicates()),
             'scatter_xmax': FIG3B_SCATTER_XMAX,
             'n_clusters_above_scatter_xmax': int(cluster_summary['example_mean_instability'].gt(FIG3B_SCATTER_XMAX).sum()),
             'n_example_clusters': n_clusters, 'spearman_rho': rho, 'spearman_pvalue': spearman_p, 'pearson_r': pearson_r, 'pearson_pvalue': pearson_p, 'populated_bins': populated_bins, 'cluster_counts': cluster_counts}
    return (np.array([ax_scatter, ax_box], dtype=object), stats)


def load_fig3c_method(method):
    frame = dgea_data.consistency().copy()
    return frame.loc[frame['method'].eq(method)].copy()

def fig3c_finite_values(df, fine_bin, column):
    values = df.loc[df['example_fine_bin'].eq(fine_bin), column].to_numpy(float)
    return values[np.isfinite(values)]

def fig3c_common_bins(pooled, top_k):
    subset = pooled.loc[pooled['top_k'].eq(top_k)]
    return [label for label in INSTABILITY_LABELS
            if any(fig3c_finite_values(subset, label, metric['column']).size
                   for metric in FIG3C_METRICS.values())]

def style_fig3c_boxplot(boxplot, colors):
    for box, color in zip(boxplot['boxes'], colors):
        box.set(facecolor=color, alpha=0.82, edgecolor='black', linewidth=1.1)
    for key in ['whiskers', 'caps']:
        for line in boxplot[key]:
            line.set(color='black', linewidth=1.1)
    for median in boxplot['medians']:
        median.set(color='black', linewidth=1.5)
    for flier in boxplot['fliers']:
        flier.set(color='black', markersize=4)

def draw_fig3c_panel(ax, pooled, top_k, metric_name, bins, show_ylabel, show_title, show_x):
    metric = FIG3C_METRICS[metric_name]
    sub = pooled.loc[pooled['top_k'].eq(top_k)]
    positions = np.arange(1, len(bins) + 1, dtype=float)
    arrays = [fig3c_finite_values(sub, fine_bin, metric['column']) for fine_bin in bins]
    nonempty = [(position, values, BIN_COLORS[fine_bin])
                for position, values, fine_bin in zip(positions, arrays, bins) if len(values)]
    if nonempty:
        boxplot = ax.boxplot([values for _, values, _ in nonempty],
                             positions=[position for position, _, _ in nonempty], widths=0.64,
                             patch_artist=True, showfliers=True, manage_ticks=False, whis=1.5)
        style_fig3c_boxplot(boxplot, [color for _, _, color in nonempty])
    else:
        ax.text(0.5, 0.5, 'No valid comparisons', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlim(0.25, max(len(bins), 1) + 0.75)
    if metric_name == 'jaccard':
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        if not show_ylabel:
            ax.set_yticklabels([])
    else:

        maximum = max((float(values.max()) for _, values, _ in nonempty), default=0)
        ax.set_ylim(0, max(1.0, 1.08 * maximum))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.set_xticks(positions)
    if show_x:
        ax.set_xticklabels(bins, rotation=45, ha='right', rotation_mode='anchor')
    else:
        ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='both', labelsize=FIG3C_TICK_FONTSIZE)
    if show_title:
        ax.set_title(f'Top-{top_k} DEGs', fontsize=FIG3C_TITLE_FONTSIZE, fontweight='bold', pad=4)
    if show_ylabel:
        ax.set_ylabel(metric['ylabel'], fontsize=FIG3C_AXIS_LABEL_FONTSIZE)
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return [len(values) for values in arrays]

def fig3c_shared_xlabel_text(pooled):

    return f'Instability Bins of Example Clusters ({analysis_method_label()})'

def plot_fig3c(fig, parent_spec, pooled):
    wrapper = parent_spec.subgridspec(nrows=2, ncols=1, height_ratios=[0.1, 0.9], hspace=0.0)
    grid = wrapper[1, 0].subgridspec(nrows=5, ncols=len(FIG3C_TOP_KS), height_ratios=FIG3C_HEIGHT_RATIOS, hspace=FIG3C_INNER_HSPACE, wspace=FIG3C_INNER_WSPACE)
    top_axes = []
    bottom_axes = []
    for index, top_k in enumerate(FIG3C_TOP_KS):
        bins = fig3c_common_bins(pooled, top_k)
        bottom_ax = fig.add_subplot(grid[2, index])
        top_ax = fig.add_subplot(grid[0, index], sharex=bottom_ax)
        draw_fig3c_panel(ax=bottom_ax, pooled=pooled, top_k=top_k, metric_name='abs_rank_diff', bins=bins, show_ylabel=index == 0, show_title=False, show_x=True)
        draw_fig3c_panel(ax=top_ax, pooled=pooled, top_k=top_k, metric_name='jaccard', bins=bins, show_ylabel=index == 0, show_title=True, show_x=False)
        top_axes.append(top_ax)
        bottom_axes.append(bottom_ax)
    top_axes[0].yaxis.set_label_coords(-0.25, 0.5)
    bottom_axes[0].yaxis.set_label_coords(-0.25, 0.5)
    label_ax = fig.add_subplot(grid[4, :])
    label_ax.set_axis_off()
    shared_xlabel = fig3c_shared_xlabel_text(pooled)
    label_ax.text(0.5, 0.18, shared_xlabel, transform=label_ax.transAxes, ha='center', va='center', fontsize=FIG3C_SHARED_XLABEL_FONTSIZE)
    return (np.asarray([top_axes, bottom_axes], dtype=object), label_ax, shared_xlabel)


def load_fig3d_clusterwise_metrics(method):
    frame = dgea_data.reference_quality().copy()
    return frame.loc[frame['method'].eq(method)].copy()

def fig3d_dataset_mask(df, dataset):
    tags = df['dataset-sample'].astype(str)
    if dataset == 'dlpfc':
        return tags.str.startswith('dlpfc_', na=False)
    return tags.eq(dataset)

def draw_fig3d_scatter(ax, stats_ax, df, metric_column, title, show_ylabel):
    x_all = df[FIG3D_XCOL].to_numpy(float)
    y_all = df[metric_column].to_numpy(float)
    valid = np.isfinite(x_all) & np.isfinite(y_all)
    xx = x_all[valid]
    yy = y_all[valid]
    r = r_sq = np.nan
    if len(xx) >= 2 and np.ptp(xx) > 0 and np.ptp(yy) > 0:
        r, _ = pearsonr(xx, yy)
        r_sq = r ** 2
        slope, intercept = np.polyfit(xx, yy, 1)
        fit_x = np.linspace(0, FIG3D_XMAX, 200)
        ax.plot(fit_x, slope * fit_x + intercept, color='black', linestyle='--', linewidth=2, zorder=4)
    print(f'[Figure 3C] {title}: r={r:.6f}, r2={r_sq:.6f}')
    datasets_to_plot = ['dlpfc', 'mouse_brain_cerebellum', 'Visium_HD_Human_Colon_Cancer_cropped_square', 'human_breast_cancer', 'mouse_brain', 'coad_ffpe', 'ov_ffpe']
    for dataset in datasets_to_plot:
        zdf = df.loc[fig3d_dataset_mask(df, dataset)]
        zx = zdf[FIG3D_XCOL].to_numpy(float)
        zy = zdf[metric_column].to_numpy(float)
        point_valid = np.isfinite(zx) & np.isfinite(zy)
        zx = zx[point_valid]
        zy = zy[point_valid]
        if zx.size == 0:
            continue
        color_index = FIG3D_DATASET_COLOR_INDEX[dataset]
        alpha = FIG3D_DLPFC_ALPHA if dataset == 'dlpfc' else FIG3D_NON_DLPFC_ALPHA
        ax.scatter(zx, zy, color=TABLEAU_20[color_index], s=FIG3D_SCATTER_SIZE, alpha=alpha, edgecolors='none', zorder=3)
    ax.set_title(title, fontsize=FIG3D_TITLE_FONTSIZE, fontweight='bold', pad=4)
    ax.set_xlabel('Cluster Instability', fontsize=FIG3D_AXIS_LABEL_FONTSIZE, labelpad=1)
    if show_ylabel:
        ax.set_ylabel('Quality w.r.t. Ref.', fontsize=FIG3D_AXIS_LABEL_FONTSIZE)
    ax.set_xlim(0, FIG3D_XMAX)
    ax.set_xticks(np.arange(0, FIG3D_XMAX + 1e-9, 0.2))
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    if not show_ylabel:
        ax.set_yticklabels([])
    ax.tick_params(axis='both', labelsize=FIG3D_TICK_FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    stats_ax.set_axis_off()
    stats_ax.text(0.5, 0.82, f'$r = {r:.2f}$\n$r^2 = {r_sq:.2f}$', transform=stats_ax.transAxes, ha='center', va='top', fontsize=FIG3D_STATS_FONTSIZE, linespacing=1.0, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='gray'))
    return {'analysis_scope': FIG3_ANALYSIS_SCOPE, 'methods': ','.join(FIG3_ANALYSIS_METHODS),
            'n_methods': int(df['method'].nunique()), 'n_samples': int(df['dataset-sample'].nunique()),
            'metric': metric_column, 'title': title, 'pearson_r': float(r),
            'r_squared': float(r_sq), 'n_clusters': int(len(xx)), 'x_axis_max': FIG3D_XMAX}

def add_fig3d_sample_legend(ax):
    ax.set_axis_off()
    labels = ['MB', 'MBC', 'HBC', 'DLPFC', 'CRC', 'OV', 'COAD']
    datasets = ['mouse_brain', 'mouse_brain_cerebellum', 'human_breast_cancer', 'dlpfc', 'Visium_HD_Human_Colon_Cancer_cropped_square', 'ov_ffpe', 'coad_ffpe']
    colors = [TABLEAU_20[FIG3D_DATASET_COLOR_INDEX[dataset]] for dataset in datasets]
    legend_marker_size = np.sqrt(140)
    handles = [Line2D([], [], linestyle='none', marker='o', markersize=legend_marker_size, markerfacecolor=color, markeredgecolor='none', label=label) for label, color in zip(labels, colors)]
    ax.text(0.03, 0.93, 'Sample', transform=ax.transAxes, fontsize=16, fontweight='bold', ha='left', va='top')
    legend = ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(0.0, 0.83), bbox_transform=ax.transAxes, ncol=2, frameon=False, fontsize=16, handlelength=1.4, handletextpad=0.5, columnspacing=1, labelspacing=0.4, borderaxespad=0.0)
    return legend

def plot_fig3d(fig, parent_spec, df):
    grid = parent_spec.subgridspec(nrows=3, ncols=4, height_ratios=FIG3D_HEIGHT_RATIOS, hspace=FIG3D_INNER_HSPACE, wspace=FIG3D_INNER_WSPACE)
    scatter_axes = []
    stats_axes = []
    metric_stats = []
    for index, (metric_column, title) in enumerate(FIG3D_METRICS):
        ax = fig.add_subplot(grid[0, index])
        stats_ax = fig.add_subplot(grid[2, index])
        stat = draw_fig3d_scatter(ax=ax, stats_ax=stats_ax, df=df, metric_column=metric_column, title=title, show_ylabel=index == 0)
        scatter_axes.append(ax)
        stats_axes.append(stats_ax)
        metric_stats.append(stat)
    legend_ax = fig.add_subplot(grid[0, 3])
    legend = add_fig3d_sample_legend(legend_ax)
    legend_lower_ax = fig.add_subplot(grid[2, 3])
    legend_lower_ax.set_axis_off()
    return {'scatter_axes': np.asarray(scatter_axes, dtype=object), 'stats_axes': np.asarray(stats_axes, dtype=object), 'legend_ax': legend_ax, 'legend_lower_ax': legend_lower_ax, 'legend': legend, 'metric_stats': metric_stats}

def make_blank_panel(fig, spec):
    ax = fig.add_subplot(spec)
    ax.set_xticks([])
    ax.set_yticks([])
    if SHOW_PANEL_GUIDES:
        ax.patch.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)
            spine.set_linestyle('--')
    else:
        ax.set_axis_off()
    return ax

def add_aligned_panel_letters(fig, run_ax, cross_run_ax, reference_ax, dgea_ax):
    fig.canvas.draw()
    left_x = run_ax.get_position().x0 - PANEL_LETTER_XPAD
    right_x = cross_run_ax.get_position().x0 - PANEL_LETTER_RIGHT_XPAD
    top_y = run_ax.get_position().y1 + PANEL_LETTER_A_YPAD
    context_y = cross_run_ax.get_position().y1 + PANEL_LETTER_YPAD
    reference_y = reference_ax.get_position().y1 + PANEL_LETTER_YPAD
    dgea_y = dgea_ax.get_position().y1 + PANEL_LETTER_YPAD
    positions = {'A': (left_x, top_y), 'B': (left_x, context_y),
                 'C': (left_x, reference_y), 'D': (right_x, context_y),
                 'E': (right_x, dgea_y)}
    return {letter: fig.text(x, y, letter, transform=fig.transFigure, fontsize=28,
                             fontweight='bold', ha='left', va='top')
            for letter, (x, y) in positions.items()}


def create_figure3(data, fig3b_matches, fig3b_cluster_summary, fig3c_pooled, fig3d_df):
    fig = plt.figure(figsize=FIGSIZE)
    outer = fig.add_gridspec(nrows=5, ncols=2, height_ratios=FIG3_HEIGHT_RATIOS, hspace=FIG3_OUTER_HSPACE, wspace=FIG3_OUTER_WSPACE, left=FIG3_LEFT, right=FIG3_RIGHT, top=FIG3_TOP, bottom=FIG3_BOTTOM)
    run_axes, row_first_axes, example_label_ax, collision_rows = plot_fig3a(fig, outer[0:2, :], data)
    context_axes = plot_fig3_context(fig, outer[2:4, 0], data)

    match_axes_size_to_reference(
        fig, row_first_axes[0],
        [context_axes['instability_ax'], context_axes['reference_ax'],
         context_axes['example_ax'], *context_axes['binned_axes']], scale=1.0)

    EXAMPLE_CONTEXT_XSHIFT = 0.05

    pos = context_axes['example_ax'].get_position()
    context_axes['example_ax'].set_position([pos.x0 + EXAMPLE_CONTEXT_XSHIFT, pos.y0, pos.width, pos.height])


    CONTEXT_BOTTOM_ROW_SHIFT = 0.022
    bottom_context_axes = [context_axes['reference_ax'], context_axes['reference_legend_ax'], *context_axes['binned_axes'], context_axes['binned_title_ax'], context_axes['workflow_bottom_ax']]
    for ax in bottom_context_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + CONTEXT_BOTTOM_ROW_SHIFT, pos.width, pos.height])


    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top_text = context_axes['workflow_top_ax'].texts[0]
    bottom_text = context_axes['workflow_bottom_ax'].texts[0]
    top_box = top_text.get_window_extent(renderer=renderer)
    bottom_box = bottom_text.get_window_extent(renderer=renderer)

    dx = (bottom_box.x1 - top_box.x1) / fig.bbox.width
    pos = context_axes['workflow_top_ax'].get_position()
    context_axes['workflow_top_ax'].set_position([pos.x0 + dx,pos.y0,pos.width,pos.height])


    panel_b_content_axes = [
        context_axes['instability_ax'], context_axes['instability_legend_ax'],
        context_axes['example_ax'], context_axes['reference_ax'],
        context_axes['reference_legend_ax'], *context_axes['binned_axes'],
        context_axes['binned_title_ax'], context_axes['workflow_top_ax'],
        context_axes['workflow_bottom_ax'],
    ]
    for ax in panel_b_content_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0 + PANEL_B_XSHIFT, pos.y0, pos.width, pos.height])

    b_axes, b_stats = plot_fig3b(fig, outer[2:3, 1], fig3b_matches, fig3b_cluster_summary)
    c_axes, c_label_ax, c_shared_xlabel = plot_fig3c(fig, outer[3:5, 1], fig3c_pooled)
    d_data = plot_fig3d(fig, outer[4:5, 0], fig3d_df)
    panel_letters = add_aligned_panel_letters(
        fig, row_first_axes[0], b_axes[0], d_data['scatter_axes'][0], c_axes[0, 0])
    return {'fig': fig, 'outer': outer, 'panel_letters': panel_letters, 'run_axes': run_axes, 'row_first_axes': row_first_axes, 'example_label_ax': example_label_ax, 'context_axes': context_axes, 'b_axes': b_axes, 'b_stats': b_stats, 'c_axes': c_axes, 'c_label_ax': c_label_ax, 'c_shared_xlabel': c_shared_xlabel, 'd_data': d_data, 'collision_rows': collision_rows}


def save_figure3(figure_data):
    fig = figure_data['fig']
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for extension in SAVE_FORMATS:
            path = OUT_STEM.with_suffix('.' + extension)
            fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.03,
                        facecolor='white')
            print(f'Saved {path}')
    finally:
        plt.close(fig)


def configure_analysis_methods(requested):
    global FIG3_ANALYSIS_METHODS, FIG3_ANALYSIS_SCOPE, OUT_DIR, OUT_STEM
    requested = list(requested)
    if requested == ['all']:
        requested = METHODS.copy()
    if not requested or len(requested) != len(set(requested)):
        raise ValueError('Select at least one method, with no duplicates')
    unsupported = set(requested) - set(METHODS)
    if unsupported:
        raise ValueError('Unsupported methods (use all by itself): ' +
                         ', '.join(sorted(unsupported)))
    FIG3_ANALYSIS_METHODS = [method for method in METHODS if method in requested]
    if FIG3_ANALYSIS_METHODS == METHODS:
        FIG3_ANALYSIS_SCOPE = 'all_methods'
    elif len(FIG3_ANALYSIS_METHODS) == 1:
        FIG3_ANALYSIS_SCOPE = FIG3_ANALYSIS_METHODS[0]
    else:
        FIG3_ANALYSIS_SCOPE = 'methods_' + '_'.join(FIG3_ANALYSIS_METHODS)
    OUT_DIR = OUTPUT_ROOT / 'fig3' / FIG3_ANALYSIS_SCOPE
    OUT_STEM = OUT_DIR / 'fig3'


def analysis_method_label():
    if FIG3_ANALYSIS_METHODS == METHODS:
        return 'all methods'
    if len(FIG3_ANALYSIS_METHODS) == 1:
        return METHOD_LABELS[FIG3_ANALYSIS_METHODS[0]]
    return f'{len(FIG3_ANALYSIS_METHODS)} methods'


def load_pooled_analysis_tables():
    match_parts, summary_parts, dgea_parts, reference_parts, coverage = [], [], [], [], []
    for method in FIG3_ANALYSIS_METHODS:
        matches = load_fig3b_method_matches(method)
        summary = summarize_fig3b_clusters(matches)
        dgea = load_fig3c_method(method)
        reference = load_fig3d_clusterwise_metrics(method)
        keys = ['sample_id', 'method', 'example_run_idx', 'example_cluster']
        summary_ids = pd.MultiIndex.from_frame(summary[keys])
        reference_ids = pd.MultiIndex.from_frame(
            reference.rename(columns={'dataset-sample': 'sample_id'})[keys])
        if reference_ids.has_duplicates or set(reference_ids) != set(summary_ids):
            raise ValueError(f'{method}: reference and cross-run tables cover different clusters')
        for top_k in FIG3C_TOP_KS:
            dgea_ids = pd.MultiIndex.from_frame(dgea.loc[dgea['top_k'].eq(top_k), keys])
            if dgea_ids.has_duplicates or set(dgea_ids) != set(summary_ids):
                raise ValueError(f'{method}: Top-{top_k} DGEA covers different clusters')
        if summary['sample_id'].nunique() != len(FIG3B_DATASET_SAMPLES):
            raise ValueError(f'{method}: incomplete sample coverage')
        match_parts.append(matches)
        summary_parts.append(summary)
        dgea_parts.append(dgea)
        reference_parts.append(reference)
        coverage.append({
            'method': method, 'n_samples': int(summary['sample_id'].nunique()),
            'n_example_clusters': len(summary), 'n_cross_run_matches': len(matches),
            'n_reference_matches': len(reference), 'n_dgea_cluster_topk_rows': len(dgea),
            'max_example_instability': float(summary['example_mean_instability'].max()),
        })
        print(f'[C/D/E] {method}: {len(summary)} clusters, '
              f'{len(matches)} cross-run matches, {len(dgea)} cluster/Top-K summaries', flush=True)
    tables = tuple(pd.concat(parts, ignore_index=True)
                   for parts in (match_parts, summary_parts, dgea_parts, reference_parts))
    return (*tables, pd.DataFrame(coverage))


def main(argv=None):
    global DPI, OUT_DIR, OUT_STEM, SAVE_FORMATS
    parser = argparse.ArgumentParser(description='Plot Figure 3 from compact saved results.')
    parser.add_argument('--methods', nargs='+', default=['all'], choices=['all', *METHODS])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dpi', type=int, default=DPI)
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf'], default=SAVE_FORMATS)
    args = parser.parse_args(argv)
    configure_analysis_methods(args.methods)
    if args.dpi <= 0:
        parser.error('--dpi must be positive')
    DPI, SAVE_FORMATS = args.dpi, args.formats
    if args.output_dir is not None:
        OUT_DIR = args.output_dir
        OUT_STEM = OUT_DIR / 'fig3'
    matches, summary, consistency, reference, coverage = load_pooled_analysis_tables()
    data = prepare_figure3_data(EXAMPLE_DATASET, EXAMPLE_SAMPLE, FIG3_METHOD,
                               SELECTED_EXAMPLE_CLUSTERS)
    figure_data = create_figure3(data, matches, summary, consistency, reference)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(OUT_DIR / 'coverage.csv', index=False)
    pd.DataFrame(figure_data['d_data']['metric_stats']).to_csv(
        OUT_DIR / 'reference_correlations.csv', index=False)
    save_figure3(figure_data)


if __name__ == '__main__':
    main()
