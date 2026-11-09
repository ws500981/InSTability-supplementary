#!/usr/bin/env python3
'Supplementary Figure S25: cluster metrics versus instability.'
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from common import DATA_DIR, OUTPUT_ROOT
from cluster_metrics import load_cluster_metrics

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
tableau_20 = plt.cm.tab20.colors

ROOT = Path(__file__).resolve().parent
INPUT_CSV = DATA_DIR / 'metrics' / 'clusterwise_metrics.csv'
nruns = 50
NROWS, NCOLS = 2, 5
FIGSIZE = (12 * 5 / 4, 6)
FIG_DPI = 600
X_COLUMN = 'clusterwise mean instability'


mthds = ['leiden', 'louvain', 'bayesspace', 'graphst', 'graphst_mclust',
         'stagate', 'stagate_mclust', 'spicemix', 'sedr', 'sedr_mclust']
mnames = ['Leiden', 'Louvain', 'BayesSpace', 'GraphST', 'GraphST (mclust)',
          'STAGATE', 'STAGATE (mclust)', 'SpiceMix', 'SEDR', 'SEDR (mclust)']
SAMPLE_TAGS = [f'dlpfc_{s}' for s in (
    '151507', '151508', '151509', '151510', '151669', '151670',
    '151671', '151672', '151673', '151674', '151675', '151676',
)] + ['mouse_brain', 'mouse_brain_cerebellum', 'human_breast_cancer',
     'Visium_HD_Human_Colon_Cancer_cropped_square', 'ov_ffpe', 'coad_ffpe']

OUTPUT_DIR = OUTPUT_ROOT / 'supp_s25'
OUTPUT_STEM = 's25_cluster_recall_wrt_ref_vs_instability'
metric = 'clusterwise recall'
HAS_CORRELATION = False
Y_MAX = 1.0
METRIC_DEFINITION = 'Overlap divided by the size of the matched annotation group; use the maximum-Jaccard annotation match.'
STYLE_SOURCE = 'supp_s25.py'


datas = [
    "dlpfc",
    "mouse_brain",
    "mouse_brain_cerebellum",
    "human_breast_cancer",
    "Visium_HD_Human_Colon_Cancer_cropped_square",
    "ov_ffpe",
    "coad_ffpe",
]

dnames = [
    "DLPFC",
    "MB",
    "MBC",
    "HBC",
    "CRC",
    "OV",
    "COAD",
]

colors = [
    tableau_20[14],
    tableau_20[8],
    tableau_20[0],
    tableau_20[12],
    tableau_20[2],
    tableau_20[18],
    tableau_20[16],
]

def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_data():
    'Select all 180 requested sample-method groups at exactly 50 runs.'
    frame = load_cluster_metrics(iterations=[nruns], methods=mthds)
    required = {'dataset-sample', 'method', 'iterations', 'reference_mode',
                'reference_run', 'reference_seed', 'expression_source',
                'cluster', 'n_spots', X_COLUMN, metric}
    require(required.issubset(frame.columns),
            f'{INPUT_CSV}: missing columns {sorted(required - set(frame.columns))}')
    frame['iterations'] = pd.to_numeric(frame['iterations'], errors='raise')
    frame = frame.loc[frame['iterations'].eq(nruns)
                      & frame['dataset-sample'].isin(SAMPLE_TAGS)
                      & frame['method'].isin(mthds)].copy()
    require(not frame.empty, f'{INPUT_CSV}: no requested 50-run rows')
    require(not frame[list(required - {metric})].isna().any().any(),
            f'{INPUT_CSV}: missing required cluster metadata')
    require(frame['reference_mode'].eq('example_50').all(),
            f'{INPUT_CSV}: expected the fixed example_50 clustering')
    require(frame['expression_source'].isin(['layers/counts', 'raw/X']).all(),
            f'{INPUT_CSV}: unexpected expression source')
    keys = ['dataset-sample', 'method', 'iterations', 'cluster']
    require(not frame.duplicated(keys).any(), f'{INPUT_CSV}: duplicate cluster rows')
    observed = set(frame[['dataset-sample', 'method']].itertuples(index=False, name=None))
    expected = {(tag, method) for tag in SAMPLE_TAGS for method in mthds}
    require(observed == expected,
            f'{INPUT_CSV}: missing sample-method groups {sorted(expected - observed)}')
    for column in (X_COLUMN, metric, 'reference_run', 'reference_seed', 'n_spots'):
        values = pd.to_numeric(frame[column], errors='raise')
        require(not np.isinf(values).any(), f'{INPUT_CSV}: infinite {column}')
        if column != metric:
            require(np.isfinite(values).all(), f'{INPUT_CSV}: nonfinite {column}')
        frame[column] = values
    for column, low, high in (('reference_run', 0, 49), ('reference_seed', 42, 91),
                              ('n_spots', 1, np.inf)):
        values = frame[column]
        require(values.between(low, high).all() and values.mod(1).eq(0).all(),
                f'{INPUT_CSV}: invalid integer {column}')
        frame[column] = values.astype(np.int64)
    require(frame['reference_seed'].eq(frame['reference_run'] + 42).all(),
            f'{INPUT_CSV}: reference seeds and run indices disagree')
    require(frame[X_COLUMN].ge(0).all(), f'{INPUT_CSV}: negative instability')
    unit_keys = ['dataset-sample', 'method']
    require(frame.groupby(unit_keys)['reference_run'].nunique().eq(1).all(),
            f'{INPUT_CSV}: multiple example runs within a sample-method group')
    spot_totals = frame.groupby(unit_keys)['n_spots'].sum()
    require(spot_totals.groupby(level=0).nunique().eq(1).all(),
            f'{INPUT_CSV}: cluster spot totals disagree between methods')


    finite = np.isfinite(frame[X_COLUMN]) & np.isfinite(frame[metric])
    require(frame.loc[finite, X_COLUMN].between(0, 0.7).all(),
            f'{INPUT_CSV}: values exceed the original 0..0.7 instability axis')
    require(frame.loc[finite, metric].between(0, Y_MAX).all(),
            f'{INPUT_CSV}: {metric} exceeds the original 0..{Y_MAX} y-axis')
    frame['plot_included'] = finite
    print(f'[loaded] {metric}: {len(frame)} clusters, {int(finite.sum())} finite pairs, '
          f'18 samples x 10 methods at {nruns} runs', flush=True)
    return frame


def write_csv(frame, filename):
    path = OUTPUT_DIR / filename
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_tables(frame):
    write_csv(frame, OUTPUT_STEM + '_data.csv')
    coverage = frame.groupby(['dataset-sample', 'method'], as_index=False).agg(
        n_clusters=('cluster', 'size'), n_clusters_plotted=('plot_included', 'sum'),
        n_spots=('n_spots', 'sum'), reference_run=('reference_run', 'first'),
        reference_seed=('reference_seed', 'first'))
    coverage['n_clusters_missing_metric'] = coverage['n_clusters'] - coverage['n_clusters_plotted']
    write_csv(coverage, OUTPUT_STEM + '_coverage.csv')
    if not HAS_CORRELATION:
        return
    rows = []
    for method in mthds:
        subset = frame.loc[frame['method'].eq(method) & frame['plot_included']]
        x, y = subset[X_COLUMN].to_numpy(float), subset[metric].to_numpy(float)
        row = dict(method=method, metric=metric, n_clusters=len(subset),
                   n_samples=subset['dataset-sample'].nunique(), pearson_r=np.nan,
                   pearson_r_squared=np.nan, pearson_pvalue=np.nan,
                   slope=np.nan, intercept=np.nan, status='insufficient_or_constant_data')
        if len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
            r, p = stats.pearsonr(x, y)
            slope, intercept = np.polyfit(x, y, 1)
            row.update(pearson_r=float(r), pearson_r_squared=float(r ** 2),
                       pearson_pvalue=float(p), slope=float(slope),
                       intercept=float(intercept), status='ok')
        rows.append(row)
    write_csv(pd.DataFrame(rows), OUTPUT_STEM + '_correlations.csv')


def save_figure(fig):
    'Export the same Figure as PDF and PNG using the old 600-DPI settings.'
    pending = []
    try:
        for extension in ('pdf', 'png'):
            path = OUTPUT_DIR / f'{OUTPUT_STEM}.{extension}'
            temporary = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.{extension}')
            pending.append((temporary, path))
            fig.savefig(temporary, format=extension, dpi=FIG_DPI)
        for temporary, path in pending:
            temporary.replace(path)
            print(f'[saved] {path}', flush=True)
    finally:
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)


def save_configuration():
    source_stat = INPUT_CSV.stat()
    config = dict(
        created_at=datetime.now().astimezone().isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_csv=str(INPUT_CSV.relative_to(ROOT)), input_bytes=source_stat.st_size,
        input_mtime_ns=source_stat.st_mtime_ns,
        input_sha256=hashlib.sha256(INPUT_CSV.read_bytes()).hexdigest(),
        output_dir=OUTPUT_DIR.name, metric=metric, metric_definition=METRIC_DEFINITION,
        methods=mthds, samples=SAMPLE_TAGS, iterations=nruns, reference_mode='example_50',
        observation_unit='one example cluster; unweighted across clusters',
        figsize=list(FIGSIZE), method_layout=[NROWS, NCOLS], dpi=FIG_DPI,
        xlim=[0, 0.7], ylim=[0, Y_MAX], style_source=STYLE_SOURCE,
        has_pearson_fit=HAS_CORRELATION,
        annotation_matching='maximum-Jaccard annotation on nonmissing annotation spots; shared match for Jaccard, recall, precision',
        expression_space='normalized/log1p counts; centered PCA; unit-length first 20 PCs',
    )
    path = OUTPUT_DIR / 'plot_config.json'
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(config, indent=2) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    require(len(mthds) == NROWS * NCOLS == 10 and len(SAMPLE_TAGS) == 18,
            "Expected a 2 x 5 method layout and 18 samples")
    df = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_tables(df)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12 * 5 / 4, 6))
    try:
        axes_flat = axes.flatten()


        for axi, ax in enumerate(axes_flat):
            print(axi)
            print(mthds[axi])
            print(mnames[axi])
            if axi == len(mthds):
                break
            else:
                mthd = mthds[axi]
                mname = mnames[axi]
                xdf = df[df['method'] == mthd]

                ax.set_title(mname, fontsize=14, fontweight='bold')
                if axi >= 5:
                    ax.set_xlabel("Cluster Instability", fontsize=12)
                if axi % 5 == 0:
                    ax.set_ylabel("Recall w.r.t. Ref.", fontsize=12)

                ax.set_ylim(0, 1.0)
                ax.set_xlim(0, 0.7)
                ax.set_xticks([0, 0.2, 0.4, 0.6])


                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)


                box_x = 0.22
                box_y = 0.0
                box_width = 30
                box_height = 0.48


                highlight_box = patches.FancyBboxPatch(
                    (box_x, box_y),
                    box_width,
                    box_height,
                    boxstyle="round,pad=0.02",
                    ec="black",
                    fc=(0.5, 0.5, 0.5, 0.2),
                    linestyle=':',
                    linewidth=2,
                    zorder=10
                )


                ax.add_patch(highlight_box)


                box_x = 0.42
                box_y = 0.0
                box_width = 30
                box_height = 0.48


                highlight_box = patches.FancyBboxPatch(
                    (box_x, box_y),
                    box_width,
                    box_height,
                    boxstyle="round,pad=0.02",
                    ec="black",
                    fc=(0.5, 0.5, 0.5, 0.2),
                    linestyle='--',
                    linewidth=2,
                    zorder=10
                )


                ax.add_patch(highlight_box)

                for j, dataset in enumerate(datas):
                    if dataset == "dlpfc":
                        ydf = xdf[
                            xdf["dataset-sample"]
                            .astype(str)
                            .str.startswith("dlpfc_", na=False)
                        ]
                    else:
                        ydf = xdf[
                            xdf["dataset-sample"] == dataset
                        ]

                    xx = ydf['clusterwise mean instability'].to_numpy(dtype=float)
                    yy = ydf[metric].to_numpy(dtype=float)

                    valid = (
                        np.isfinite(xx)
                        & np.isfinite(yy)
                    )

                    xx = xx[valid]
                    yy = yy[valid]

                    if axi == mthds.index("sedr"):
                        ax.scatter(xx, yy, color=colors[j],
                            s=20, label=dnames[j], alpha=0.8)
                    else:
                        ax.scatter(xx, yy, color=colors[j],
                           s=20, alpha=0.8)

                    if len(xx) > 0:
                        if np.max(xx) > 0.7:
                            print("oh no")

                if axi == mthds.index("sedr"):
                    print("adding legend")
                    fig.legend(loc='upper center',
                           bbox_to_anchor=(0.5, 0.08),
                           bbox_transform=fig.transFigure,
                           fontsize=14,
                           ncol=7,
                           markerscale=2.5,
                           frameon=False,)

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.15)
        save_figure(fig)
    finally:
        plt.close(fig)
    save_configuration()


if __name__ == "__main__":
    main()
