"""Read compact DEG/spatial summaries and derive shared cluster metadata."""

from functools import lru_cache

import h5py
from anndata.io import read_elem
import numpy as np
import pandas as pd

from common import DATA_DIR


def read_csv(filename):
    return pd.read_csv(DATA_DIR / filename, keep_default_na=False, na_values=[''],
                       dtype={'sample_id': str, 'method': str, 'example_cluster': str,
                              'target_cluster': str}, float_precision='round_trip')


def cluster_strings(values):
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.number) and np.equal(values, np.rint(values)).all():
        values = values.astype(np.int64)
    return values.astype(str)


@lru_cache(maxsize=1)
def cluster_metadata():
    table = read_csv('dgea/clusters.csv.gz')
    parts = []
    for sample, group in table.groupby('sample_id', sort=False):
        with h5py.File(DATA_DIR / 'spatial' / f'{sample}.h5ad', 'r') as handle:
            obs = read_elem(handle['obs'])
            labels = read_elem(handle['uns/example_labels'])
            indices = read_elem(handle['uns/example_indices'])
            for method in group['method'].unique():
                if method not in labels:
                    labels[method] = np.asarray(read_elem(handle[f'uns/{method}_results']))[int(indices[method])]
        for method, selected in group.groupby('method', sort=False):
            selected = selected.copy()
            values = obs[f'{method}_uncertainty'].to_numpy(float)
            assignment = cluster_strings(labels[method])
            if len(assignment) != len(values) or not np.isfinite(values).all():
                raise ValueError(f'Invalid spatial metadata: {sample}/{method}')
            selected['example_run_idx'] = int(indices[method])
            selected['example_n_cells'] = [int(np.count_nonzero(assignment == c))
                                            for c in selected['example_cluster']]
            selected['example_mean_instability'] = [float(values[assignment == c].mean())
                                                    for c in selected['example_cluster']]
            if not selected['example_n_cells'].gt(0).all():
                raise ValueError(f'Missing example cluster: {sample}/{method}')
            parts.append(selected)
    result = pd.concat(parts, ignore_index=True).sort_values('cluster_id').reset_index(drop=True)
    labels = [f'[{i / 10:.1f},{(i + 1) / 10:.1f})' for i in range(9)] + ['[0.9,1.0]']
    bins = np.r_[np.arange(0.0, 1.0, 0.1), np.nextafter(1.0, np.inf)]
    result['example_fine_bin'] = pd.cut(result['example_mean_instability'], bins=bins,
                                       labels=labels, right=False, include_lowest=True).astype(str)
    return result


def with_clusters(frame):
    return frame.merge(cluster_metadata(), on='cluster_id', how='left', sort=False,
                       validate='many_to_one')


@lru_cache(maxsize=1)
def matches():
    return with_clusters(read_csv('dgea/matches.csv.gz'))


@lru_cache(maxsize=1)
def cluster_summary():
    means = matches().groupby('cluster_id', sort=False)['best_jaccard'].mean()
    table = cluster_metadata().copy()
    table['best_jaccard_mean'] = table['cluster_id'].map(means)
    return table


@lru_cache(maxsize=1)
def consistency():
    table = with_clusters(read_csv('dgea/consistency.csv.gz'))
    table['n_comparisons_expected'] = 49
    table['jaccard_n_valid'] = table['n_comparisons_available']
    return table


@lru_cache(maxsize=1)
def reference_quality():
    table = with_clusters(read_csv('fig3/reference_quality.csv.gz'))
    return table.rename(columns={'sample_id': 'dataset-sample',
                                 'example_mean_instability': 'clusterwise mean instability'})


def showcase_table(filename):
    metadata = cluster_metadata()
    metadata = metadata.loc[metadata['sample_id'].eq('human_breast_cancer')
                            & metadata['method'].eq('sedr_mclust')].copy()
    metadata['dataset'] = 'human_breast_cancer'
    if filename == 'example_clusters.csv':
        return metadata
    if filename == 'cross_run_matches.csv':
        table = read_csv('fig3/showcase_alignment.csv.gz').merge(
            read_csv('dgea/matches.csv.gz'), on=['cluster_id', 'target_run'],
            how='left', sort=False, validate='one_to_one')
    elif filename == 'reference_matches.csv':
        table = read_csv('fig3/showcase_reference.csv.gz').merge(
            read_csv('fig3/reference_quality.csv.gz')[['cluster_id', 'clusterwise jaccard']],
            on='cluster_id', how='left', sort=False, validate='one_to_one')
        table = table.rename(columns={'clusterwise jaccard': 'best_jaccard'})
    else:
        raise ValueError(f'Unknown showcase table: {filename}')
    return table.merge(metadata, on='cluster_id', how='left', sort=False, validate='many_to_one')
