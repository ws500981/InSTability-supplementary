"""Reconstruct cluster summaries from the compact plotting inputs."""

from anndata.io import read_elem
import h5py
import numpy as np
import pandas as pd

from common import DATA_DIR

CLUSTER_METRICS_CSV = DATA_DIR / 'metrics' / 'clusterwise_metrics.csv'
RUN_COUNTS = (5, 10, 20, 40, 50)


def cluster_sort_key(value):
    value = str(value)
    return (0, int(value)) if value.isdigit() else (1, value)


def match_cluster_to_annotation(cluster_mask, valid_annotation, annotation, annotation_sizes):
    """Match by Jaccard, overlap size, then annotation order."""
    eval_mask = cluster_mask & valid_annotation
    if not eval_mask.any():
        return None, np.nan, np.nan, np.nan
    predicted_size = int(eval_mask.sum())
    labels, overlaps = np.unique(annotation[eval_mask], return_counts=True)
    candidates = [(str(label), int(n), int(annotation_sizes[str(label)]),
                   float(n / (predicted_size + annotation_sizes[str(label)] - n)))
                  for label, n in zip(labels, overlaps)]
    best = max(row[3] for row in candidates)
    ties = [row for row in candidates if np.isclose(row[3], best, rtol=1e-12, atol=1e-12)]
    label, n, reference_size, jaccard = min(ties, key=lambda row: (-row[1], cluster_sort_key(row[0])))
    return label, jaccard, n / predicted_size, n / reference_size


def load_cluster_metrics(iterations=(50,), methods=None):
    """Derive cluster summaries without loading expression matrices or images."""
    iterations = list(dict.fromkeys(iterations))
    if not iterations or not set(iterations).issubset(RUN_COUNTS):
        raise ValueError(f'Run counts must be chosen from {RUN_COUNTS}')
    fixed = pd.read_csv(CLUSTER_METRICS_CSV, dtype={'cluster': str})
    if methods is not None:
        fixed = fixed.loc[fixed['method'].isin(methods)]
    keys = ['dataset-sample', 'method', 'cluster']
    if fixed.empty or fixed.duplicated(keys).any():
        raise ValueError('Cluster expression metrics are empty or contain duplicates')
    output = []
    for tag, sample_rows in fixed.groupby('dataset-sample', sort=False):
        with h5py.File(DATA_DIR / 'spatial' / f'{tag}.h5ad', 'r') as handle:
            obs = read_elem(handle['obs'])
            labels_by_method = read_elem(handle['uns/example_labels'])
            reference_indices = read_elem(handle['uns/example_indices'])
        valid = obs['annotation'].notna().to_numpy() if 'annotation' in obs else np.zeros(len(obs), bool)
        annotation = obs['annotation'].astype(str).to_numpy() if 'annotation' in obs else np.full(len(obs), '')
        labels, counts = np.unique(annotation[valid], return_counts=True)
        annotation_sizes = dict(zip(labels, counts))
        for method, method_rows in sample_rows.groupby('method', sort=False):
            predicted = np.asarray(labels_by_method[method]).astype(str)
            if predicted.shape != (len(obs),):
                raise ValueError(f'{tag}/{method}: example labels do not align with spots')
            if set(predicted) != set(method_rows['cluster']):
                raise ValueError(f'{tag}/{method}: expression metrics do not cover example clusters')
            reference_index = int(reference_indices[method])
            scores = {}
            if 50 in iterations:
                scores[50] = obs[f'{method}_uncertainty'].to_numpy(dtype=float)
            if any(k != 50 for k in iterations):
                with np.load(DATA_DIR / 'spot_instability' / f'{tag}__{method}.npz', allow_pickle=False) as data:
                    scores.update({k: np.asarray(data[f'scores_{k}'], dtype=float) for k in iterations if k != 50})
            for k, values in scores.items():
                if values.shape != (len(obs),) or not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(f'{tag}/{method}: invalid scores for {k} runs')
            metrics = []
            for row in method_rows.to_dict('records'):
                indices = np.flatnonzero(predicted == row['cluster'])
                matched, jaccard, precision, recall = match_cluster_to_annotation(
                    predicted == row['cluster'], valid, annotation, annotation_sizes)
                metrics.append((indices, dict(row, n_spots=len(indices),
                    **{'matched annotation': matched, 'clusterwise jaccard': jaccard,
                       'clusterwise purity (precision)': precision, 'clusterwise recall': recall})))
            for k in iterations:
                for indices, row in metrics:
                    output.append(dict(row, iterations=k, reference_mode='example_50',
                        reference_run=reference_index, reference_seed=reference_index + 42,
                        **{'clusterwise mean instability': float(scores[k][indices].mean())}))
    columns = ['dataset-sample', 'method', 'iterations', 'reference_mode', 'reference_run',
               'reference_seed', 'expression_source', 'clusterwise mean instability', 'cluster',
               'n_spots', 'matched annotation', 'clusterwise jaccard',
               'clusterwise purity (precision)', 'clusterwise recall',
               'expression coherence', 'expression variance']
    return pd.DataFrame(output, columns=columns)
