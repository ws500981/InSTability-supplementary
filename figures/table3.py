#!/usr/bin/env python3
'Table 3: cluster instability by method for the benchmark results.'
from __future__ import annotations

from common import DATA_DIR, OUTPUT_ROOT

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from anndata.io import read_elem
from cluster_metrics import load_cluster_metrics
from scipy.stats import pearsonr

BASE = OUTPUT_ROOT
INSTABILITY_ROOT = DATA_DIR / 'spatial'
CLUSTER_CSV = DATA_DIR / 'metrics/clusterwise_metrics.csv'
OUT_DIR = OUTPUT_ROOT / 'tables/t3'
OUT_CSV = OUT_DIR / "table3_cluster_instability_by_method.csv"
OUT_TEX = OUT_DIR / "table3_cluster_instability_by_method.tex"
INSTABILITY_COLUMN = "clusterwise mean instability"
UNSTABLE_THRESHOLD = 0.4
CATEGORIES = ("s", "ms", "u", "hu")

DLPFC = [
    "151507", "151508", "151509", "151510", "151669", "151670",
    "151671", "151672", "151673", "151674", "151675", "151676",
]
SAMPLE_ROWS = [
    ("mouse_brain_cerebellum", None, "MBC"),
    ("Visium_HD_Human_Colon_Cancer_cropped_square", None, "CRC"),
    ("human_breast_cancer", None, "HBC"),
    ("mouse_brain", None, "MB"),
    ("coad_ffpe", None, "COAD"),
    ("ov_ffpe", None, "OV"),
] + [("dlpfc", sample, sample) for sample in DLPFC]
SAMPLES = [(dataset, sample) for dataset, sample, _ in SAMPLE_ROWS]

METHOD_ROWS = [
    ("sedr", "SEDR"), ("sedr_mclust", "SEDR (mclust)"),
    ("graphst", "GraphST"), ("graphst_mclust", "GraphST (mclust)"),
    ("stagate", "STAGATE"), ("stagate_mclust", "STAGATE (mclust)"),
    ("bayesspace", "BayesSpace"), ("louvain", "Louvain"),
    ("leiden", "Leiden"), ("spicemix", "SpiceMix"),
]
METHODS = [method for method, _ in METHOD_ROWS]
METRICS = {
    "Purity R": "clusterwise purity (precision)",
    "Recall R": "clusterwise recall",
    "Jaccard R": "clusterwise jaccard",
    "Exp. Coherence R": "expression coherence",
    "Exp. Variance R": "expression variance",
}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def tag(dataset, sample):
    return f"{dataset}_{sample}" if sample else dataset

def file_record(path):
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)

def benchmark_cluster_data():
    'Select and validate all 180 units at 50 runs; never silently drop units.'
    source_record = file_record(CLUSTER_CSV)
    df = load_cluster_metrics(iterations=[50])
    required = {"dataset-sample", "method", "iterations", "reference_mode",
                "reference_run", "reference_seed", "expression_source", "cluster",
                "n_spots", INSTABILITY_COLUMN, *METRICS.values()}
    require(required.issubset(df.columns),
            f"{CLUSTER_CSV}: missing columns {sorted(required - set(df.columns))}")
    allowed = {tag(dataset, sample) for dataset, sample in SAMPLES}
    df["iterations"] = pd.to_numeric(df["iterations"], errors="raise")
    df = df.loc[df["iterations"].eq(50) & df["dataset-sample"].isin(allowed)
                & df["method"].isin(METHODS)].copy()
    require(not df[list(required - set(METRICS.values()))].isna().any().any(),
            f"{CLUSTER_CSV}: missing required cluster metadata")
    require(df["reference_mode"].eq("example_50").all(),
            f"{CLUSTER_CSV}: expected fixed example_50 clusters")
    require(df["expression_source"].isin(["layers/counts", "raw/X"]).all(),
            f"{CLUSTER_CSV}: unexpected expression source")
    keys = ["dataset-sample", "method"]
    observed = set(df[keys].itertuples(index=False, name=None))
    expected = {(sample, method) for sample in allowed for method in METHODS}
    require(observed == expected, f"{CLUSTER_CSV}: missing units {sorted(expected - observed)}")
    require(not df.duplicated(keys + ["cluster"]).any(), f"{CLUSTER_CSV}: duplicate clusters")
    for column in [INSTABILITY_COLUMN, "n_spots", "reference_run", "reference_seed", *METRICS.values()]:
        df[column] = pd.to_numeric(df[column], errors="raise")
        require(not np.isinf(df[column]).any(), f"{CLUSTER_CSV}: infinite {column}")
        if column not in METRICS.values():
            require(np.isfinite(df[column]).all(), f"{CLUSTER_CSV}: nonfinite {column}")
    for column, low, high in [("n_spots", 1, np.inf), ("reference_run", 0, 49),
                              ("reference_seed", 42, 91)]:
        require(df[column].between(low, high).all() and df[column].mod(1).eq(0).all(),
                f"{CLUSTER_CSV}: invalid {column}")
        df[column] = df[column].astype(np.int64)
    require(df["reference_seed"].eq(df["reference_run"] + 42).all(),
            f"{CLUSTER_CSV}: reference seeds and indices disagree")
    require(df.groupby(keys)["reference_run"].nunique().eq(1).all(),
            f"{CLUSTER_CSV}: multiple example runs per unit")
    require(df[INSTABILITY_COLUMN].ge(0).all(), f"{CLUSTER_CSV}: negative instability")
    require(file_record(CLUSTER_CSV) == source_record, f"{CLUSTER_CSV}: changed while reading")
    df["cluster_category"] = df[INSTABILITY_COLUMN].map(category)
    return df, source_record

def spot_fraction_data(clusters):
    "Compute unstable-spot fractions from the bundled 50-run scores."
    rows = []
    for dataset, sample in SAMPLES:
        sample_tag = tag(dataset, sample)
        with h5py.File(DATA_DIR / 'spatial' / f'{sample_tag}.h5ad', 'r') as f:
            obs = read_elem(f['obs'])
        for method in METHODS:
            group = clusters.loc[clusters['dataset-sample'].eq(sample_tag) & clusters['method'].eq(method)]
            require(group['n_spots'].sum() == len(obs), 'Cluster and spot counts disagree')
            n_unstable = int((obs[method + '_uncertainty'] >= UNSTABLE_THRESHOLD).sum())
            rows.append(dict(dataset=dataset, **{'dataset-sample': sample_tag}, method=method,
                             n_spots=len(obs), n_unstable_spots=n_unstable,
                             frac_unstable=n_unstable / len(obs), n_clusters=len(group),
                             reference_run=int(group['reference_run'].iloc[0]),
                             reference_seed=int(group['reference_seed'].iloc[0])))
    return pd.DataFrame(rows), []


def load_benchmark_data():
    clusters, source_record = benchmark_cluster_data()
    fractions, manifest = spot_fraction_data(clusters)
    digest = hashlib.sha256(CLUSTER_CSV.read_bytes()).hexdigest()
    require(file_record(CLUSTER_CSV) == source_record, f"{CLUSTER_CSV}: changed during input loading")
    provenance = dict(
        cluster_csv=dict(source_record, sha256=digest), instability_root=str(INSTABILITY_ROOT),
        methods=METHODS, samples=[tag(*sample) for sample in SAMPLES], iterations=50,
        reference_mode="example_50", unstable_threshold=UNSTABLE_THRESHOLD,
        category_intervals={"s": "[0,0.2)", "ms": "[0.2,0.4)", "u": "[0.4,0.6)", "hu": "[0.6,infinity)"},
        standard_deviation_ddof=1, units=manifest,
    )
    return clusters, fractions, provenance

def mean_sd(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else np.nan

def category(value):
    if not np.isfinite(value):
        return None
    if value < 0.2:
        return "s"
    if value < 0.4:
        return "ms"
    if value < 0.6:
        return "u"
    return "hu"

def cluster_fraction_text(group):
    if not len(group):
        return "--"
    return "/".join(f"{group['cluster_category'].eq(label).mean():.2f}" for label in CATEGORIES)

def category_statistics(group):
    return {key: value for label in CATEGORIES for key, value in (
        (f"n_clusters_{label}", int(group["cluster_category"].eq(label).sum())),
        (f"fraction_clusters_{label}", float(group["cluster_category"].eq(label).mean())),
    )}

def finite_pearson_r(group, metric):
    x = group[INSTABILITY_COLUMN].to_numpy(float)
    y = group[metric].to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, len(x)
    return float(pearsonr(x, y).statistic), len(x)

def format_mean_sd(mean, sd):
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(sd):
        return f"{mean:.2f} $\\pm$ --"
    return f"{mean:.2f} $\\pm$ {sd:.2f}"

def format_r(row, metric):
    r, r2 = row[metric], row[f"{metric}^2"]
    return "--" if pd.isna(r) else f"{r:.2f} ({r2:.2f})"

def latex_table(table):
    lines = []
    for _, row in table.iterrows():
        cells = [row["Method"],
                 format_mean_sd(row["mean frac. unstable spots"], row["sd frac. unstable spots"]),
                 str(int(row["total # clusters"])), row["fraction of clusters (s/ms/u/hu)"]]
        cells += [format_r(row, metric) for metric in METRICS]
        lines.append(" & ".join(cells) + r"\\")
    body = "\n".join(lines)
    return rf"""% Requires rotating, array, and booktabs in the manuscript preamble.
\begin{{sidewaystable}}[p]
\centering
\scriptsize
\caption{{Cluster instability by method.}}
\label{{tab:bymethod_new}}
\begin{{tabular}}{{
>{{\raggedright\arraybackslash}}m{{1.6cm}}
>{{\centering\arraybackslash}}m{{2cm}}
>{{\centering\arraybackslash}}m{{1.2cm}}
>{{\centering\arraybackslash}}m{{3cm}}
>{{\raggedleft\arraybackslash}}m{{1.6cm}}
>{{\raggedleft\arraybackslash}}m{{1.6cm}}
>{{\raggedleft\arraybackslash}}m{{1.6cm}}
>{{\raggedleft\arraybackslash}}m{{2.6cm}}
>{{\raggedleft\arraybackslash}}m{{2.3cm}}
}}
\toprule
Method & mean frac. unstable spots & total \# clusters & fraction of clusters (s/ms/u/hu) & Purity R & Recall R & Jaccard R & Exp. Coherence R & Exp. Variance R \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\vspace{{0.3em}}
\parbox{{0.95\textheight}}{{
\scriptsize \textit{{Note.}}
Rows correspond to ten clustering methods evaluated on the same 18 benchmark samples.
The second column averages each sample's fraction of spots with nested 50-run instability $\geq 0.4$, giving each sample equal weight; sample standard deviation is shown.
The third column counts clusters from the fixed example run (the first run with the fewest clusters among 50) for each of the 18 samples.
The fourth column pools these clusters into $s=$ stable $[0,0.2)$, $ms=$ modestly stable $[0.2,0.4)$, $u=$ unstable $[0.4,0.6)$, and $hu=$ highly unstable $[0.6,\infty)$ using their mean spot instability.
The remaining columns show Pearson $R$ ($R^2$) between cluster instability and the indicated metric, pooling clusters across samples with equal weight per cluster and using finite pairs for each metric. Undefined correlations are shown as --.
Standalone mclust and the three full Visium HD samples are excluded; the GraphST, STAGATE, and SEDR mclust variants and the cropped samples are retained.
}}
\end{{sidewaystable}}
"""

def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, float_format="%.10g", na_rep="")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def write_text(text, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def save_supporting_data(out_dir, clusters, fractions, provenance, script, averaging):
    write_csv(clusters, out_dir / "cluster_data_50runs.csv")
    write_csv(fractions, out_dir / "sample_method_spot_fractions.csv")
    configuration = dict(provenance, output_dir=str(out_dir), averaging=averaging,
                         script=str(script), script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                         shared_code=str(Path(__file__).resolve()),
                         shared_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_text(json.dumps(configuration, indent=2) + "\n", out_dir / "input_provenance.json")

def main():
    clusters, spot_fractions, provenance = load_benchmark_data()
    rows = []
    for method, display in METHOD_ROWS:
        cluster_group = clusters.loc[clusters["method"].eq(method)]
        spot_group = spot_fractions.loc[spot_fractions["method"].eq(method)]
        frac_mean, frac_sd = mean_sd(spot_group["frac_unstable"])
        row = {
            "Method": display, "method": method, "analysis_scope": "all",
            "n_samples": len(spot_group), "mean frac. unstable spots": frac_mean,
            "sd frac. unstable spots": frac_sd, "total # clusters": len(cluster_group),
            "fraction of clusters (s/ms/u/hu)": cluster_fraction_text(cluster_group),
            **category_statistics(cluster_group),
        }
        for name, column in METRICS.items():
            r, n_pairs = finite_pearson_r(cluster_group, column)
            row[name] = r
            row[f"{name}^2"] = r * r if np.isfinite(r) else np.nan
            row[f"{name} n_pairs"] = n_pairs
        rows.append(row)
    table = pd.DataFrame(rows)
    write_csv(table, OUT_CSV)
    write_text(latex_table(table), OUT_TEX)
    save_supporting_data(OUT_DIR, clusters, spot_fractions, provenance, Path(__file__).resolve(),
                         "Equal-weight sample mean and sample SD per method; cluster statistics pool samples")
    print(table.to_string(index=False))
    print(f"\nSaved table and supporting data to {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
