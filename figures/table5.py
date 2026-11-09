#!/usr/bin/env python3
'Table 5: mean ARI across 50 runs, using saved sample metrics.'
from __future__ import annotations

from common import DATA_DIR, OUTPUT_ROOT

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sample_metrics import load_sample_metrics

BASE = OUTPUT_ROOT
METRICS_CSV = DATA_DIR / 'metrics/sample_metrics.csv'
OUT_DIR = OUTPUT_ROOT / 'tables/t5'
OUTPUT_STEM = "table5_mean_ari_sample_method"
SUMMARY_STEM = "table5_mean_ari_method_summary"
METHODS = [
    ("sedr", "SEDR"), ("sedr_mclust", "SEDR (mclust)"),
    ("graphst", "GraphST"), ("graphst_mclust", "GraphST (mclust)"),
    ("stagate", "STAGATE"), ("stagate_mclust", "STAGATE (mclust)"),
    ("bayesspace", "BayesSpace"), ("louvain", "Louvain"),
    ("leiden", "Leiden"), ("spicemix", "SpiceMix"),
]
SAMPLES = [
    ("mouse_brain_cerebellum", "", "MBC"),
    ("Visium_HD_Human_Colon_Cancer_cropped_square", "", "CRC"),
    ("human_breast_cancer", "", "HBC"), ("mouse_brain", "", "MB"),
    ("coad_ffpe", "", "COAD"), ("ov_ffpe", "", "OV"),
] + [("dlpfc", section, section) for section in [
    "151507", "151508", "151509", "151510", "151669", "151670",
    "151671", "151672", "151673", "151674", "151675", "151676",
]]
SAMPLE_ORDER = [label for _, _, label in SAMPLES]
DLPFC = SAMPLE_ORDER[6:]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def file_record(path):
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)

def load_metrics(columns):
    'Require one unambiguous 50-run row for each requested sample identity.'
    record = file_record(METRICS_CSV)
    frame = load_sample_metrics()
    identity = ["dataset", "sample", "sample_label", "iterations", "example_mode"]
    required = identity + list(columns)
    require(set(required).issubset(frame.columns),
            f"{METRICS_CSV}: missing columns {sorted(set(required) - set(frame.columns))}")
    frame["sample"] = frame["sample"].fillna("")
    frame["iterations"] = pd.to_numeric(frame["iterations"], errors="raise")
    requested = {(dataset, sample) for dataset, sample, _ in SAMPLES}
    selected = [(dataset, sample) in requested
                for dataset, sample in frame[["dataset", "sample"]].itertuples(index=False, name=None)]
    frame = frame.loc[frame["iterations"].eq(50) & np.asarray(selected), required].copy()
    require(len(frame) == len(SAMPLES) and not frame.duplicated(["dataset", "sample"]).any(),
            f"{METRICS_CSV}: expected one 50-run row per sample; no duplicate rows may be discarded")
    observed = set(frame[["dataset", "sample", "sample_label"]].itertuples(index=False, name=None))
    require(observed == set(SAMPLES), f"{METRICS_CSV}: sample identities or display labels disagree")
    require(frame["example_mode"].eq("example_50").all(),
            f"{METRICS_CSV}: expected the current example_50 metrics")
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        require(np.isfinite(frame[column]).all(), f"{METRICS_CSV}: missing/nonfinite {column}")
        if column.endswith("mean_ari_all_runs"):
            require(frame[column].between(-1, 1).all(), f"{METRICS_CSV}: invalid ARI")
        else:

            require(frame[column].ge(0).all(), f"{METRICS_CSV}: negative {column}")
    digest = hashlib.sha256(METRICS_CSV.read_bytes()).hexdigest()
    require(file_record(METRICS_CSV) == record, f"{METRICS_CSV}: changed while reading")
    frame = frame.set_index("sample_label").loc[SAMPLE_ORDER].reset_index()
    provenance = dict(input_csv=dict(record, sha256=digest), iterations=50,
                      example_mode="example_50", samples=SAMPLES, columns=list(columns),
                      excluded_methods=["mclust"],
                      excluded_samples=["Visium_HD_Human_Colon_Cancer", "ov_ffpe_full", "coad_ffpe_full"],
                      standard_deviation_ddof=1)
    return frame, provenance

def mean_sd(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else np.nan

def fmt_value(value, bold=False):
    if not np.isfinite(value):
        return "--"
    text = f"{value:.2f}"
    return rf"\textbf{{{text}}}" if bold else text

def fmt_mean_sd(mean, sd):
    if not np.isfinite(mean):
        return "--"
    sd_text = "--" if not np.isfinite(sd) else f"{sd:.2f}"
    return rf"${mean:.2f} \pm {sd_text}$"

def build_matrix(frame, metric):
    rows = []
    source = frame.set_index("sample_label")
    for sample in SAMPLE_ORDER:
        row = {"Sample": sample}
        for method, display in METHODS:
            row[display] = float(source.loc[sample, f"{method}_{metric}"])
        values = [row[display] for _, display in METHODS]
        row["Mean across methods"], row["SD across methods"] = mean_sd(values)
        row["n_methods"] = len(values)
        rows.append(row)
    table = pd.DataFrame(rows)
    summaries = []
    for scope, samples in [("All", SAMPLE_ORDER), ("DLPFC only", DLPFC)]:
        subset = table.loc[table["Sample"].isin(samples)]
        for method, display in METHODS:
            mean, sd = mean_sd(subset[display])
            summaries.append(dict(Scope=scope, Method=display, method=method,
                                  n_samples=len(subset), Mean=mean, SD=sd))
    return table, pd.DataFrame(summaries)

def latex_matrix(table, summary, caption, label, highlight_best=False):
    'Keep the old scriptsize, 4pt spacing, resized layout, and two-line header.'
    ncolumns = len(METHODS) + 2
    lines = []
    for i, row in table.iterrows():
        if i == 6:
            lines += [r"\midrule", rf"\multicolumn{{{ncolumns}}}{{l}}{{\textbf{{DLPFC}}}}\\"]
        numeric = np.asarray([row[display] for _, display in METHODS], dtype=float)
        maximum = numeric.max()
        cells = [fmt_value(value, highlight_best and np.isclose(value, maximum, rtol=0, atol=1e-12))
                 for value in numeric]
        lines.append(" & ".join([row["Sample"], *cells, fmt_mean_sd(*mean_sd(numeric))]) + r" \\")
    lines += [r"\midrule", rf"\multicolumn{{{ncolumns}}}{{l}}{{\textbf{{Mean $\pm$ S.D.}}}}\\"]
    indexed = summary.set_index(["Scope", "Method"])
    for scope in ("All", "DLPFC only"):
        cells = [fmt_mean_sd(indexed.loc[(scope, display), "Mean"],
                             indexed.loc[(scope, display), "SD"]) for _, display in METHODS]
        lines.append(" & ".join([scope, *cells, ""]) + r" \\")
    top, bottom = ["Sample"], [""]
    for method, display in METHODS:
        top.append("Bayes" if method == "bayesspace" else display.replace(" (mclust)", ""))
        bottom.append("Space" if method == "bayesspace" else "mclust" if method.endswith("_mclust") else "")
    top.append(r"Mean $\pm$ S.D.")
    bottom.append("")
    header = " & ".join(top) + r" \\" + "\n" + " & ".join(bottom) + r" \\"
    body = "\n".join(lines)
    alignment = "l" + "r" * (ncolumns - 1)
    return rf"""% Requires booktabs and graphicx in the manuscript preamble.
\begin{{table}}[p]
\centering
\scriptsize
\caption{{{caption}}}
\label{{{label}}}
\setlength{{\tabcolsep}}{{4pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{{alignment}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""

def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, float_format="%.8f", na_rep="")
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

def save_provenance(provenance, out_dir, script, definition):
    configuration = dict(provenance, output_dir=str(out_dir), definition=definition,
                         script=str(script), script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                         shared_code=str(Path(__file__).resolve()),
                         shared_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_text(json.dumps(configuration, indent=2) + "\n", out_dir / "input_provenance.json")

def generate_matrix(metric, out_dir, output_stem, summary_stem, caption, label, script, highlight_best=False):
    frame, provenance = load_metrics([f"{method}_{metric}" for method, _ in METHODS])
    table, summary = build_matrix(frame, metric)
    write_csv(table, out_dir / f"{output_stem}.csv")
    write_csv(summary, out_dir / f"{summary_stem}.csv")
    write_text(latex_matrix(table, summary, caption, label, highlight_best), out_dir / f"{output_stem}.tex")
    write_csv(frame, out_dir / "source_metrics_50runs.csv")
    save_provenance(dict(provenance, methods=METHODS), out_dir, script, caption)
    print(table.to_string(index=False))
    print(f"\nSaved table, summaries, source metrics, and provenance to {out_dir}", flush=True)

def main():
    generate_matrix(
        "mean_ari_all_runs", OUT_DIR, OUTPUT_STEM, SUMMARY_STEM,
        "Mean ARI against the reference annotation across 50 runs for each sample--method pair. "
        "Each of ten methods is evaluated on the same 18 samples. Bold values indicate the highest "
        "mean ARI within a sample, including ties. Summary values are means and sample standard deviations.",
        "tab:mean_ari_sample_method", Path(__file__).resolve(), highlight_best=True,
    )

if __name__ == "__main__":
    main()
