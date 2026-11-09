#!/usr/bin/env python3
"""Format Table 8 from saved preparation and mean clustering runtimes."""
import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
from common import DATA_DIR, OUTPUT_ROOT

METHODS = [('leiden', 'Leiden'), ('louvain', 'Louvain'), ('bayesspace', 'BayesSpace'), ('graphst', 'GraphST'), ('graphst_mclust', 'GraphST (mclust)'), ('stagate', 'STAGATE'), ('stagate_mclust', 'STAGATE (mclust)'), ('spicemix', 'SpiceMix'), ('sedr', 'SEDR'), ('sedr_mclust', 'SEDR (mclust)')]

def caption(hardware_sentence):
    return r"""\textbf{Full preparation time and mean clustering time for one run.}
Each cell shows \textbf{preparation / clustering}, in seconds. Preparation is
charged in full, without division by 50, and includes recorded data loading,
preprocessing, representation training, graph construction, and applicable
resolution search. Clustering is the mean of 50 successful per-seed run timers
(seeds 42--91), including loading and saving inside those timers.
Their sum estimates the baseline recorded work required for one clustering run
with preparation. Untimed startup and gaps, queue time, discarded failed work,
and final consolidation of the 50-run batch are excluded. This is an estimate
of the required stages, not a direct fresh end-to-end benchmark.
Original preparation is recovered even when later jobs reuse cached results;
extra retry cache checks are excluded. Shared preparation is charged in full
to each method independently, including its mclust variant, following the
recorded shared-graph workflow. GraphST, STAGATE, SEDR, and SpiceMix train one
representation; BayesSpace fits a 50,000-iteration MCMC chain per clustering run.
""" + hardware_sentence + r"""
Clustering means reflect cached inputs and concurrent workers, not isolated
single-worker or total parallel-job duration. Standalone mclust and the three full Visium HD samples are excluded."""

def latex(rows, text):
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             "\\caption{" + text + "}", r"\label{tab:preparation-mean-clustering}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrrrrrrr}", r"\toprule",
             "Dataset & Spots & " + " & ".join(n.replace(" (mclust)", "") for _, n in METHODS) + r" \\",
             " &  & " + " & ".join("(mclust)" if "mclust" in n else "" for _, n in METHODS) + r" \\"]
    previous = None
    for row in rows:
        if row["Group"] != previous:
            lines.append(r"\midrule")
            if row["Group"] != "Other":
                lines.append(r"\multicolumn{12}{l}{\textbf{" + row["Group"] + r"}} \\")
            previous = row["Group"]
        lines.append(f'{row["Dataset"]} & {row["Spots"]:,} & ' +
                     " & ".join(row[n] for _, n in METHODS) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}", ""])

def html_preview(rows, text):
    lines = ['<!doctype html><meta charset="utf-8"><title>Table 8: preparation / clustering</title>',
             '<style>body{font:14px Arial;margin:24px}p{max-width:1100px;line-height:1.5}'
             'table{border-collapse:collapse}th,td{padding:7px 9px;border-bottom:1px solid #ccc;'
             'text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}</style>',
             '<h1>Full preparation / mean clustering (seconds)</h1>',
             '<p>' + html.escape(text.replace(r"\textbf{", "").replace("}", "")) + '</p>',
             '<table><tr><th>Dataset</th><th>Spots</th>' +
             ''.join('<th>' + html.escape(n) + '</th>' for _, n in METHODS) + '</tr>']
    previous = None
    for row in rows:
        if row['Group'] != previous:
            if row['Group'] != 'Other':
                lines.append('<tr><th colspan="12">' + html.escape(row['Group']) + '</th></tr>')
            previous = row['Group']
        cells = [row['Dataset'], f'{row["Spots"]:,}'] + [row[n] for _, n in METHODS]
        lines.append('<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in cells) + '</tr>')
    return '\n'.join(lines + ['</table>', ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_ROOT / 't8')
    args = parser.parse_args()
    data = DATA_DIR / 't8'
    frame = pd.read_csv(data / 'runtime.csv', float_precision='round_trip')
    samples = pd.read_csv(data / 'samples.csv', dtype={'Dataset': str})
    expected = pd.MultiIndex.from_product([samples.Sample, [key for key, _ in METHODS]])
    observed = pd.MultiIndex.from_frame(frame[['sample', 'method']])
    if observed.has_duplicates or len(observed) != len(expected) or len(expected.difference(observed)):
        raise ValueError('Incomplete or duplicate runtime inputs')
    numbers = frame[['preparation_seconds_once', 'mean_clustering_seconds']].to_numpy()
    if not np.isfinite(numbers).all() or (numbers < 0).any():
        raise ValueError('Invalid runtime measurements')
    indexed = frame.set_index(['sample', 'method'])
    rows = []
    for sample in samples.to_dict('records'):
        row = dict(sample)
        for method, name in METHODS:
            value = indexed.loc[(sample['Sample'], method)]
            row[name] = f'{value.preparation_seconds_once:.2f} / {value.mean_clustering_seconds:.2f}'
        rows.append(row)
    text = caption((data / 'hardware.txt').read_text().strip())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / 'table8_preparation_clustering'
    pd.DataFrame(rows).to_csv(stem.with_suffix('.csv'), index=False)
    stem.with_suffix('.tex').write_text(latex(rows, text))
    stem.with_suffix('.html').write_text(html_preview(rows, text))
    print(f'[saved] {stem}.csv/.tex/.html')


if __name__ == '__main__':
    main()
