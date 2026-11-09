#!/usr/bin/env python3
"""Format Table 9 from saved instability runtime measurements; never benchmark."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from common import DATA_DIR, OUTPUT_ROOT

METHODS = [
    ("Leiden", "leiden"), ("Louvain", "louvain"), ("BayesSpace", "bayesspace"),
    ("GraphST", "graphst"), ("GraphST (mclust)", "graphst_mclust"),
    ("STAGATE", "stagate"), ("STAGATE (mclust)", "stagate_mclust"),
    ("SpiceMix", "spicemix"), ("SEDR", "sedr"), ("SEDR (mclust)", "sedr_mclust"),
]

TABLE_ROWS = [("DLPFC", section, "dlpfc", section) for section in [
    "151507", "151508", "151509", "151510", "151669", "151670",
    "151671", "151672", "151673", "151674", "151675", "151676",
]] + [
    ("Other", "MB", "mouse_brain", ""),
    ("Other", "HBC", "human_breast_cancer", ""),
    ("Other", "MBC", "mouse_brain_cerebellum", ""),
    ("Cropped Visium HD Samples", "CRC", "Visium_HD_Human_Colon_Cancer_cropped_square", ""),
    ("Cropped Visium HD Samples", "COAD", "coad_ffpe", ""),
    ("Cropped Visium HD Samples", "OV", "ov_ffpe", ""),
]


def require(condition, message):
    if not condition:
        raise ValueError(message)

def build_table(runtime, mode):
    runtime_column = "estimated_total_seconds" if mode == "log" else "end_to_end_seconds"
    rows = []
    for section, label, dataset, sample_name in TABLE_ROWS:
        sample_key = f"{dataset}_{sample_name}" if sample_name else dataset
        row = {"Section": section, "Dataset": label, "dataset-sample": sample_key,
               "# Spots": None, "Runtime measure": runtime_column, "Units": "seconds"}
        for display, method in METHODS:
            hit = runtime.loc[runtime["sample_key"].eq(sample_key) & runtime["method"].eq(method)]
            require(len(hit) == 1, f"{sample_key}/{method}: expected exactly one runtime")
            result = hit.iloc[0]
            n_spots = int(result["n_spots"])
            require(row["# Spots"] is None or row["# Spots"] == n_spots,
                    f"{sample_key}: inconsistent spot counts")
            row["# Spots"] = n_spots
            value = float(result["runtime_seconds"])
            require(np.isfinite(value) and value >= 0, f"{sample_key}/{method}: invalid runtime")
            row[display] = value
        rows.append(row)
    return pd.DataFrame(rows)

def latex_table(table, mode, workers):
    ncolumns = len(METHODS) + 2
    lines, previous_section = [], None
    for _, row in table.iterrows():
        section = row["Section"]
        if section != previous_section:
            if previous_section is not None:
                lines.append(r"\midrule")
            if section != "Other":
                lines.append(rf"\multicolumn{{{ncolumns}}}{{l}}{{\textbf{{{section}}}}} \\")
            previous_section = section
        values = [row["Dataset"], f"{int(row['# Spots']):,}"]
        values.extend("NA" if pd.isna(row[display]) else f"{row[display]:.2f}" for display, _ in METHODS)
        lines.append(" & ".join(values) + r" \\")
    body = "\n".join(lines)
    header = " & ".join(["Dataset", r"\# Spots", *[display for display, _ in METHODS]]) + r" \\"
    specs = (r">{\raggedright\arraybackslash}p{1.4cm}" + "\n"
             + r">{\raggedleft\arraybackslash}p{1.4cm}" + "\n"
             + "\n".join([r">{\raggedleft\arraybackslash}p{1.1cm}"] * len(METHODS)))
    if mode == "log":
        title = "Estimated runtime of instability score calculation from 50 runs"
        description = (
            "Values in seconds are assembled from the existing nested 50-run timing records: "
            "shared loading/encoding time plus 50-run calculation time (including array setup) plus result saving time. "
            "The full loading time, shared across the original run-count analyses, is attributed to the 50-run estimate. "
            "Some untimed setup and output-packaging overhead is excluded. Timings reflect the original parallel production run."
        )
    else:
        title = "Measured runtime of instability score calculation from 50 runs"
        description = (
            "Values in seconds measure a fresh calculation from saved clustering results, including loading, spot alignment, "
            "label encoding, array setup, direct counts, instability calculation, and temporary result serialization. "
            f"Each worker uses one CPU thread; up to {workers} independent sample--method calculations run concurrently. "
            "Worker startup, validation against saved scores, and temporary-file deletion are outside the timing interval. "
            "The detected node hardware and job settings are recorded with the runtime data."
        )
    return rf"""% Requires array, booktabs, and graphicx in the manuscript preamble.
\begin{{table}}[t]
\centering
\scriptsize
\caption[{title}]{{\textbf{{{title}.}} {description}
Standalone mclust and the three full Visium HD samples are excluded.
}}
\label{{tab:instability-runtime-{mode}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{
{specs}
}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['log', 'recompute', 'both'], default='log',
                        help='Select saved timing measurements, not a new timing run.')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_ROOT / 't9')
    args = parser.parse_args()
    modes = ('log', 'recompute') if args.mode == 'both' else (args.mode,)
    metadata = json.loads((DATA_DIR / 't9/benchmark.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        runtime = pd.read_csv(DATA_DIR / 't9' / f'{mode}.csv', float_precision='round_trip')
        require(len(runtime) == 180, 'Expected 18 samples x 10 methods')
        table = build_table(runtime, mode)
        stem = args.output_dir / f'table9_instability_runtime_50runs_{mode}'
        table.to_csv(stem.with_suffix('.csv'), index=False, float_format='%.8f')
        stem.with_suffix('.tex').write_text(latex_table(table, mode, metadata['workers']))
        stem.with_suffix('.html').write_text(table.to_html(index=False, float_format=lambda x: f'{x:.2f}'))
        print(f'[saved] {stem}.csv/.tex/.html')


if __name__ == '__main__':
    main()
