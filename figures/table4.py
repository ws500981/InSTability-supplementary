#!/usr/bin/env python3
'Table 4: cluster instability by sample for the benchmark results.'

from common import DATA_DIR, OUTPUT_ROOT
from pathlib import Path

import pandas as pd

from table3 import (
    BASE, METHODS, SAMPLE_ROWS, category_statistics, cluster_fraction_text,
    format_mean_sd, load_benchmark_data, mean_sd, save_supporting_data, tag,
    write_csv, write_text,
)

OUT_DIR = OUTPUT_ROOT / 'tables/t4'
OUT_CSV = OUT_DIR / "table4_by_sample.csv"
OUT_TEX = OUT_DIR / "table4_by_sample.tex"

def latex_table(table):
    lines = []
    for i, row in table.iterrows():
        if i == 6:
            lines += [r"\midrule", r"\multicolumn{4}{l}{\textbf{DLPFC}}\\"]
        cells = [row["Sample"],
                 format_mean_sd(row["mean frac. unstable spots"], row["sd frac. unstable spots"]),
                 str(int(row["# clusters"])), row["fraction of clusters (s/ms/u/hu)"]]
        lines.append(" & ".join(cells) + r"\\")
    body = "\n".join(lines)
    return rf"""% Requires rotating, array, and booktabs in the manuscript preamble.
\begin{{sidewaystable}}[p]
\centering
\scriptsize
\caption{{Cluster instability by sample.}}
\label{{tab:bysample_new}}
\begin{{tabular}}{{
>{{\raggedright\arraybackslash}}m{{1.6cm}}
>{{\centering\arraybackslash}}m{{2cm}}
>{{\centering\arraybackslash}}m{{1.2cm}}
>{{\centering\arraybackslash}}m{{3cm}}
}}
\toprule
Sample & mean frac. unstable spots & \# clusters & fraction of clusters (s/ms/u/hu) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\vspace{{0.3em}}
\parbox{{0.95\textheight}}{{
\scriptsize \textit{{Note.}}
Rows correspond to the 18 benchmark samples.
The second column averages each method's fraction of spots with nested 50-run instability $\geq 0.4$, giving each of the ten methods equal weight; sample standard deviation across methods is shown.
The third column counts clusters from the fixed example run (the first run with the fewest clusters among 50) for each of these ten methods.
The fourth column pools these clusters into $s=$ stable $[0,0.2)$, $ms=$ modestly stable $[0.2,0.4)$, $u=$ unstable $[0.4,0.6)$, and $hu=$ highly unstable $[0.6,\infty)$ using their mean spot instability.
Standalone mclust and the three full Visium HD samples are excluded; the GraphST, STAGATE, and SEDR mclust variants and the cropped samples are retained.
}}
\end{{sidewaystable}}
"""

def main():
    clusters, spot_fractions, provenance = load_benchmark_data()
    rows = []
    for dataset, sample, display in SAMPLE_ROWS:
        sample_tag = tag(dataset, sample)
        group = clusters.loc[clusters["dataset-sample"].eq(sample_tag)]
        fractions = spot_fractions.loc[spot_fractions["dataset-sample"].eq(sample_tag)]
        frac_mean, frac_sd = mean_sd(fractions["frac_unstable"])
        rows.append({
            "Sample": display, "dataset-sample": sample_tag, "n_methods": len(METHODS),
            "mean frac. unstable spots": frac_mean, "sd frac. unstable spots": frac_sd,
            "# clusters": len(group),
            "fraction of clusters (s/ms/u/hu)": cluster_fraction_text(group),
            **category_statistics(group),
        })
    table = pd.DataFrame(rows)
    write_csv(table, OUT_CSV)
    write_text(latex_table(table), OUT_TEX)
    save_supporting_data(OUT_DIR, clusters, spot_fractions, provenance, Path(__file__).resolve(),
                         "Equal-weight method mean and sample SD per sample; cluster statistics pool methods")
    print(table.to_string(index=False))
    print(f"\nSaved table and supporting data to {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
