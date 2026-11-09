#!/usr/bin/env python3
'Table 7: BayesSpace entropy versus nested 50-run mean spot instability.'

from common import DATA_DIR, OUTPUT_ROOT
from pathlib import Path

import pandas as pd

from table5 import BASE, load_metrics, save_provenance, write_csv, write_text

OUT_DIR = OUTPUT_ROOT / 'tables/t7'
OUTPUT_STEM = "table7_bayesspace_entropy_instability"

def latex_table(table):
    lines = []
    for i, row in table.iterrows():
        if i == 6:
            lines += [r"\midrule", r"\multicolumn{3}{l}{\textbf{DLPFC}}\\"]
        lines.append(f"{row['Sample']} & {row['Mean Instability']:.2f} & {row['Mean Entropy']:.2f} " + r"\\")
    body = "\n".join(lines)
    return rf"""% Requires booktabs in the manuscript preamble.
\begin{{table}}[p]
\centering
\small
\caption[BayesSpace Entropy vs. Instability]{{\textbf{{BayesSpace Entropy vs. Instability.}}
Mean instability is the nested 50-run instability score averaged across all spots in the sample.
Mean entropy is Shannon entropy in nats averaged across all spots in each run, then averaged across all 50 BayesSpace runs.
The 18 benchmark samples are included; the three full Visium HD samples are excluded.
}}
\label{{tab:bayesspace_entropy_ari_instability}}
\setlength{{\tabcolsep}}{{7pt}}
\begin{{tabular}}{{lcc}}
\toprule
Sample & Mean Instability & Mean Entropy \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""

def main():
    columns = ["bayesspace_mean_instability", "bayesspace_mean_entropy_all_runs"]
    frame, provenance = load_metrics(columns)
    table = pd.DataFrame({
        "Sample": frame["sample_label"],
        "Mean Instability": frame[columns[0]],
        "Mean Entropy": frame[columns[1]],
    })
    write_csv(table, OUT_DIR / f"{OUTPUT_STEM}.csv")
    write_text(latex_table(table), OUT_DIR / f"{OUTPUT_STEM}.tex")
    write_csv(frame, OUT_DIR / "source_metrics_50runs.csv")
    save_provenance(dict(provenance, methods=["bayesspace"]), OUT_DIR, Path(__file__).resolve(),
                    "Nested 50-run mean spot instability; mean spot Shannon entropy (nats) averaged over all 50 runs")
    print(table.to_string(index=False))
    print(f"\nSaved table, source metrics, and provenance to {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
