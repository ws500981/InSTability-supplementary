#!/usr/bin/env python3
'Table 6: nested 50-run mean spot instability from the shared sample metrics.'

from common import DATA_DIR, OUTPUT_ROOT
from pathlib import Path

from table5 import BASE, generate_matrix

OUT_DIR = OUTPUT_ROOT / 'tables/t6'
OUTPUT_STEM = "table6_mean_instability_sample_method"
SUMMARY_STEM = "table6_mean_instability_method_summary"

def main():
    generate_matrix(
        "mean_instability", OUT_DIR, OUTPUT_STEM, SUMMARY_STEM,
        "Mean instability (the nested 50-run instability score averaged across all spots) "
        "for each sample--method pair. Each of ten methods is evaluated on the same 18 samples. "
        "Summary values are means and sample standard deviations.",
        "tab:mean_instability_sample_method", Path(__file__).resolve(),
    )

if __name__ == "__main__":
    main()
