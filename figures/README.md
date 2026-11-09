# Figures and tables

Portable plotting code and minimal inputs for the paper's 10 methods and 18 samples.
Clustering, gene-expression analysis, and instability calculation do not need to be rerun.

We provide code for figures and tables that showcase data; Figure 1 is a motivating example created in Adobe Illustrator, and Tables 1–2 use publicly available information, so their code is not included.

From the repository root:

```bash
conda env create -f figures/environment.yml
conda activate instability-figures
python figures/fig2_s1_s17.py --samples dlpfc_151674
```

Outputs go to `figures/output/`, which Git ignores. To choose another output folder,
set `INSTABILITY_FIGURE_OUTPUT`; relative values are resolved from `figures/`.

| Output | Script |
| --- | --- |
| Figure 2 and S1–S17, one plot per sample | `fig2_s1_s17.py` |
| Figure 3 | `fig3.py` |
| Figure 4 and S28–S36, one plot per method | `fig4_s28_s36.py` |
| S18: BayesSpace entropy | `supp_s18.py` |
| S19: cluster overlap | `supp_s19.py` |
| S20–S21: DEG consistency | `supp_s20_s21.py` |
| S22–S26: expression and annotation agreement | `supp_s22.py`–`supp_s26.py` |
| S27: convergence | `supp_s27.py` |
| S37: spatial instability and reference agreement | `supp_s37.py` |
| Tables 3–7: instability, ARI, and entropy | `table3.py`–`table7.py` |
| Tables 8–9: runtime | `table8.py`, `table9.py` |

Run any script with `python figures/<script>`. Run everything with
`python figures/run_all.py`, or only tables with `python figures/run_all.py --group tables`.
Table 9 supports `--mode log` and `--mode recompute`, using saved measurements in both cases.

The [data notes](data/README.md) describe the retained inputs. Histology is cropped to
the displayed area; the large OV and COAD backgrounds are reduced to 1,600 pixels
per side. Coordinates, labels, and numeric analysis values are preserved.

`environment.yml` records the direct package versions in the environment used for
validation. A fresh environment installation has not been tested. Figure 3 uses
SpatialData for its spatial panels. Matplotlib falls back to DejaVu Sans if Arial
is unavailable.
