# Plot inputs

These files contain the inputs needed to regenerate the figures and tables.
They exclude expression matrices, fitted models, raw timing logs, and clustering
labels that are never displayed. CSVs can be read with pandas; `.csv.gz` files
are gzip-compressed CSVs. NPZ files contain numeric arrays and load without pickle.

| Folder | Retained data |
| --- | --- |
| `spatial/` | Spot IDs, annotations, coordinates, 50-run instability, selected example labels, and displayed histology |
| `spot_instability/` | Spot instability at 5, 10, 20, and 40 runs |
| `metrics/` | Mean ARI across runs, mean BayesSpace entropy, and cluster expression summaries |
| `dgea/`, `fig3/` | Cross-run cluster overlap, DEG consistency, and reference comparison values |
| `s18/` | Per-spot entropy of the selected BayesSpace run's posterior membership probabilities |
| `s27/` | Per-sample convergence summaries for nested and sampled runs |
| `t8/`, `t9/` | Saved preparation, clustering, and instability runtimes |

Each spatial H5AD has zero expression variables. It includes one example label
vector per method, selected as the first of 50 runs with the fewest clusters.
The HBC SEDR-mclust labels for all 50 runs are retained because Figure 3 displays
all of them. Example indices are zero-based; clustering seeds are 42–91.

`common.read_spatial_sample()` restores losslessly encoded floating-point images.
`common.histology_kwargs()` places cropped images at their original coordinates
and preserves the original grayscale limits. Histology crops include plot margins;
images larger than 1,600 pixels are resized for plotting. Only OV and COAD require
this reduction. Full source images and the HBC Figure 3 background are otherwise
not altered in the analysis workspace.

Sample mean instability, cluster mean instability, cluster sizes, annotation
overlap, unstable-spot fractions, and other repeated summaries are calculated
from these shared inputs. Expression summaries and DEG comparisons require the
original expression data, so their compact results are retained. Mean ARI over
all runs requires undisplayed labels and is retained as a summary.

The scripts preserve their original annotation conventions: Figure 3 reference
matching treats missing annotations as an `NA` category; Figure 4, S24–S26, and
Table 3 calculate annotation agreement using annotated spots only.

S18 entropy is computed within a selected run from soft membership probabilities;
hard cluster labels cannot reconstruct it. Its compact values are smaller than
the probability matrices. The mean entropy used in S37 and Table 7 averages
entropy across all 50 runs and is a separate required input.

Source records and checksums are included with the relevant data folders.
The runtime tables include the same 18 samples; their notes retain the original
benchmark conditions, including concurrency, even where other samples ran in
the same benchmark job.
