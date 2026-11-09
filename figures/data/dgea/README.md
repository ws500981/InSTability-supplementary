# Figure 3 and Supplementary Figures S19–S21

These inputs cover 10 methods, 18 samples, and 50 clustering runs. They contain 1,414 example clusters, 69,286 matches to the other 49 runs, and 5,656 cluster/Top-K DEG summaries.

| File | Contents |
| --- | --- |
| `clusters.csv.gz` | Cluster IDs and their sample, method, and original cluster label. |
| `matches.csv.gz` | Best spatial Jaccard match for each example cluster and target run. |
| `consistency.csv.gz` | DEG Jaccard, shared-gene rank differences, and available-comparison counts for Top-10/20/50/200 genes. |
| `../fig3/reference_quality.csv.gz` | Cluster-level Jaccard, recall, and purity against reference annotations. |
| `../fig3/showcase_alignment.csv.gz` | Matched cluster labels for the human breast cancer SEDR-mclust showcase. |
| `../fig3/showcase_reference.csv.gz` | Showcase matches to reference annotations. |

`cluster_id` joins the tables. `dgea_data.py` derives cluster instability, bins, sizes, and example-run indices from the shared spatial metadata. It calculates cluster mean Jaccard from the saved run-level matches. Those derived values are not stored again.

Empty CSV cells are missing values, not zeros. DEG metrics average available comparisons equally within each example cluster. Rank differences average absolute rank differences over shared genes within a comparison, then across valid comparisons. Counts preserve the denominators when DEG lists or shared genes are unavailable.

`manifest.json` records original workspace-relative source files, columns, row counts, byte sizes, and SHA-256 hashes. Inputs were checked against the saved Figure 3/S19/S20/S21 tables and all 180 completed sample-method outputs. Reconstructed plotting values match the original inputs; normalization produced identical preview pixels for all four figures. No gene expression matrices or gene-level differential-expression tables are included.
