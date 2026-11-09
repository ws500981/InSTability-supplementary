# Shared metrics

`sample_metrics.csv` retains the saved 5-, 10-, 20-, 40-, and 50-run mean ARI and
BayesSpace mean entropy.
Sample mean instability is calculated from the bundled spot scores by
`sample_metrics.load_sample_metrics()`.

`clusterwise_metrics.csv` retains expression coherence and variance once per
example cluster.
`cluster_metrics.load_cluster_metrics()` calculates instability, sizes, and annotation
agreement for the requested run counts from the shared scores and labels.

`sources.json` records source and packaged-file checksums. Numerical comparisons
matched all 90 sample rows and all 7,070 reconstructed cluster rows; the largest
cluster difference was 1.12e-16 from floating-point rounding.
