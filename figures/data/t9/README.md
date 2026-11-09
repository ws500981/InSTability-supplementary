Saved inputs for Table 9 (original analysis Table 7), restricted to 18 samples
and 10 methods. Both modes format recorded measurements; neither runs instability.

- `log.csv`: shared loading/encoding + 50-run calculation (including array setup)
  + result saving, with full shared loading attributed to the 50-run estimate.
- `recompute.csv`: recorded end-to-end benchmark from saved clustering labels,
  including alignment, encoding, array setup, calculation, and serialization.
  Validation, worker startup, and temporary-file deletion were outside timing.

`benchmark.json` records hardware and original concurrency for the recomputed
measurements. Its 186-task benchmark also included six full-sample tasks, which
are excluded from this publication table. Untimed overhead is excluded from the
log estimate; the two modes are distinct measurements.
