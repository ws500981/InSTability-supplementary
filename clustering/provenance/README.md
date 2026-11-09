# Source verification

Audited 2026-11-09 for the 10 methods and 18 samples selected for publication.

- All 180 final result files match the byte-size and modification-time signatures saved by the instability analysis.
- Successful run logs contain exactly seeds 42–91 for every result: 9,000 runs. Each final file matches its consolidation event. All 18 BayesSpace CSVs also have the expected spot counts and 50 seed columns.
- Recorded scientific settings agree with the current sources where metadata is available. All 18 GraphST caches record the current `functions.py` hash; SpiceMix input exports match all 18 preparation logs.
- Current `spicemix.py` exactly matches its archived submitted copy and SHA256 record for the three-sample November rerun. Exact historical source identity cannot be established for the other runners.

The current original Leiden/Louvain runner uses `logs/`, while the saved run logs are under `run_logs/`. Its historical logs report Scanpy 1.10.3.

`results_audit.csv` lists result hashes, source signatures and matched run evidence. `source_verification.json` records original source hashes, parameter checks and verification limits. Paths in these records are relative to the original `outs_review` workspace; raw results and logs are not included here.

No clustering was rerun. Pickle/joblib arrays were not deserialized; their identities and run coverage were checked from file signatures and logs. Source hashes refer to originals before publication edits.
