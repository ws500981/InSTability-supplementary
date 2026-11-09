S18 uses one saved per-spot BayesSpace entropy array per sample, plus the shared
spatial metadata in `../spatial/`. The arrays contain natural-log Shannon
entropy, `-sum(p * log(p))`, in nats, with zero-probability terms set to zero.
Entropy is not divided by log(number of clusters). It describes the selected
run's posterior memberships; hard labels across runs cannot recover it.

The example run is the first of seeds 42–91 with the fewest distinct clusters.
For these 18 BayesSpace samples this is seed 42. Entropy was extracted from that
seed's saved membership probabilities after validating sample, seed, spot order,
hard labels, and row sums. Plotting requires neither R nor membership RDS files.
A spot-order SHA256 digest enforces alignment with the shared spatial metadata
without duplicating its spot IDs.

`sources.json` records analysis-relative membership paths and checksums.
