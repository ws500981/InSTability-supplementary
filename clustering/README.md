# Clustering

The paper uses 10 methods on 18 samples, with 50 runs (seeds 42–91) each.
Preprocessing and learned representations are fixed across downstream
clustering runs; BayesSpace instead repeats independent MCMC chains.

| Script | Methods | Environment |
|---|---|---|
| `leiden_louvain.py` | Leiden, Louvain | `inst-leiden-louvain` |
| `graphst.py` | GraphST, GraphST + mclust | `inst-graphst` |
| `stagate.py` | STAGATE, STAGATE + mclust | `inst-stagate` |
| `sedr.py` | SEDR, SEDR + mclust | `inst-sedr` |
| `spicemix.py` | SpiceMix | `inst-spicemix` |
| `bayesspace.R` | BayesSpace | R; see environment instructions |

## Inputs

Place the 18 prepared `*_filtered.h5ad` files listed in [samples.tsv](samples.tsv)
in the repository's `data/` directory. These datasets are not bundled here.
They must contain raw counts in `layers['counts']`, `raw.X`, or `X`,
reference labels in `obs['annotation']`, and spatial coordinates in
`obsm['spatial']`. BayesSpace also needs lattice coordinates
(`array_row`/`array_col`, or the supported row/column or bin-ID equivalents)
for Visium and Visium HD samples. The cerebellum uses continuous coordinates.

## Run

Run commands from the repository root. Create the method-specific environments
using [environments/README.md](environments/README.md); it also covers R/mclust
and the Leiden/Louvain Scanpy version.

For example:

```bash
conda env create -f clustering/environments/graphst.yml
conda activate inst-graphst
export PYTHONNOUSERSITE=1
Rscript --vanilla clustering/environments/install_r_packages.R
python -u clustering/graphst.py
```

Use the matching script and environment from the table for the other methods.
SpiceMix also requires a separate upstream checkout. Follow
[the clone and setup instructions](THIRD_PARTY.md#spicemix-setup), then prepare
its inputs in the same environment:

```bash
python -u clustering/spicemix_prepare_input.py
python -u clustering/spicemix.py
```

After the R setup, run BayesSpace with `Rscript --vanilla clustering/bayesspace.R`.
Each runner processes all 18 samples and retains the original scientific settings.

Inputs default to `data/` and outputs to `results/`, both relative to the
repository, independent of the working directory. Override them if needed:

```bash
export INSTABILITY_DATA_ROOT=data
export INSTABILITY_OUTPUT_ROOT=results
```

Relative overrides are also resolved from the repository root. Outputs include
cluster labels, saved representations, and run metadata. SpiceMix preparation
writes to the same output tree that its runner reads. Data and generated results
are excluded from Git. The supplied environments target Linux; GPU methods use
CUDA when available and otherwise run on CPU.

External method attribution and SpiceMix setup are in [THIRD_PARTY.md](THIRD_PARTY.md).
