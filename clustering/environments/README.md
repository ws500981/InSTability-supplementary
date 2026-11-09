# Environments

Create one environment per runner from the repository root:

```bash
conda env create -f clustering/environments/graphst.yml
conda activate inst-graphst
```

| Runner | YAML | Environment |
| --- | --- | --- |
| Leiden / Louvain | `leiden_louvain.yml` | `inst-leiden-louvain` |
| GraphST / GraphST + mclust | `graphst.yml` | `inst-graphst` |
| STAGATE / STAGATE + mclust | `stagate.yml` | `inst-stagate` |
| SEDR / SEDR + mclust | `sedr.yml` | `inst-sedr` |
| SpiceMix and input preparation | `spicemix.yml` | `inst-spicemix` |

The SpiceMix YAML installs its dependencies. Obtain the implementation separately
using [the pinned upstream checkout](../THIRD_PARTY.md#spicemix-setup).

These are Linux x86_64 environments. GPU runs require an NVIDIA GPU and a driver compatible with CUDA 12.8 (GraphST/SEDR) or CUDA 12.1 (STAGATE/SpiceMix). The CUDA 12.1 YAML files include the official [PyTorch wheel index](https://pytorch.org/get-started/previous-versions/) and [PyG wheel source](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). Git is required to install the pinned STAGATE and SEDR sources. The full exports include packages that were present but are not all required by these runners.

For GraphST, STAGATE and SEDR, also install the R package used by their mclust step into the active environment:

```bash
Rscript --vanilla clustering/environments/install_r_packages.R
```

This installs [mclust 6.1.2 from the CRAN archive](https://cran.r-project.org/src/contrib/Archive/mclust/). A working C/Fortran toolchain is needed for source installation. Do not point `R_HOME` or `R_LIBS_USER` at another machine's R installation. Verify the Python/R bridge after setup:

```bash
python -c "import rpy2.robjects as r; print(r.r('R.version.string')); print(r.r('as.character(packageVersion(\"mclust\"))'))"
```

For BayesSpace, use a separate R 4.5.2 environment and Bioconductor 3.22:

```bash
conda create -n inst-bayesspace -c conda-forge r-base=4.5.2 r-biocmanager c-compiler cxx-compiler fortran-compiler make
conda activate inst-bayesspace
Rscript --vanilla clustering/environments/install_r_packages.R --bayesspace
```

The setup installs `clue` and checks for BayesSpace 1.20.2, which is available in the [Bioconductor 3.22 archive](https://bioconductor.posit.co/packages/3.22/bioc/html/BayesSpace.html). An exact source-package URL is provided as a fallback. Bioconductor dependencies are resolved from release 3.22; this is a setup recipe, not a complete historical R lockfile. Package builds used for the saved analysis are recorded in `r_packages.tsv`. The BayesSpace runner uses zellkonverter's native R reader for H5AD files.
ata records an R 4.5.2 build. The original BayesSpace launcher loaded an unversioned system R module, so the exact historical R runtime cannot be established from that launcher alone.

Environment files were checked for syntax and personal paths. No environments were installed or modified, and fresh installation/full clustering runs have not been tested.
