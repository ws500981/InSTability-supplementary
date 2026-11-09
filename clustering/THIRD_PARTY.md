# External methods

The SpiceMix implementation belongs to the authors of
[ma-compbio/SpiceMix](https://github.com/ma-compbio/SpiceMix); it is not our
contribution and is not distributed in this repository. Our `spicemix.py` and
`spicemix_prepare_input.py` implement the paper's experiment and input preparation.

## SpiceMix setup

After creating the [SpiceMix environment](environments/README.md), run these
commands from a fresh clone of the supplementary repository. The dependency is
placed in `not-included/`, which is excluded from Git:

```bash
git clone https://github.com/ma-compbio/SpiceMix.git not-included/SpiceMix
git -C not-included/SpiceMix checkout --detach aea69f8081f33e8ca64b734c680804db907ed4eb
export SPICEMIX_SOURCE_DIR=not-included/SpiceMix/SpiceMix
conda activate inst-spicemix
export PYTHONNOUSERSITE=1
python -u clustering/spicemix_prepare_input.py
python -u clustering/spicemix.py
```

An existing matching checkout or the original workspace's local source copies
can be used without cloning again. `SPICEMIX_SOURCE_DIR` must point to the subdirectory containing
`model.py` and `util.py`. Relative paths are resolved from the supplementary
repository root; the default is `not-included/SpiceMix/SpiceMix`. No `PYTHONPATH` setting
or `pip install` of SpiceMix is needed.

The checkout also supplies `load_data.py`, `initialization.py`,
`estimate_parameters.py`, `estimate_weights.py`, and `sample_for_integral.py`.
Keep these upstream files together. `spicemix_external.py` only locates the
checkout and adapts the upstream loader's removed `np.float`/`np.int` aliases
to their equivalent built-in types. It contains no SpiceMix model code and
does not edit the external checkout. The upstream repository carries its own
MIT license.

GraphST, STAGATE, SEDR, and BayesSpace are installed separately using the
[environment instructions](environments/README.md).
