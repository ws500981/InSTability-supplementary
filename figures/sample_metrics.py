"""Combine saved agreement metrics with means calculated from spot scores."""

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

from common import DATA_DIR


def load_sample_metrics():
    frame = pd.read_csv(DATA_DIR / "metrics/sample_metrics.csv", dtype={"sample": str})
    frame["sample"] = frame["sample"].fillna("")
    frame["example_mode"] = "example_50"
    methods = [c.removesuffix("_mean_ari_all_runs") for c in frame if c.endswith("_mean_ari_all_runs")]
    for (dataset, sample), group in frame.groupby(["dataset", "sample"], sort=False):
        tag = f"{dataset}_{sample}" if sample else dataset
        with h5py.File(DATA_DIR / "spatial" / f"{tag}.h5ad", "r") as source:
            obs = read_elem(source["obs"])
        for method in methods:
            with np.load(DATA_DIR / "spot_instability" / f"{tag}__{method}.npz", allow_pickle=False) as scores:
                for index, row in group.iterrows():
                    k = int(row["iterations"])
                    values = obs[f"{method}_uncertainty"].to_numpy(float) if k == 50 else scores[f"scores_{k}"].astype(float)
                    frame.loc[index, f"{method}_mean_instability"] = round(float(values.mean()), 8)
    return frame
