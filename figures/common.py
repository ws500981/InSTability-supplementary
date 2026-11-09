"""Shared locations and readers for the figure data."""

import os
from pathlib import Path

import anndata as ad
import numpy as np

FIGURE_ROOT = Path(__file__).resolve().parent
DATA_DIR = FIGURE_ROOT / "data"
OUTPUT_ROOT = Path(os.environ.get("INSTABILITY_FIGURE_OUTPUT", FIGURE_ROOT / "output"))
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = FIGURE_ROOT / OUTPUT_ROOT


def read_spatial_sample(tag):
    """Read metadata and restore losslessly encoded histology images."""
    if Path(tag).name != tag:
        raise ValueError("Expected a sample tag, not a path")
    data = ad.read_h5ad(DATA_DIR / "spatial" / f"{tag}.h5ad")
    for library in data.uns.get("spatial", {}).values():
        for key, dtype in library.pop("image_encoding", {}).items():
            library["images"][key] = library["images"][key].astype(np.dtype(dtype)) / 255
    return data


def histology_kwargs(data):
    """Place a cropped image in its original coordinates and preserve contrast."""
    for library in data.uns.get("spatial", {}).values():
        kwargs = {}
        if "image_extent" in library:
            kwargs["extent"] = library["image_extent"]
        if "image_gray_limits" in library:
            kwargs["vmin"], kwargs["vmax"] = library["image_gray_limits"]
        return kwargs
    return {}
