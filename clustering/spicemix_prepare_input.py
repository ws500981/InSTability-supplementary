#!/usr/bin/env python3
"Prepare normalized expression, spatial edges, and metadata for SpiceMix."

from paths import DATA_ROOT as INPUT_DATA_ROOT, OUTPUT_ROOT

import os
import time

PROGRAM_START = time.perf_counter()
for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(name, "1")

import json
import platform
import resource
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

from functions import get_raw_counts

DATA_ROOT = INPUT_DATA_ROOT
SPICEMIX_ROOT = OUTPUT_ROOT
LIBRARY_SIZE = 1e4
MIN_NONZERO_FRACTION = 0.10  # proportion of observations; same rule for all samples
SPATIAL_K_VISIUM = 6
SPATIAL_K_VISIUM_HD = 8
SPATIAL_K_SLIDE_SEQ = 10
SMOOTH_RADIUS = 2.001  # SpiceMix's radius=2, radius_eps=0.001
RUN_ID = os.environ.setdefault(
    "SPICEMIX_PREP_RUN_ID",
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}",
)
LOG_DIR = SPICEMIX_ROOT / "run_logs" / "spicemix_prepare_input" / RUN_ID
RECORDS = []


def hardware_info():
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "run_id": RUN_ID,
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "available_cpus": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "total_node_ram_gib": round(
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30, 2
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node_mb": os.environ.get("SLURM_MEM_PER_NODE"),
        "slurm_mem_per_cpu_mb": os.environ.get("SLURM_MEM_PER_CPU"),
        "numpy_version": np.__version__,
        "anndata_version": ad.__version__,
    }


@contextmanager
def timed(stage, sample="all", started=None):
    started = time.perf_counter() if started is None else started
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        row = {
            "run_id": RUN_ID,
            "sample": sample,
            "stage": stage,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "device": "cpu",
            "status": status,
            "process_peak_rss_gib": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3
            ),
        }
        RECORDS.append(row)
        (LOG_DIR / f"{sample}_{stage}.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"TIME {sample} {stage}: {row['elapsed_seconds']:.2f}s ({status})",
            flush=True,
        )


def _get_expression_matrix(
    adata, library_size=LIBRARY_SIZE, min_nonzero_fraction=MIN_NONZERO_FRACTION
):
    "Verified counts -> sample-specific gene selection -> total-normalized log1p."
    C, source = get_raw_counts(adata, check_values=True)
    print(f"SpiceMix raw-count source: {source}")
    if sparse.issparse(C):
        nonzero_fraction = np.asarray(C.getnnz(axis=0)).ravel() / C.shape[0]
    else:
        nonzero_fraction = (np.asarray(C) > 0).mean(axis=0)
    keep = nonzero_fraction >= float(min_nonzero_fraction)
    if not np.any(keep):
        raise ValueError("Gene filtering removed every gene")
    genes = adata.var_names[keep].astype(str).tolist()
    C = C[:, keep]
    print(
        f"SpiceMix genes retained: {len(genes):,}/{adata.n_vars:,} "
        f"(min_nonzero_fraction={min_nonzero_fraction:g})"
    )

    if sparse.issparse(C):
        C = C.tocsr().astype(np.float32)
        totals = np.asarray(C.sum(axis=1)).ravel()
        if np.any(totals <= 0):
            raise ValueError("Some observations have zero counts after gene filtering")
        C_norm = C.multiply((float(library_size) / totals)[:, None]).tocsr()
        C_norm.data = np.log1p(C_norm.data)
        X = C_norm.toarray()
    else:
        C = np.asarray(C, dtype=np.float32)
        totals = C.sum(axis=1)
        if np.any(totals <= 0):
            raise ValueError("Some observations have zero counts after gene filtering")
        X = np.log1p((C / totals[:, None]) * float(library_size))
    return np.asarray(X, dtype=np.float32), genes


def _build_knn_edges(coords, n_neighbors=SPATIAL_K_VISIUM):
    "Zero-based, undirected union of Euclidean kNN edges (model graph)."
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("Spatial graph needs finite Nx2 coordinates")
    n_obs = len(coords)
    if not 1 <= int(n_neighbors) < n_obs:
        raise ValueError(f"Expected 1 <= k < {n_obs}, got {n_neighbors}")
    nn = NearestNeighbors(n_neighbors=int(n_neighbors) + 1, metric="euclidean")
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    edges = set()
    for i, row in enumerate(indices):
        # Take exactly k non-self neighbors, including coincident positions.

        other = [int(j) for j in row if j != i][: int(n_neighbors)]
        if len(other) != int(n_neighbors):
            raise RuntimeError(
                f"Observation {i} has fewer than {n_neighbors} KNN entries"
            )
        for j in other:
            edges.add(tuple(sorted((i, j))))
    return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)


def _scaled_smoothing_coordinates(coords):
    "Scale each sample to median nearest DISTINCT position = one unit."
    xy = np.asarray(coords, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("SpiceMix smoothing requires finite Nx2 coordinates")
    unique_xy = np.unique(xy, axis=0)
    if len(unique_xy) < 2:
        raise ValueError("Fewer than two distinct spatial positions")

    nearest_distinct = cKDTree(unique_xy).query(xy, k=2)[0][:, 1]
    spacing = float(np.median(nearest_distinct))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Cannot infer a positive median nearest-position spacing")
    normalized = (xy - xy.min(axis=0)) / spacing
    neighbors = (
        cKDTree(normalized).query_ball_point(
            normalized, r=SMOOTH_RADIUS, return_length=True
        )
        - 1
    )
    return normalized, {
        "original_nearest_distinct_spacing": spacing,
        "smoothing_radius_in_spacing_units": SMOOTH_RADIUS,
        "mean_other_observations_in_smoothing_radius": float(np.mean(neighbors)),
        "median_other_observations_in_smoothing_radius": float(np.median(neighbors)),
        "fraction_without_other_observations": float(np.mean(neighbors == 0)),
        "repeated_coordinate_entries": int(len(xy) - len(unique_xy)),
    }


def export_h5ad_to_spicemix_files(
    path_h5ad,
    out_dir,
    fov_name,
    expr_suffix="",
    neigh_suffix="",
    library_size=LIBRARY_SIZE,
    min_nonzero_fraction=MIN_NONZERO_FRACTION,
    n_neighbors=SPATIAL_K_VISIUM,
    neighbor_from="spatial",
):
    if neighbor_from not in {"spatial", "connectivities"}:
        raise ValueError("neighbor_from must be 'spatial' or 'connectivities'")
    out_dir = Path(out_dir)
    files_dir = out_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    expr_suffix_str = f"_{expr_suffix}" if expr_suffix else ""
    neigh_suffix_str = f"_{neigh_suffix}" if neigh_suffix else ""

    for candidate in (
        files_dir / f"expression_{fov_name}{expr_suffix_str}.pkl",
        files_dir / f"meta_{fov_name}.pkl",
    ):
        if candidate.exists():
            raise FileExistsError(
                f"Remove or reconcile older SpiceMix input: {candidate}"
            )

    with timed("read_input", fov_name):
        adata = ad.read_h5ad(path_h5ad)
    n_obs = adata.n_obs
    with timed("expression_preprocessing", fov_name):
        X, genes = _get_expression_matrix(
            adata, library_size=library_size, min_nonzero_fraction=min_nonzero_fraction
        )
        if X.shape != (n_obs, len(genes)):
            raise RuntimeError("Expression matrix/genes/observations do not align")
    with timed("expression_export", fov_name):
        np.savetxt(
            files_dir / f"expression_{fov_name}{expr_suffix_str}.txt",
            X,
            fmt="%.8g",
            delimiter="\t",
        )
        with (files_dir / f"genes_{fov_name}{expr_suffix_str}.txt").open("w") as handle:
            for gene in genes:
                handle.write(f"{gene}\n")
    del X

    with timed("spatial_graph_export", fov_name):
        if neighbor_from == "connectivities":
            if "connectivities" not in adata.obsp:
                raise KeyError("adata.obsp['connectivities'] is missing")
            A = adata.obsp["connectivities"]
            if not sparse.issparse(A):
                A = sparse.csr_matrix(A)
            rows, cols = A.nonzero()
            edges = np.asarray(
                sorted(
                    {
                        tuple(sorted((int(i), int(j))))
                        for i, j in zip(rows, cols)
                        if i != j
                    }
                ),
                dtype=np.int64,
            ).reshape(-1, 2)
        else:
            if "spatial" not in adata.obsm:
                raise KeyError("adata.obsm['spatial'] is missing")
            edges = _build_knn_edges(adata.obsm["spatial"], n_neighbors=n_neighbors)
        if not len(edges):
            raise ValueError("No spatial edges were generated")
        edge_xy = np.asarray(adata.obsm["spatial"], dtype=float)
        zero_length_edges = int(
            np.count_nonzero(
                np.all(edge_xy[edges[:, 0]] == edge_xy[edges[:, 1]], axis=1)
            )
        )
        graph_qc = {
            "spatial_graph_type": (
                "undirected_union_of_euclidean_knn"
                if neighbor_from == "spatial"
                else "undirected_connectivities"
            ),
            "requested_knn_neighbors": int(n_neighbors)
            if neighbor_from == "spatial"
            else None,
            "spatial_edge_count": int(len(edges)),
            "mean_spatial_degree": float(2 * len(edges) / n_obs),
            "zero_distance_spatial_edges": zero_length_edges,
        }
        np.savetxt(
            files_dir / f"neighborhood_{fov_name}{neigh_suffix_str}.txt",
            edges,
            fmt="%d",
            delimiter="\t",
        )

    with timed("metadata_export", fov_name):
        meta = adata.obs.copy()
        if "spatial" not in adata.obsm:
            raise KeyError("Smoothing requires adata.obsm['spatial']")
        smooth_xy, qc = _scaled_smoothing_coordinates(adata.obsm["spatial"])
        qc.update(graph_qc)
        qc.update(
            {
                "n_observations": int(n_obs),
                "n_genes_retained": int(len(genes)),
                "gene_filter_min_nonzero_fraction": float(min_nonzero_fraction),
                "normalization_library_size": float(library_size),
            }
        )
        meta["coor X"] = smooth_xy[:, 0]
        meta["coor Y"] = smooth_xy[:, 1]
        meta.to_csv(files_dir / f"meta_{fov_name}.csv", index=False)
        (files_dir / f"spicemix_coordinate_qc_{fov_name}.json").write_text(
            json.dumps(qc, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[{fov_name}] spatial graph and smoothing coordinate QC: {qc}")
    print(
        f"[{fov_name}] {n_obs:,} observations, {len(genes):,} genes, "
        f"{len(edges):,} undirected spatial edges -> {files_dir}"
    )


DATASETS = [
    "human_breast_cancer",
    "mouse_brain",
    "mouse_brain_cerebellum",
    "Visium_HD_Human_Colon_Cancer_cropped_square",
    "coad_ffpe",
    "ov_ffpe",
    "dlpfc_151507",
    "dlpfc_151508",
    "dlpfc_151509",
    "dlpfc_151510",
    "dlpfc_151669",
    "dlpfc_151670",
    "dlpfc_151671",
    "dlpfc_151672",
    "dlpfc_151673",
    "dlpfc_151674",
    "dlpfc_151675",
    "dlpfc_151676",
]


def spatial_k_for_dataset(dataset):
    "Select a documented graph rule by technology, never by cluster labels."
    if dataset == "mouse_brain_cerebellum":
        return SPATIAL_K_SLIDE_SEQ
    if dataset in {
        "Visium_HD_Human_Colon_Cancer_cropped_square",
        "coad_ffpe",
        "ov_ffpe",
    }:
        return SPATIAL_K_VISIUM_HD
    if dataset in {"human_breast_cancer", "mouse_brain"} or dataset.startswith(
        "dlpfc_"
    ):
        return SPATIAL_K_VISIUM
    raise ValueError(f"No spatial graph rule for {dataset!r}")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "hardware.json").write_text(
        json.dumps(hardware_info(), indent=2) + "\n", encoding="utf-8"
    )
    print("Preparation hardware and timing logs:", LOG_DIR)
    for dataset in DATASETS:
        spatial_k = spatial_k_for_dataset(dataset)
        print(
            f"\nPreparing {dataset}: min_nonzero_fraction="
            f"{MIN_NONZERO_FRACTION:g}, spatial_k={spatial_k}"
        )
        if dataset.startswith("dlpfc_"):
            out_dir = SPICEMIX_ROOT / "dlpfc" / "spicemix" / dataset.split("_")[-1]
        else:
            out_dir = SPICEMIX_ROOT / dataset / "spicemix"
        with timed("total_sample", dataset):
            export_h5ad_to_spicemix_files(
                path_h5ad=DATA_ROOT / f"{dataset}_filtered.h5ad",
                out_dir=out_dir,
                fov_name=dataset,
                library_size=LIBRARY_SIZE,
                min_nonzero_fraction=MIN_NONZERO_FRACTION,
                n_neighbors=spatial_k,
                neighbor_from="spatial",
            )


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with timed("total_wall", started=PROGRAM_START):
            main()
    finally:
        pd.DataFrame(RECORDS).to_csv(LOG_DIR / "timings.csv", index=False)
        print("Preparation timing summary:", LOG_DIR / "timings.csv", flush=True)
