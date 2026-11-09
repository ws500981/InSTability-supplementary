#!/usr/bin/env python3
"Repeat Leiden and Louvain clustering on one fixed graph per sample."

from paths import DATA_ROOT as INPUT_DATA_ROOT, OUTPUT_ROOT

import os

for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_thread_var] = "1"
import gc
import hashlib
import importlib.metadata
import inspect
import json
import platform
import resource
import socket
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from functions import (
    get_n_neighbors,
    get_target_n_clusters,
    preprocessing,
    search_res,
)

DATA_ROOT = INPUT_DATA_ROOT
OUT_ROOT = OUTPUT_ROOT

ITERATIONS = 50
BASE_SEED = 42
ANNOTATION_KEY = "annotation"

N_HVG = 3000
HVG_FLAVOR = "seurat_v3"
NORMALIZE_TARGET_SUM = 1e4
LOG1P = True  # preprocessing() applies log1p after normalize_total
SCALE = True
ZERO_CENTER = True
SCALE_MAX_VALUE = 10

N_PCS = 50
PCA_SOLVER = "arpack"
PCA_SEED = 42

DEFAULT_N_NEIGHBORS = 15
KNN_METRIC = "euclidean"
CONNECTIVITY_METHOD = "umap"  # connectivity kernel, NOT a UMAP embedding
GRAPH_SEED = 42

USE_DATASET_SPECIFIC_N_NEIGHBORS = False
N_NEIGHBORS_BY_DATASET = {}

LEIDEN_USE_WEIGHTS = True
LEIDEN_DIRECTED = False
LOUVAIN_USE_WEIGHTS = True
LOUVAIN_DIRECTED = False
LEIDEN_N_ITERATIONS = -1
LEIDEN_FLAVOR = "leidenalg"
LOUVAIN_FLAVOR = "vtraag"

RES_START = 0.02
RES_END = 4.0
RES_FINE_STEP = 0.01
RES_TOLERANCE = 2
RESOLUTION_SEED = 42
RESOLUTION_OVERRIDES = {}

_affinity = (
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else (os.cpu_count() or 1)
)
_slurm = os.environ.get("SLURM_CPUS_PER_TASK")
ALLOCATED_CPUS = min(_affinity, int(_slurm)) if _slurm else _affinity
if ALLOCATED_CPUS < 1:
    raise ValueError("No CPUs allocated for Leiden/Louvain.")


def _jobs(name: str, default: int) -> int:
    jobs = int(os.environ.get(name, str(min(default, ALLOCATED_CPUS))))
    if not 1 <= jobs <= ALLOCATED_CPUS:
        raise ValueError(f"{name} must be between 1 and {ALLOCATED_CPUS}.")
    return jobs


GRAPH_BUILD_JOBS = _jobs("GRAPH_BUILD_JOBS", 10)
RESOLUTION_JOBS = _jobs("RESOLUTION_JOBS", 100)
CLUSTER_JOBS = _jobs("CLUSTER_JOBS", 70)

FORCE_REBUILD_GRAPH = False


@dataclass(frozen=True)
class SampleSpec:
    key: str
    h5ad_name: str
    output_group: str


SAMPLES = [
    SampleSpec(
        "Visium_HD_Human_Colon_Cancer_cropped_square",
        "Visium_HD_Human_Colon_Cancer_cropped_square_filtered.h5ad",
        "Visium_HD_Human_Colon_Cancer_cropped_square",
    ),
    SampleSpec("mouse_brain", "mouse_brain_filtered.h5ad", "mouse_brain"),
    SampleSpec(
        "mouse_brain_cerebellum",
        "mouse_brain_cerebellum_filtered.h5ad",
        "mouse_brain_cerebellum",
    ),
    SampleSpec(
        "human_breast_cancer",
        "human_breast_cancer_filtered.h5ad",
        "human_breast_cancer",
    ),
    SampleSpec("ov_ffpe", "ov_ffpe_filtered.h5ad", "ov_ffpe"),
    SampleSpec("coad_ffpe", "coad_ffpe_filtered.h5ad", "coad_ffpe"),
    SampleSpec("dlpfc_151507", "dlpfc_151507_filtered.h5ad", "dlpfc"),
    SampleSpec("dlpfc_151508", "dlpfc_151508_filtered.h5ad", "dlpfc"),
    SampleSpec("dlpfc_151509", "dlpfc_151509_filtered.h5ad", "dlpfc"),
    SampleSpec("dlpfc_151510", "dlpfc_151510_filtered.h5ad", "dlpfc"),
    SampleSpec("dlpfc_151669", "dlpfc_151669_filtered.h5ad", "dlpfc"),
    SampleSpec("dlpfc_151670", "dlpfc_151670_filtered.h5ad", "dlpfc"),
    SampleSpec("dlpfc_151671", "dlpfc_151671_filtered.h5ad", "dlpfc"),
    SampleSpec("dlpfc_151672", "dlpfc_151672_filtered.h5ad", "dlpfc"),
    SampleSpec("dlpfc_151673", "dlpfc_151673_filtered.h5ad", "dlpfc"),
    SampleSpec("dlpfc_151674", "dlpfc_151674_filtered.h5ad", "dlpfc"),
    SampleSpec("dlpfc_151675", "dlpfc_151675_filtered.h5ad", "dlpfc"),
    SampleSpec("dlpfc_151676", "dlpfc_151676_filtered.h5ad", "dlpfc"),
]

RUN_ID = os.environ.setdefault(
    "LEIDEN_LOUVAIN_RUN_ID",
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}",
)
if RUN_ID in ("", ".", "..") or "/" in RUN_ID or "\\" in RUN_ID:
    raise ValueError("LEIDEN_LOUVAIN_RUN_ID must be a plain directory name.")
LOG_ROOT = OUT_ROOT / "logs" / "leiden_louvain" / RUN_ID
PROGRAM_STARTED = time.perf_counter()


def log_timing(
    stage: str,
    started: float,
    sample=None,
    method=None,
    seed=None,
    status="ok",
    **extra,
) -> None:
    entry = {
        "run_id": RUN_ID,
        "stage": stage,
        "sample": sample,
        "method": method,
        "seed": seed,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "device": "cpu",
        "status": status,
        "process_peak_rss_gib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3
        ),
        **extra,
    }
    folder = LOG_ROOT / (sample or "_run")
    folder.mkdir(parents=True, exist_ok=True)
    name = stage + (f"_{method}" if method else "")
    name += (f"_seed{seed}" if seed is not None else "") + ".json"
    (folder / name).write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    print(
        f"TIME {stage} sample={sample or 'all'} method={method} "
        f"seed={seed} seconds={entry['elapsed_seconds']:.3f} "
        f"status={status}",
        flush=True,
    )


@contextmanager
def timed(stage: str, sample=None, method=None, seed=None):
    started = time.perf_counter()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        log_timing(
            stage, started, sample=sample, method=method, seed=seed, status=status
        )


def consolidate_timing_files() -> None:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(LOG_ROOT.glob("*/*.json"))
    ]
    if rows:
        pd.DataFrame(rows).to_csv(LOG_ROOT / "timings.csv", index=False)
        print(f"Timing summary: {LOG_ROOT / 'timings.csv'}", flush=True)


def write_hardware() -> None:
    def version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    (LOG_ROOT / "hardware.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "python_executable": sys.executable,
                "allocated_cpus": ALLOCATED_CPUS,
                "workers": {
                    "graph_build": GRAPH_BUILD_JOBS,
                    "resolution": RESOLUTION_JOBS,
                    "clustering": CLUSTER_JOBS,
                },
                "packages": {
                    name: version(name)
                    for name in (
                        "scanpy",
                        "anndata",
                        "numpy",
                        "scipy",
                        "pandas",
                        "joblib",
                        "leidenalg",
                        "louvain",
                        "igraph",
                        "scikit-misc",
                    )
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _seed_kw(func, seed):
    "Support Scanpy versions using either rng= or random_state=."
    params = inspect.signature(func).parameters
    if "rng" in params:
        return {"rng": int(seed)}
    return {"random_state": int(seed)}


def _sample_k(spec):
    overrides = N_NEIGHBORS_BY_DATASET if USE_DATASET_SPECIFIC_N_NEIGHBORS else {}
    return get_n_neighbors(
        spec.key,
        default=DEFAULT_N_NEIGHBORS,
        overrides=overrides,
    )


def shared_pca_filename(spec: SampleSpec) -> str:
    "Must remain identical to the convention used by standalone mclust.py."
    return (
        f"{spec.key}_shared_pca_"
        f"hvg{N_HVG}_{HVG_FLAVOR}_"
        f"norm{int(NORMALIZE_TARGET_SUM)}_log{int(LOG1P)}_"
        f"scale{int(SCALE)}_center{int(ZERO_CENTER)}_clip{SCALE_MAX_VALUE}_"
        f"pca{N_PCS}_{PCA_SOLVER}_seed{PCA_SEED}.npy"
    )


def shared_pca_path(spec: SampleSpec) -> Path:
    return (
        OUT_ROOT
        / spec.output_group
        / "shared_representation"
        / shared_pca_filename(spec)
    )


def _graph_filename(spec: SampleSpec, k: int) -> str:
    "Parameterized graph filename tied to the persisted PCA settings."
    return (
        f"{spec.key}_shared_graph_"
        f"hvg{N_HVG}_{HVG_FLAVOR}_"
        f"norm{int(NORMALIZE_TARGET_SUM)}_log{int(LOG1P)}_"
        f"scale{int(SCALE)}_center{int(ZERO_CENTER)}_clip{SCALE_MAX_VALUE}_"
        f"pca{N_PCS}_{PCA_SOLVER}_pcaseed{PCA_SEED}_"
        f"k{k}_{KNN_METRIC}_{CONNECTIVITY_METHOD}_graphseed{GRAPH_SEED}.npz"
    )


def _make_rep_adata(rep: np.ndarray) -> ad.AnnData:
    n_obs = int(rep.shape[0])
    tmp = ad.AnnData(
        X=sparse.csr_matrix((n_obs, 0), dtype=np.float32),
        obs=pd.DataFrame(index=pd.RangeIndex(n_obs).astype(str)),
    )
    tmp.obsm["X_pca"] = np.asarray(rep)
    return tmp


def _build_graph_from_saved_pca(rep: np.ndarray, k: int, graph_path: Path):
    "Build graph ONLY from the exact persisted PCA matrix."
    tmp = _make_rep_adata(rep)
    kwargs = dict(
        n_neighbors=int(k),
        use_rep="X_pca",
        metric=KNN_METRIC,
        method=CONNECTIVITY_METHOD,
        knn=True,
        key_added="shared_graph",
    )
    kwargs.update(_seed_kw(sc.pp.neighbors, GRAPH_SEED))
    sc.pp.neighbors(tmp, **kwargs)
    conn_key = tmp.uns["shared_graph"]["connectivities_key"]
    graph = tmp.obsp[conn_key].tocsr(copy=True)
    graph.sort_indices()
    sparse.save_npz(graph_path, graph, compressed=True)
    return graph


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "input_h5ad": str(path),
        "input_size_bytes": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
    }


def _prepare_sample(spec: SampleSpec):
    input_path = DATA_ROOT / spec.h5ad_name
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    with timed("input_metadata", sample=spec.key):
        backed = sc.read_h5ad(input_path, backed="r")
        try:
            target_clusters = int(get_target_n_clusters(backed, ANNOTATION_KEY))
            spot_ids = np.asarray(backed.obs_names.astype(str), dtype=str)
            n_obs = len(spot_ids)
        finally:
            if backed.file is not None:
                backed.file.close()
        if len(set(spot_ids)) != n_obs or not all(spot_ids):
            raise ValueError(f"{spec.key}: spot IDs must be unique and nonempty.")
        if not 2 <= target_clusters < n_obs:
            raise ValueError(
                f"{spec.key}: invalid annotation-derived G={target_clusters}."
            )
        source_info = _source_signature(input_path)
    k = _sample_k(spec)
    sample_root = OUT_ROOT / spec.output_group
    rep_path = shared_pca_path(spec)
    graph_path = sample_root / "shared_graph" / _graph_filename(spec, k)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path = rep_path.with_name(rep_path.stem + "_obs_names.npy")
    rep_meta_path = rep_path.with_suffix(".json")
    graph_meta_path = graph_path.with_suffix(".json")

    rep_ok = False
    with timed("cached_pca_validation", sample=spec.key):
        if rep_path.is_file() and not FORCE_REBUILD_GRAPH:
            rep = np.load(rep_path, mmap_mode="r", allow_pickle=False)
            rep_ok = rep.shape == (n_obs, N_PCS)
            if rep_ok:
                for lo in range(0, n_obs, 16384):
                    if not np.isfinite(rep[lo : lo + 16384]).all():
                        rep_ok = False
                        break
            del rep
            if rep_ok and ids_path.is_file() and rep_meta_path.is_file():
                saved_ids = np.load(ids_path, allow_pickle=False)
                saved_meta = json.loads(rep_meta_path.read_text(encoding="utf-8"))
                rep_ok = np.array_equal(saved_ids.astype(str), spot_ids) and all(
                    saved_meta.get(key) == value for key, value in source_info.items()
                )
            else:
                rep_ok = False

    if not rep_ok:
        print(
            f"[{spec.key}] building one shared PCA (n={n_obs:,}, G={target_clusters})",
            flush=True,
        )
        with timed("read_expression", sample=spec.key):
            adata = sc.read_h5ad(input_path)
            if not np.array_equal(np.asarray(adata.obs_names.astype(str)), spot_ids):
                raise ValueError(f"{spec.key}: input spot order changed while reading.")
        with timed("expression_preprocessing", sample=spec.key):
            adata = preprocessing(
                adata,
                highly_var=True,
                flavor=HVG_FLAVOR,
                n_hvg=N_HVG,
                batch=False,
                check_raw_counts=True,
                normalize_target_sum=NORMALIZE_TARGET_SUM,
                scale=SCALE,
                zero_center=ZERO_CENTER,
                max_value=SCALE_MAX_VALUE,
            )
            if not np.array_equal(np.asarray(adata.obs_names.astype(str)), spot_ids):
                raise ValueError(f"{spec.key}: preprocessing changed spot order.")
            n_comps = min(N_PCS, adata.n_obs - 1, adata.n_vars - 1)
            if n_comps != N_PCS:
                raise ValueError(
                    f"{spec.key}: PCA-{N_PCS} needs >{N_PCS} "
                    f"spots/HVGs; got {adata.n_obs}/{adata.n_vars}."
                )
        with timed("pca_fit", sample=spec.key):
            pca_kwargs = dict(
                n_comps=N_PCS,
                svd_solver=PCA_SOLVER,
                zero_center=ZERO_CENTER,
                use_highly_variable=False,
            )
            pca_kwargs.update(_seed_kw(sc.pp.pca, PCA_SEED))
            sc.pp.pca(adata, **pca_kwargs)
            representation = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
            if (
                representation.shape != (n_obs, N_PCS)
                or not np.isfinite(representation).all()
            ):
                raise RuntimeError(f"{spec.key}: PCA has invalid shape or values.")
        with timed("save_shared_pca", sample=spec.key):
            np.save(rep_path, representation, allow_pickle=False)
            np.save(ids_path, spot_ids, allow_pickle=False)
            rep_meta_path.write_text(
                json.dumps(source_info, indent=2) + "\n", encoding="utf-8"
            )
        del representation, adata
        gc.collect()

    with timed("pca_fingerprint", sample=spec.key):
        pca_sha256 = _sha256_file(rep_path)

    graph_ok = False
    with timed("cached_graph_validation", sample=spec.key):
        if (
            graph_path.is_file()
            and graph_meta_path.is_file()
            and not FORCE_REBUILD_GRAPH
        ):
            saved = json.loads(graph_meta_path.read_text(encoding="utf-8"))
            if (
                saved.get("pca_sha256") == pca_sha256
                and saved.get("graph_sha256") == _sha256_file(graph_path)
                and saved.get("n_obs") == n_obs
                and saved.get("n_neighbors") == k
                and saved.get("metric") == KNN_METRIC
                and saved.get("connectivity_method") == CONNECTIVITY_METHOD
                and saved.get("graph_seed") == GRAPH_SEED
            ):
                graph = sparse.load_npz(graph_path)
                graph_ok = graph.shape == (n_obs, n_obs)
                del graph
    if not graph_ok:
        with timed("neighbor_graph", sample=spec.key):
            frozen_rep = np.load(rep_path, mmap_mode="r", allow_pickle=False)
            graph = _build_graph_from_saved_pca(frozen_rep, k, graph_path)
            nnz = graph.nnz
            del graph, frozen_rep
        with timed("save_graph_provenance", sample=spec.key):
            graph_meta_path.write_text(
                json.dumps(
                    {
                        "pca_path": str(rep_path),
                        "pca_sha256": pca_sha256,
                        "graph_sha256": _sha256_file(graph_path),
                        "n_obs": n_obs,
                        "n_neighbors": k,
                        "metric": KNN_METRIC,
                        "connectivity_method": CONNECTIVITY_METHOD,
                        "graph_seed": GRAPH_SEED,
                        "graph_nnz": nnz,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        print(f"[{spec.key}] saved graph k={k}, nnz={nnz:,}", flush=True)
    else:
        print(f"[{spec.key}] reused verified PCA and graph (k={k})", flush=True)
    return {
        "key": spec.key,
        "representation_path": str(rep_path),
        "graph_path": str(graph_path),
        "n_obs": n_obs,
        "target_clusters": target_clusters,
        "n_neighbors": k,
    }


def prepare_sample(spec: SampleSpec):
    with timed("sample_preparation_total", sample=spec.key):
        return _prepare_sample(spec)


def tune_one_sample(spec, info):
    with timed("load_graph_for_tuning", sample=spec.key):
        graph = sparse.load_npz(info["graph_path"]).tocsr()
    target = int(info["target_clusters"])
    overrides = RESOLUTION_OVERRIDES.get(spec.key, {})
    if "leiden" in overrides:
        with timed("resolution_override", sample=spec.key, method="leiden"):
            leiden_res = float(overrides["leiden"])
    else:
        with timed("resolution_search", sample=spec.key, method="leiden"):
            leiden_res = search_res(
                adjacency=graph,
                n_clusters=target,
                method="leiden",
                start=RES_START,
                end=RES_END,
                increment=RES_FINE_STEP,
                tolerance=RES_TOLERANCE,
                random_state=RESOLUTION_SEED,
                directed=LEIDEN_DIRECTED,
                use_weights=LEIDEN_USE_WEIGHTS,
                leiden_n_iterations=LEIDEN_N_ITERATIONS,
                leiden_flavor=LEIDEN_FLAVOR,
                louvain_flavor=LOUVAIN_FLAVOR,
                verbose=False,
            )
    if "louvain" in overrides:
        with timed("resolution_override", sample=spec.key, method="louvain"):
            louvain_res = float(overrides["louvain"])
    else:
        with timed("resolution_search", sample=spec.key, method="louvain"):
            louvain_res = search_res(
                adjacency=graph,
                n_clusters=target,
                method="louvain",
                start=RES_START,
                end=RES_END,
                increment=RES_FINE_STEP,
                tolerance=RES_TOLERANCE,
                random_state=RESOLUTION_SEED,
                directed=LOUVAIN_DIRECTED,
                use_weights=LOUVAIN_USE_WEIGHTS,
                leiden_n_iterations=LEIDEN_N_ITERATIONS,
                leiden_flavor=LEIDEN_FLAVOR,
                louvain_flavor=LOUVAIN_FLAVOR,
                verbose=False,
            )
    for method, res in (("leiden", leiden_res), ("louvain", louvain_res)):
        if not np.isfinite(res) or res <= 0:
            raise ValueError(f"{spec.key}/{method}: invalid resolution={res}.")
    print(
        f"[{spec.key}] target={target}; "
        f"Leiden res={leiden_res:.4f}; Louvain res={louvain_res:.4f}"
    )
    return spec.key, {"leiden": leiden_res, "louvain": louvain_res}


_GRAPH_CACHE = None


def _load_graph_cached(path):
    global _GRAPH_CACHE
    if _GRAPH_CACHE is None or _GRAPH_CACHE[0] != path:
        graph = sparse.load_npz(path).tocsr()
        _GRAPH_CACHE = (path, graph)  # one graph per worker bounds memory
    return _GRAPH_CACHE[1]


def _empty_adata(n_obs):
    return ad.AnnData(
        X=sparse.csr_matrix((n_obs, 0), dtype=np.float32),
        obs=pd.DataFrame(index=pd.RangeIndex(n_obs).astype(str)),
    )


def cluster_one(spec, info, method, resolution, seed):
    "One stochastic run; only the final clustering seed varies."
    started = time.perf_counter()
    status = "error"
    try:
        with timed("load_graph", sample=spec.key, method=method, seed=seed):
            graph = _load_graph_cached(info["graph_path"])
            tmp = _empty_adata(info["n_obs"])
        with timed("cluster_fit", sample=spec.key, method=method, seed=seed):
            if method == "leiden":
                kwargs = dict(
                    adjacency=graph,
                    resolution=float(resolution),
                    key_added="cluster",
                    directed=LEIDEN_DIRECTED,
                    use_weights=LEIDEN_USE_WEIGHTS,
                    n_iterations=LEIDEN_N_ITERATIONS,
                    flavor=LEIDEN_FLAVOR,
                )
                kwargs.update(_seed_kw(sc.tl.leiden, seed))
                sc.tl.leiden(tmp, **kwargs)
            elif method == "louvain":
                sc.tl.louvain(
                    tmp,
                    adjacency=graph,
                    resolution=float(resolution),
                    random_state=int(seed),
                    key_added="cluster",
                    directed=LOUVAIN_DIRECTED,
                    use_weights=LOUVAIN_USE_WEIGHTS,
                    flavor=LOUVAIN_FLAVOR,
                )
            else:
                raise ValueError(method)
        with timed("extract_validate", sample=spec.key, method=method, seed=seed):
            labels = tmp.obs["cluster"].astype(int).to_numpy()
            if labels.shape != (info["n_obs"],) or not np.isfinite(labels).all():
                raise ValueError(f"{spec.key}/{method}/{seed}: invalid labels.")
            n_clusters = int(np.unique(labels).size)
        with timed("save_labels", sample=spec.key, method=method, seed=seed):
            temp_dir = OUT_ROOT / spec.output_group / method / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            out_path = temp_dir / f"{spec.key}-{method}-seed{seed}.pkl"
            joblib.dump(labels, out_path, compress=0)
        status = "ok"
        return spec.key, method, int(seed), str(out_path), n_clusters
    finally:
        log_timing(
            "cluster_run_total",
            started,
            sample=spec.key,
            method=method,
            seed=seed,
            status=status,
        )


def consolidate(spec, records, info):
    sample_records = [r for r in records if r[0] == spec.key]
    for method in ("leiden", "louvain"):
        started = time.perf_counter()
        status = "error"
        try:
            _consolidate_one_method(spec, sample_records, method, info["n_obs"])
            status = "ok"
        finally:
            log_timing(
                "consolidate", started, sample=spec.key, method=method, status=status
            )


def _consolidate_one_method(spec, sample_records, method, n_obs):
    recs = sorted(
        [r for r in sample_records if r[1] == method],
        key=lambda r: r[2],
    )
    if [r[2] for r in recs] != list(range(BASE_SEED, BASE_SEED + ITERATIONS)):
        raise RuntimeError(
            f"{spec.key}/{method}: missing or repeated seed among {ITERATIONS} runs"
        )
    results = [joblib.load(r[3]) for r in recs]
    if any(np.asarray(labels).shape != (n_obs,) for labels in results):
        raise RuntimeError(f"{spec.key}/{method}: inconsistent label lengths.")
    method_dir = OUT_ROOT / spec.output_group / method
    method_dir.mkdir(parents=True, exist_ok=True)
    if spec.output_group == "dlpfc":
        section = spec.key.replace("dlpfc_", "")
        name = f"{section}_{method}_results_{ITERATIONS}iterations.pkl"
    else:
        name = f"{method}_results_{ITERATIONS}iterations.pkl"
    joblib.dump(results, method_dir / name, compress=0)
    pd.DataFrame(
        {"seed": [r[2] for r in recs], "n_clusters": [r[4] for r in recs]}
    ).to_csv(
        method_dir / f"{spec.key}_{method}_{ITERATIONS}iterations_cluster_counts.csv",
        index=False,
    )


def main():
    "Run one shared PCA/graph and 50 seeds per method for each sample."
    status = "error"
    sample_started = {spec.key: time.perf_counter() for spec in SAMPLES}
    try:
        write_hardware()
        print("Standalone Leiden/Louvain fixed-PCA fixed-graph experiment")
        print(f"  samples={len(SAMPLES)} (18 selected samples)")
        print(f"  HVGs={N_HVG} ({HVG_FLAVOR})")
        print(f"  normalize_total={NORMALIZE_TARGET_SUM:g}; log1p={LOG1P}")
        print(
            f"  scale={SCALE}, zero_center={ZERO_CENTER}, max_value={SCALE_MAX_VALUE}"
        )
        print(f"  PCA={N_PCS}, solver={PCA_SOLVER}, PCA seed={PCA_SEED}")
        print("  exact PCA saved for standalone mclust, with spot-ID sidecar")
        print(
            f"  KNN k={DEFAULT_N_NEIGHBORS}, metric={KNN_METRIC}, "
            f"method={CONNECTIVITY_METHOD}, graph seed={GRAPH_SEED}"
        )
        print(
            "  graph interpretation: "
            f"Leiden(weights={LEIDEN_USE_WEIGHTS}, directed={LEIDEN_DIRECTED}); "
            f"Louvain(weights={LOUVAIN_USE_WEIGHTS}, directed={LOUVAIN_DIRECTED})"
        )
        print(
            f"  resolution={RES_START}..{RES_END}, step={RES_FINE_STEP}, "
            f"tolerance={RES_TOLERANCE}, seed={RESOLUTION_SEED}"
        )
        print(f"  clustering seeds={BASE_SEED}..{BASE_SEED + ITERATIONS - 1}")
        print(
            f"  workers: graph={GRAPH_BUILD_JOBS}, resolution={RESOLUTION_JOBS}, "
            f"clustering={CLUSTER_JOBS} (allocated={ALLOCATED_CPUS})"
        )
        print(f"  timing logs: {LOG_ROOT}", flush=True)
        with timed("phase1_prepare"):
            prepared = joblib.Parallel(
                n_jobs=GRAPH_BUILD_JOBS,
                backend="loky",
                batch_size=1,
                verbose=10,
            )(joblib.delayed(prepare_sample)(spec) for spec in SAMPLES)
            info_by_key = {x["key"]: x for x in prepared}
        print("Targets derived from adata.obs['annotation']:")
        print({k: v["target_clusters"] for k, v in info_by_key.items()}, flush=True)
        with timed("phase2_resolution"):
            tuned = joblib.Parallel(
                n_jobs=RESOLUTION_JOBS,
                backend="loky",
                batch_size=1,
                verbose=10,
            )(
                joblib.delayed(tune_one_sample)(spec, info_by_key[spec.key])
                for spec in SAMPLES
            )
            resolutions = dict(tuned)
            pd.DataFrame(
                [
                    {
                        "sample": spec.key,
                        "method": method,
                        "target_clusters": info_by_key[spec.key]["target_clusters"],
                        "resolution": resolutions[spec.key][method],
                        "tuning_seed": RESOLUTION_SEED,
                        "resolution_tolerance": RES_TOLERANCE,
                    }
                    for spec in SAMPLES
                    for method in ("leiden", "louvain")
                ]
            ).to_csv(LOG_ROOT / "resolutions.csv", index=False)
        tasks = [
            (spec, method, resolutions[spec.key][method], BASE_SEED + i)
            for spec in SAMPLES
            for i in range(ITERATIONS)
            for method in ("leiden", "louvain")
        ]
        print(
            f"Submitting {len(tasks)} clustering jobs, workers={CLUSTER_JOBS}",
            flush=True,
        )
        with timed("phase3_clustering"):
            records = joblib.Parallel(
                n_jobs=CLUSTER_JOBS,
                backend="loky",
                batch_size=1,
                verbose=10,
            )(
                joblib.delayed(cluster_one)(
                    spec, info_by_key[spec.key], method, resolution, seed
                )
                for spec, method, resolution, seed in tasks
            )
        with timed("phase4_consolidate"):
            for spec in SAMPLES:
                consolidate(spec, records, info_by_key[spec.key])
                log_timing("sample_total", sample_started[spec.key], sample=spec.key)
            pd.DataFrame(
                [
                    {
                        "sample": row[0],
                        "method": row[1],
                        "seed": row[2],
                        "n_clusters": row[4],
                        "target_clusters": info_by_key[row[0]]["target_clusters"],
                        "resolution": resolutions[row[0]][row[1]],
                    }
                    for row in records
                ]
            ).to_csv(LOG_ROOT / "cluster_counts.csv", index=False)
        status = "ok"
        print("Done.", flush=True)
    finally:
        log_timing("total_wall", PROGRAM_STARTED, status=status)
        consolidate_timing_files()


if __name__ == "__main__":
    main()
