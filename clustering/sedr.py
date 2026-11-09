#!/usr/bin/env python3
"Repeat Leiden and mclust clustering on fixed SEDR representations."

from paths import DATA_ROOT as INPUT_DATA_ROOT, OUTPUT_ROOT
import os
import sys
import time
from pathlib import Path

PROGRAM_STARTED = time.perf_counter()

# Use the active environment's C++ runtime when workers load R.

WORKER_LIBSTDCXX = Path(sys.prefix) / "lib" / "libstdc++.so.6"
USING_ENV_LIBSTDCXX = (
    WORKER_LIBSTDCXX.is_file() and b"GLIBCXX_3.4.26" in WORKER_LIBSTDCXX.read_bytes()
)
if USING_ENV_LIBSTDCXX:
    _library_paths = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    if _library_paths[0] == str(WORKER_LIBSTDCXX.parent):
        _library_paths.pop(0)
        if _library_paths:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(_library_paths)
        else:
            os.environ.pop("LD_LIBRARY_PATH", None)
    _preloads = os.environ.get("LD_PRELOAD", "").replace(":", " ").split()
    os.environ["LD_PRELOAD"] = " ".join(
        [str(WORKER_LIBSTDCXX)]
        + [p for p in _preloads if not Path(p).name.startswith("libstdc++.so")]
    )

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
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from sklearn.decomposition import PCA
import SEDR
from functions import get_raw_counts, get_target_n_clusters, search_res

DATA_ROOT = INPUT_DATA_ROOT

OUT_ROOT = OUTPUT_ROOT

METHOD_NAME = "sedr"

RUN_ID = os.environ.setdefault(
    "SEDR_RUN_ID",
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}",
)
if RUN_ID in ("", ".", "..") or "/" in RUN_ID or "\\" in RUN_ID:
    raise ValueError("SEDR_RUN_ID must be a simple directory name.")
LOG_ROOT = OUT_ROOT / "run_logs" / METHOD_NAME / RUN_ID

ANNOTATION_KEY = "annotation"

ITERATIONS = 50

CLUSTER_BASE_SEED = 42  # downstream Leiden and mclust seeds: 42..91

SEDR_TRAINING_SEED = 2023  # SEDR tutorial: reproduces manuscript results

RESOLUTION_SEED = 42  # fixed Leiden resolution-tuning seed

MCLUST_MODEL = "EEE"

FORCE_RETRAIN = False

SEDR_MIN_CELLS_PER_GENE = 50

SEDR_MIN_COUNTS_PER_GENE = 10

SEDR_N_HVG = 2000

SEDR_HVG_FLAVOR = "seurat_v3"

SEDR_NORMALIZE_TARGET_SUM = 1e6

SEDR_LOG1P = False

SEDR_SCALE = True

SEDR_ZERO_CENTER = True  # sc.pp.scale() default

SEDR_SCALE_MAX_VALUE = None  # official tutorials do not clip

SEDR_PCA_N_COMPONENTS = 200

SEDR_PCA_SEED = 42

SEDR_SPATIAL_NEIGHBORS_VISIUM = 12

SEDR_SPATIAL_NEIGHBORS_HIGHRES = 6

SEDR_SPATIAL_GRAPH_MODE = "KNN"

SEDR_MODE = "clustering"

SEDR_USE_DEC = True

SEDR_DEC_N = 1

SEDR_REC_W = 10

SEDR_GCN_W = 0.1

SEDR_SELF_W = 1

SEDR_DEC_KL_W = 1

SEDR_FEAT_HIDDEN1 = 64

SEDR_FEAT_HIDDEN2 = 16

SEDR_GCN_HIDDEN1 = 64

SEDR_GCN_HIDDEN2 = 16

SEDR_DROPOUT = 0.2

SEDR_ALPHA = 1.0

SEDR_INTERNAL_DEC_CLUSTERS = 10

SEDR_MASK_RATE = 0.8

SEDR_SCE_ALPHA = 3

SEDR_PRETRAIN_EPOCHS = 200

SEDR_PRETRAIN_LR = 0.01

SEDR_PRETRAIN_WEIGHT_DECAY = 0.01

SEDR_DEC_EPOCHS = 200

SEDR_DEC_INTERVAL = 20

SEDR_DEC_TOL = 0.0

SEDR_CLUSTER_N_NEIGHBORS = 15

SEDR_CLUSTER_METRIC = "cosine"

SEDR_CLUSTER_CONNECTIVITY_METHOD = "umap"

SEDR_CLUSTER_GRAPH_SEED = 0

SEDR_CLUSTER_KNN = True

SEDR_CLUSTER_USE_WEIGHTS = True

SEDR_CLUSTER_DIRECTED = False

SEDR_CLUSTER_N_ITERATIONS = -1

SEDR_CLUSTER_FLAVOR = "leidenalg"

SEDR_RES_START = 0.02

SEDR_RES_END = 4.0

SEDR_RES_INCREMENT = 0.01

SEDR_RES_TOLERANCE = 2

RESOLUTION_OVERRIDES = {}

_affinity = (
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else (os.cpu_count() or 1)
)
_slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
ALLOCATED_CPUS = min(_affinity, int(_slurm_cpus)) if _slurm_cpus else _affinity
if ALLOCATED_CPUS < 1:
    raise ValueError("No CPU cores allocated to SEDR.")


def _jobs(env_name, default):
    value = int(os.environ.get(env_name, str(min(default, ALLOCATED_CPUS))))
    if not 1 <= value <= ALLOCATED_CPUS:
        raise ValueError(f"{env_name} must be 1..{ALLOCATED_CPUS}.")
    return value


RESOLUTION_JOBS = _jobs("SEDR_RESOLUTION_JOBS", 4)
CLUSTER_JOBS = _jobs("SEDR_CLUSTER_JOBS", 8)
MCLUST_JOBS = _jobs("SEDR_MCLUST_JOBS", 4)

VISIBLE_GPUS = torch.cuda.device_count()

REQUESTED_GPUS = int(os.environ.get("SEDR_NUM_GPUS", str(VISIBLE_GPUS)))

N_GPUS = min(VISIBLE_GPUS, max(0, REQUESTED_GPUS))


@dataclass(frozen=True)
class SampleSpec:
    key: str

    h5ad_name: str

    output_group: str

    platform: str = "Visium"


SAMPLES = [
    SampleSpec("mouse_brain", "mouse_brain_filtered.h5ad", "mouse_brain", "Visium"),
    SampleSpec(
        "mouse_brain_cerebellum",
        "mouse_brain_cerebellum_filtered.h5ad",
        "mouse_brain_cerebellum",
        "Slide-seq",
    ),
    SampleSpec(
        "human_breast_cancer",
        "human_breast_cancer_filtered.h5ad",
        "human_breast_cancer",
        "Visium",
    ),
    SampleSpec(
        "Visium_HD_Human_Colon_Cancer_cropped_square",
        "Visium_HD_Human_Colon_Cancer_cropped_square_filtered.h5ad",
        "Visium_HD_Human_Colon_Cancer_cropped_square",
        "Visium-HD",
    ),
    SampleSpec("ov_ffpe", "ov_ffpe_filtered.h5ad", "ov_ffpe", "Visium-HD"),
    SampleSpec("coad_ffpe", "coad_ffpe_filtered.h5ad", "coad_ffpe", "Visium-HD"),
    SampleSpec("dlpfc_151507", "dlpfc_151507_filtered.h5ad", "dlpfc", "Visium"),
    SampleSpec("dlpfc_151508", "dlpfc_151508_filtered.h5ad", "dlpfc", "Visium"),
    SampleSpec("dlpfc_151509", "dlpfc_151509_filtered.h5ad", "dlpfc", "Visium"),
    SampleSpec("dlpfc_151510", "dlpfc_151510_filtered.h5ad", "dlpfc", "Visium"),
    SampleSpec("dlpfc_151669", "dlpfc_151669_filtered.h5ad", "dlpfc", "Visium"),
    SampleSpec("dlpfc_151670", "dlpfc_151670_filtered.h5ad", "dlpfc", "Visium"),
    SampleSpec("dlpfc_151671", "dlpfc_151671_filtered.h5ad", "dlpfc", "Visium"),
    SampleSpec("dlpfc_151672", "dlpfc_151672_filtered.h5ad", "dlpfc", "Visium"),
    SampleSpec("dlpfc_151673", "dlpfc_151673_filtered.h5ad", "dlpfc", "Visium"),
    SampleSpec("dlpfc_151674", "dlpfc_151674_filtered.h5ad", "dlpfc", "Visium"),
    SampleSpec("dlpfc_151675", "dlpfc_151675_filtered.h5ad", "dlpfc", "Visium"),
    SampleSpec("dlpfc_151676", "dlpfc_151676_filtered.h5ad", "dlpfc", "Visium"),
]


def log_timing(stage, started, sample=None, seed=None, device="cpu", status="ok"):
    "Write one event per process and merge them into timings.csv at exit."
    event = {
        "run_id": RUN_ID,
        "stage": stage,
        "sample": sample,
        "seed": seed,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "device": str(device),
        "status": status,
        "process_peak_rss_gib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3
        ),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        gpu = torch.device(device)
        event.update(
            gpu_name=torch.cuda.get_device_name(gpu),
            peak_gpu_allocated_gib=round(
                torch.cuda.max_memory_allocated(gpu) / 2**30, 3
            ),
        )
    directory = LOG_ROOT / "timing_events"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{uuid.uuid4().hex}.json").write_text(
        json.dumps(event, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"TIME {stage} sample={sample or 'all'} seed={seed} "
        f"seconds={event['elapsed_seconds']:.3f} status={status}",
        flush=True,
    )


@contextmanager
def timed(stage, sample=None, seed=None, device="cpu"):
    gpu = torch.device(device) if str(device).startswith("cuda") else None
    if gpu is not None:
        torch.cuda.synchronize(gpu)
    started = time.perf_counter()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        if gpu is not None:
            torch.cuda.synchronize(gpu)
        log_timing(
            stage, started, sample=sample, seed=seed, device=device, status=status
        )


def write_timing_summary():
    directory = LOG_ROOT / "timing_events"
    rows = (
        [json.loads(p.read_text(encoding="utf-8")) for p in directory.glob("*.json")]
        if directory.exists()
        else []
    )
    if rows:
        pd.DataFrame(rows).sort_values("finished_utc").to_csv(
            LOG_ROOT / "timings.csv", index=False
        )
        print(f"Timing summary: {LOG_ROOT / 'timings.csv'}", flush=True)


def write_hardware():
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": RUN_ID,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "allocated_cpus": ALLOCATED_CPUS,
        "resolution_jobs": RESOLUTION_JOBS,
        "cluster_jobs": CLUSTER_JOBS,
        "mclust_jobs": MCLUST_JOBS,
        "gpus": N_GPUS,
        "worker_libstdcxx": str(WORKER_LIBSTDCXX) if USING_ENV_LIBSTDCXX else None,
        "packages": {
            name: version(name)
            for name in (
                "SEDR",
                "torch",
                "scanpy",
                "anndata",
                "numpy",
                "scikit-learn",
                "rpy2",
                "joblib",
            )
        },
    }
    (LOG_ROOT / "hardware.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def _seed_kw(func, seed):
    "Support Scanpy random_state and newer rng keyword."

    params = inspect.signature(func).parameters

    if "rng" in params:
        return {"rng": int(seed)}

    return {"random_state": int(seed)}


def preprocessing_profile(spec: SampleSpec) -> str:
    "Return the documented SEDR preprocessing profile for this platform."

    if spec.platform == "Visium":
        return "visium_hvg2000"

    if spec.platform == "Slide-seq":
        return "highres_allgenes"

    if spec.platform == "Visium-HD":
        return "visiumhd_hvg2000_spk6_extrapolation"

    raise ValueError(f"Unsupported SEDR platform: {spec.platform!r}")


def spatial_neighbors(spec: SampleSpec) -> int:
    if spec.platform in {"Slide-seq", "Visium-HD"}:
        return SEDR_SPATIAL_NEIGHBORS_HIGHRES

    if spec.platform == "Visium":
        return SEDR_SPATIAL_NEIGHBORS_VISIUM

    raise ValueError(f"Unsupported SEDR platform: {spec.platform!r}")


def representation_tag(spec: SampleSpec) -> str:
    "Tag for the frozen SEDR representation shared by Leiden and mclust."

    platform = spec.platform.lower().replace("-", "")

    profile = preprocessing_profile(spec)

    return (
        f"{spec.key}_sedr_fixed_"
        f"platform{platform}_prep{profile}_"
        f"norm{int(SEDR_NORMALIZE_TARGET_SUM)}_pca{SEDR_PCA_N_COMPONENTS}_"
        f"spk{spatial_neighbors(spec)}_trainseed{SEDR_TRAINING_SEED}_"
        f"dec{int(SEDR_USE_DEC)}"
    )


def embedding_filename(spec: SampleSpec) -> str:
    return f"{representation_tag(spec)}_embedding.npy"


def graph_filename(spec: SampleSpec) -> str:
    "Parameterized Leiden graph filename based on the same frozen embedding."

    return (
        f"{representation_tag(spec)}_"
        f"clusterk{SEDR_CLUSTER_N_NEIGHBORS}_{SEDR_CLUSTER_METRIC}.npz"
    )


def representation_provenance(spec, input_path, spot_ids):
    "Invalidate cached vectors when input data or training choices change."
    stat = input_path.stat()
    recipe = {
        "platform": spec.platform,
        "profile": preprocessing_profile(spec),
        "min_cells": SEDR_MIN_CELLS_PER_GENE,
        "min_counts": SEDR_MIN_COUNTS_PER_GENE,
        "n_hvg": SEDR_N_HVG,
        "hvg_flavor": SEDR_HVG_FLAVOR,
        "normalize_target": SEDR_NORMALIZE_TARGET_SUM,
        "log1p": SEDR_LOG1P,
        "scale": SEDR_SCALE,
        "zero_center": SEDR_ZERO_CENTER,
        "scale_max": SEDR_SCALE_MAX_VALUE,
        "pca_n": SEDR_PCA_N_COMPONENTS,
        "pca_seed": SEDR_PCA_SEED,
        "spatial_neighbors": spatial_neighbors(spec),
        "spatial_mode": SEDR_SPATIAL_GRAPH_MODE,
        "train_seed": SEDR_TRAINING_SEED,
        "mode": SEDR_MODE,
        "use_dec": SEDR_USE_DEC,
        "dec_n": SEDR_DEC_N,
        "loss_weights": [SEDR_REC_W, SEDR_GCN_W, SEDR_SELF_W, SEDR_DEC_KL_W],
        "architecture": [
            SEDR_FEAT_HIDDEN1,
            SEDR_FEAT_HIDDEN2,
            SEDR_GCN_HIDDEN1,
            SEDR_GCN_HIDDEN2,
            SEDR_DROPOUT,
            SEDR_ALPHA,
            SEDR_INTERNAL_DEC_CLUSTERS,
            SEDR_MASK_RATE,
            SEDR_SCE_ALPHA,
        ],
        "pretrain": [
            SEDR_PRETRAIN_EPOCHS,
            SEDR_PRETRAIN_LR,
            SEDR_PRETRAIN_WEIGHT_DECAY,
        ],
        "dec": [SEDR_DEC_EPOCHS, SEDR_DEC_INTERVAL, SEDR_DEC_TOL],
    }
    ids_hash = hashlib.sha256("\0".join(map(str, spot_ids)).encode()).hexdigest()
    return {
        "input_path": str(input_path.resolve()),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "spot_ids_sha256": ids_hash,
        "recipe": recipe,
        "software": {
            "scanpy": sc.__version__,
            "torch": torch.__version__,
            "SEDR": getattr(SEDR, "__version__", None),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
    }


def target_and_size(input_path: Path):
    backed = sc.read_h5ad(input_path, backed="r")

    try:
        target = get_target_n_clusters(backed, ANNOTATION_KEY)

        n_obs = int(backed.n_obs)

        spot_ids = np.asarray(backed.obs_names.astype(str), dtype=str)

    finally:
        if backed.file is not None:
            backed.file.close()

    if spot_ids.shape != (n_obs,) or len(set(spot_ids)) != n_obs or not all(spot_ids):
        raise ValueError(f"{input_path}: invalid or duplicated observation IDs.")
    return target, n_obs, spot_ids


def preprocess_sedr(adata: ad.AnnData, spec: SampleSpec, device="cpu") -> ad.AnnData:
    "Apply released Visium recipe or documented high-resolution adaptations."
    if adata.n_obs <= 0:
        raise ValueError(f"{spec.key}: no observations")
    with timed("raw_counts_check", sample=spec.key, device=device):
        counts, count_source = get_raw_counts(adata, check_values=True)
        print(f"[{spec.key}] SEDR raw-count source: {count_source}")
        if count_source != "adata.X":
            adata.X = counts.copy()
    profile = preprocessing_profile(spec)
    if spec.platform in {"Visium", "Visium-HD"}:
        with timed("gene_filter_hvg", sample=spec.key, device=device):
            sc.pp.filter_genes(adata, min_cells=SEDR_MIN_CELLS_PER_GENE)
            sc.pp.filter_genes(adata, min_counts=SEDR_MIN_COUNTS_PER_GENE)
            if adata.n_vars < 2:
                raise ValueError(f"{spec.key}: fewer than two genes after filtering")

            sc.pp.highly_variable_genes(
                adata,
                flavor=SEDR_HVG_FLAVOR,
                n_top_genes=min(SEDR_N_HVG, adata.n_vars),
                check_values=True,
            )
        with timed("normalize", sample=spec.key, device=device):
            sc.pp.normalize_total(adata, target_sum=SEDR_NORMALIZE_TARGET_SUM)
            adata = adata[:, adata.var["highly_variable"]].copy()
    elif spec.platform == "Slide-seq":
        with timed("normalize", sample=spec.key, device=device):
            sc.pp.normalize_total(adata, target_sum=SEDR_NORMALIZE_TARGET_SUM)
    else:
        raise ValueError(f"Unsupported SEDR platform: {spec.platform!r}")
    if SEDR_LOG1P:
        with timed("log1p", sample=spec.key, device=device):
            sc.pp.log1p(adata)
    if adata.n_vars < 2:
        raise ValueError(f"{spec.key}: fewer than two genes for PCA")
    if SEDR_SCALE:
        with timed("scale", sample=spec.key, device=device):
            sc.pp.scale(
                adata,
                zero_center=SEDR_ZERO_CENTER,
                max_value=SEDR_SCALE_MAX_VALUE,
            )
    with timed("pca", sample=spec.key, device=device):
        n_comps = min(SEDR_PCA_N_COMPONENTS, adata.n_obs - 1, adata.n_vars)
        if n_comps < 1:
            raise ValueError(f"{spec.key}: cannot compute PCA on {adata.shape}")
        pca_input = (
            adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        )
        adata.obsm["X_pca"] = PCA(
            n_components=n_comps,
            random_state=SEDR_PCA_SEED,
        ).fit_transform(pca_input)
        del pca_input
    print(
        f"[{spec.key}] profile={profile}: n={adata.n_obs:,}, genes={adata.n_vars:,}; "
        f"PCA={n_comps}; spatial_k={spatial_neighbors(spec)}"
    )
    return adata


def make_cluster_adata(embedding: np.ndarray) -> ad.AnnData:
    n_obs = embedding.shape[0]

    tmp = ad.AnnData(
        X=sparse.csr_matrix((n_obs, 0), dtype=np.float32),
        obs=pd.DataFrame(index=pd.RangeIndex(n_obs).astype(str)),
    )

    tmp.obsm["SEDR"] = np.array(embedding, copy=True, order="C")

    return tmp


def build_leiden_graph_from_embedding(embedding: np.ndarray, graph_path: Path):
    "Build and persist the one fixed SEDR->Leiden graph."

    cluster_adata = make_cluster_adata(embedding)

    neighbor_kwargs = dict(
        n_neighbors=SEDR_CLUSTER_N_NEIGHBORS,
        use_rep="SEDR",
        metric=SEDR_CLUSTER_METRIC,
        method=SEDR_CLUSTER_CONNECTIVITY_METHOD,
        knn=SEDR_CLUSTER_KNN,
        key_added="shared_graph",
    )

    neighbor_kwargs.update(_seed_kw(sc.pp.neighbors, SEDR_CLUSTER_GRAPH_SEED))

    sc.pp.neighbors(cluster_adata, **neighbor_kwargs)

    conn_key = cluster_adata.uns["shared_graph"]["connectivities_key"]

    graph = cluster_adata.obsp[conn_key].tocsr(copy=True)

    graph.sort_indices()

    sparse.save_npz(graph_path, graph, compressed=True)

    return graph


def train_and_build_one(spec: SampleSpec, device):
    input_path = DATA_ROOT / spec.h5ad_name
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    with timed("input_metadata", sample=spec.key, device=device):
        target, n_obs, spot_ids = target_and_size(input_path)
        if not 2 <= int(target) < n_obs:
            raise ValueError(f"{spec.key}: invalid annotation-derived target {target}")
        if n_obs <= SEDR_SPATIAL_NEIGHBORS_VISIUM:
            raise ValueError(f"{spec.key}: insufficient observations for spatial graph")
        provenance = representation_provenance(spec, input_path, spot_ids)

    method_dir = OUT_ROOT / spec.output_group / METHOD_NAME
    embedding_dir = method_dir / "shared_embedding"
    graph_dir = method_dir / "shared_graph"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = embedding_dir / embedding_filename(spec)
    spot_ids_path = embedding_dir / f"{representation_tag(spec)}_spot_ids.npy"
    manifest_path = embedding_dir / f"{representation_tag(spec)}_provenance.json"
    graph_path = graph_dir / graph_filename(spec)
    graph_manifest = graph_dir / f"{graph_path.stem}_provenance.json"
    graph_settings = {
        "embedding": provenance,
        "n_neighbors": SEDR_CLUSTER_N_NEIGHBORS,
        "metric": SEDR_CLUSTER_METRIC,
        "method": SEDR_CLUSTER_CONNECTIVITY_METHOD,
        "knn": SEDR_CLUSTER_KNN,
        "seed": SEDR_CLUSTER_GRAPH_SEED,
        "scanpy_version": sc.__version__,
    }
    info = {
        "key": spec.key,
        "embedding_path": str(embedding_path),
        "graph_path": str(graph_path),
        "spot_ids_path": str(spot_ids_path),
        "n_obs": n_obs,
        "target_clusters": int(target),
        "platform": spec.platform,
        "spatial_neighbors": spatial_neighbors(spec),
    }

    if embedding_path.is_file() and manifest_path.is_file() and not FORCE_RETRAIN:
        with timed("check_embedding_cache", sample=spec.key, device=device):
            saved_provenance = json.loads(manifest_path.read_text(encoding="utf-8"))
            cache_matches = saved_provenance == provenance and spot_ids_path.is_file()
            if cache_matches:
                cached_ids = np.load(spot_ids_path, allow_pickle=False)
                cache_matches = np.array_equal(cached_ids, spot_ids)
            if cache_matches:
                embedding = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
                cache_matches = (
                    embedding.ndim == 2
                    and embedding.shape[0] == n_obs
                    and embedding.shape[1] > 0
                    and np.issubdtype(embedding.dtype, np.number)
                    and np.isfinite(embedding).all()
                )
        if cache_matches:
            info["embedding_dim"] = int(embedding.shape[1])
            graph_ok = False
            if graph_path.is_file() and graph_manifest.is_file():
                with timed("check_leiden_graph_cache", sample=spec.key, device=device):
                    stored_settings = json.loads(
                        graph_manifest.read_text(encoding="utf-8")
                    )
                    if stored_settings == graph_settings:
                        graph = sparse.load_npz(graph_path).tocsr()
                        graph_ok = (
                            graph.shape == (n_obs, n_obs)
                            and np.isfinite(graph.data).all()
                        )
                        del graph
            if not graph_ok:
                with timed("build_leiden_graph", sample=spec.key, device=device):
                    graph = build_leiden_graph_from_embedding(embedding, graph_path)
                    graph_manifest.write_text(
                        json.dumps(graph_settings, indent=2) + "\n", encoding="utf-8"
                    )
                    del graph
            del embedding
            print(
                f"[{spec.key}] reused validated frozen SEDR embedding; "
                f"n={n_obs}, dim={info['embedding_dim']}, target={target}"
            )
            return info
        print(f"[{spec.key}] cached embedding provenance/shape changed; retraining")
    elif embedding_path.is_file() and not FORCE_RETRAIN:
        print(
            f"[{spec.key}] cached embedding lacks provenance; retraining for safe reuse"
        )

    print(
        f"[{spec.key}] SEDR train ONCE on {device}; n={n_obs:,}, target={target}, "
        f"platform={spec.platform}, spatial_k={spatial_neighbors(spec)}"
    )
    with timed("read_h5ad", sample=spec.key, device=device):
        adata = sc.read_h5ad(input_path)
        if not np.array_equal(np.asarray(adata.obs_names.astype(str)), spot_ids):
            raise ValueError(f"{spec.key}: observation IDs changed during input read")
        if "spatial" not in adata.obsm:
            raise ValueError(f"{spec.key}: missing obsm['spatial'] for SEDR graph")
        coords = np.asarray(adata.obsm["spatial"])
        if (
            coords.ndim != 2
            or coords.shape[0] != n_obs
            or coords.shape[1] < 2
            or not np.isfinite(coords[:, :2]).all()
        ):
            raise ValueError(f"{spec.key}: invalid spatial coordinates {coords.shape}")
    with timed("sedr_preprocess_pca", sample=spec.key, device=device):
        adata = preprocess_sedr(adata, spec, device=device)
        if not np.array_equal(np.asarray(adata.obs_names.astype(str)), spot_ids):
            raise ValueError(f"{spec.key}: preprocessing changed observation order")
    SEDR.fix_seed(SEDR_TRAINING_SEED)
    with timed("sedr_spatial_graph", sample=spec.key, device=device):
        graph_dict = SEDR.graph_construction(
            adata, spatial_neighbors(spec), mode=SEDR_SPATIAL_GRAPH_MODE
        )
    with timed(
        "sedr_fit_pretrain_and_dec" if SEDR_USE_DEC else "sedr_fit_pretrain",
        sample=spec.key,
        device=device,
    ):
        sedr_net = SEDR.Sedr(
            adata.obsm["X_pca"],
            graph_dict,
            rec_w=SEDR_REC_W,
            gcn_w=SEDR_GCN_W,
            self_w=SEDR_SELF_W,
            dec_kl_w=SEDR_DEC_KL_W,
            mode=SEDR_MODE,
            device=device,
        )
        if SEDR_USE_DEC:
            sedr_net.train_with_dec(
                epochs=SEDR_DEC_EPOCHS,
                dec_interval=SEDR_DEC_INTERVAL,
                dec_tol=SEDR_DEC_TOL,
                N=SEDR_DEC_N,
            )
        else:
            sedr_net.train_without_dec(
                epochs=SEDR_PRETRAIN_EPOCHS,
                lr=SEDR_PRETRAIN_LR,
                decay=SEDR_PRETRAIN_WEIGHT_DECAY,
                N=SEDR_DEC_N,
            )
    with timed("extract_embedding", sample=spec.key, device=device):
        sedr_feat, _, _, _ = sedr_net.process()
        sedr_feat = np.asarray(sedr_feat)
        if (
            sedr_feat.ndim != 2
            or sedr_feat.shape[0] != n_obs
            or sedr_feat.shape[1] == 0
            or not np.isfinite(sedr_feat).all()
        ):
            raise RuntimeError(f"{spec.key}: invalid SEDR embedding {sedr_feat.shape}")
        info["embedding_dim"] = int(sedr_feat.shape[1])
    with timed("save_embedding", sample=spec.key, device=device):
        np.save(embedding_path, sedr_feat, allow_pickle=False)
        np.save(spot_ids_path, spot_ids, allow_pickle=False)
        manifest_path.write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        del sedr_feat
    with timed("build_leiden_graph", sample=spec.key, device=device):
        frozen_embedding = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        graph = build_leiden_graph_from_embedding(frozen_embedding, graph_path)
        graph_manifest.write_text(
            json.dumps(graph_settings, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[{spec.key}] shared embedding={embedding_path.name}; "
            f"shape={tuple(frozen_embedding.shape)}; Leiden graph nnz={graph.nnz:,}"
        )
        del graph, frozen_embedding
    del sedr_net, graph_dict, adata
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return info


def train_gpu_batch(specs, gpu_id):
    "One worker owns one GPU and trains assigned samples serially."

    torch.cuda.set_device(gpu_id)

    device = f"cuda:{gpu_id}"

    out = []

    for spec in specs:
        out.append(train_and_build_one(spec, device))

    return out


def train_cpu_batch(specs):
    return [train_and_build_one(spec, "cpu") for spec in specs]


def _tune_one_sample(spec: SampleSpec, info: dict):
    if spec.key in RESOLUTION_OVERRIDES:
        res = float(RESOLUTION_OVERRIDES[spec.key])

        print(f"[{spec.key}] manual SEDR-Leiden resolution={res:.4f}")

        return spec.key, res

    graph = sparse.load_npz(info["graph_path"]).tocsr()

    target = int(info["target_clusters"])

    res = search_res(
        adjacency=graph,
        n_clusters=target,
        method="leiden",
        start=SEDR_RES_START,
        end=SEDR_RES_END,
        increment=SEDR_RES_INCREMENT,
        tolerance=SEDR_RES_TOLERANCE,
        random_state=RESOLUTION_SEED,
        directed=SEDR_CLUSTER_DIRECTED,
        use_weights=SEDR_CLUSTER_USE_WEIGHTS,
        leiden_n_iterations=SEDR_CLUSTER_N_ITERATIONS,
        leiden_flavor=SEDR_CLUSTER_FLAVOR,
        verbose=False,
    )

    print(
        f"[{spec.key}] SEDR fixed embedding/graph: target={target}; "
        f"Leiden resolution={res:.4f}"
    )

    return spec.key, float(res)


def tune_one_sample(spec: SampleSpec, info: dict):
    with timed("tune_leiden_resolution", sample=spec.key):
        return _tune_one_sample(spec, info)


_GRAPH_CACHE = {}

_EMBEDDING_CACHE = {}

_MCLUST_BACKEND = None


def load_graph_cached(path):
    graph = _GRAPH_CACHE.get(path)

    if graph is None:
        graph = sparse.load_npz(path).tocsr()

        _GRAPH_CACHE[path] = graph

    return graph


def load_embedding_cached(path):
    embedding = _EMBEDDING_CACHE.get(path)

    if embedding is None:
        embedding = np.asarray(
            np.load(path, allow_pickle=False),
            dtype=np.float64,
            order="C",
        )

        embedding.setflags(write=False)

        _EMBEDDING_CACHE[path] = embedding

    return embedding


def empty_adata(n_obs):
    return ad.AnnData(
        X=sparse.csr_matrix((n_obs, 0), dtype=np.float32),
        obs=pd.DataFrame(index=pd.RangeIndex(n_obs).astype(str)),
    )


def _leiden_one(spec: SampleSpec, info: dict, resolution: float, seed: int):
    graph = load_graph_cached(info["graph_path"])

    tmp = empty_adata(info["n_obs"])

    kwargs = dict(
        adjacency=graph,
        resolution=float(resolution),
        key_added="domain",
        directed=SEDR_CLUSTER_DIRECTED,
        use_weights=SEDR_CLUSTER_USE_WEIGHTS,
        n_iterations=SEDR_CLUSTER_N_ITERATIONS,
        flavor=SEDR_CLUSTER_FLAVOR,
    )

    kwargs.update(_seed_kw(sc.tl.leiden, seed))

    sc.tl.leiden(tmp, **kwargs)

    labels = tmp.obs["domain"].astype(int).to_numpy()

    method_dir = OUT_ROOT / spec.output_group / METHOD_NAME

    temp_dir = method_dir / "temp_leiden"

    temp_dir.mkdir(parents=True, exist_ok=True)

    out_path = temp_dir / f"{spec.key}-sedr-leiden-seed{seed}.pkl"

    joblib.dump(labels, out_path, compress=0)

    return spec.key, int(seed), str(out_path), int(np.unique(labels).size)


def leiden_one(spec: SampleSpec, info: dict, resolution: float, seed: int):
    with timed("leiden_run", sample=spec.key, seed=seed):
        return _leiden_one(spec, info, resolution, seed)


def _get_mclust_backend():
    "Initialize R once per loky worker; avoid rpy2-generated long call names."
    global _MCLUST_BACKEND
    if _MCLUST_BACKEND is not None:
        return _MCLUST_BACKEND
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr

        importr("mclust")

        r_mclust = ro.r(
            "function(x, G, modelNames) mclust::Mclust(x, G=G, modelNames=modelNames)"
        )
        r_set_seed = ro.r["set.seed"]
    except Exception as exc:
        raise RuntimeError(
            "SEDR+mclust needs working R, rpy2 and R package mclust in the "
            "active SEDR environment."
        ) from exc
    _MCLUST_BACKEND = (ro, r_mclust, r_set_seed)
    return _MCLUST_BACKEND


def mclust_worker_preflight():
    "Test R loading in a fresh loky process before lengthy SEDR training."
    import scipy

    ro, _, _ = _get_mclust_backend()
    return (
        scipy.__version__,
        str(ro.r("as.character(getRversion())")[0]),
        str(ro.r('as.character(utils::packageVersion("mclust"))')[0]),
    )


def _mclust_one(spec: SampleSpec, info: dict, seed: int):
    "Use the frozen SEDR vectors, retaining mclust's posterior z for entropy."
    with timed("mclust_load_embedding", sample=spec.key, seed=seed):
        embedding = load_embedding_cached(info["embedding_path"])
    target = int(info["target_clusters"])
    n_obs = int(info["n_obs"])
    with timed("mclust_backend_init", sample=spec.key, seed=seed):
        ro, r_mclust, r_set_seed = _get_mclust_backend()
    with timed("mclust_matrix_conversion", sample=spec.key, seed=seed):
        r_embedding = ro.r["matrix"](
            ro.FloatVector(embedding.ravel(order="F")),
            nrow=int(embedding.shape[0]),
            ncol=int(embedding.shape[1]),
        )
    with timed("mclust_fit", sample=spec.key, seed=seed):
        r_set_seed(int(seed))
        fit = r_mclust(
            r_embedding, ro.IntVector([target]), ro.StrVector([MCLUST_MODEL])
        )
    with timed("mclust_extract_validate", sample=spec.key, seed=seed):
        if fit is None or not hasattr(fit, "rx2"):
            raise RuntimeError(f"{spec.key} seed={seed}: mclust returned NULL")
        labels = np.array(fit.rx2("classification"), dtype=np.int64, copy=True).reshape(
            -1
        )
        z = np.array(fit.rx2("z"), dtype=np.float64, copy=True)
        fitted_g = int(fit.rx2("G")[0])
        model_name = str(fit.rx2("modelName")[0])
        if (
            labels.shape != (n_obs,)
            or z.shape != (n_obs, target)
            or fitted_g != target
            or model_name != MCLUST_MODEL
            or not np.isfinite(z).all()
            or np.any(z < -1e-10)
            or np.any(z > 1 + 1e-10)
            or not np.allclose(z.sum(axis=1), 1, atol=1e-6)
            or not np.array_equal(np.argmax(z, axis=1) + 1, labels)
        ):
            raise RuntimeError(
                f"{spec.key} seed={seed}: invalid mclust labels/posteriors: "
                f"labels={labels.shape}, z={z.shape}, fitted G={fitted_g}, "
                f"model={model_name}"
            )
    with timed("mclust_save_run", sample=spec.key, seed=seed):
        method_dir = OUT_ROOT / spec.output_group / METHOD_NAME
        temp_dir = method_dir / "temp_mclust"
        soft_dir = method_dir / "memberships" / "mclust"
        temp_dir.mkdir(parents=True, exist_ok=True)
        soft_dir.mkdir(parents=True, exist_ok=True)
        out_path = temp_dir / f"{spec.key}-sedr-mclust-seed{seed}.pkl"
        soft_path = soft_dir / f"{spec.key}-sedr-mclust-seed{seed}_memberships.npz"
        ids = np.load(info["spot_ids_path"], allow_pickle=False)
        if len(ids) != n_obs:
            raise RuntimeError(f"{spec.key}: saved spot ID count changed")
        joblib.dump(labels, out_path, compress=0)
        np.savez_compressed(
            soft_path,
            spot_id=ids,
            z=z,
            hard_label=labels,
            uncertainty=1 - z.max(axis=1),
            cluster_id=np.arange(1, target + 1),
            seed=np.int64(seed),
            model_name=model_name,
        )
    return spec.key, int(seed), str(out_path), int(np.unique(labels).size)


def mclust_one(spec: SampleSpec, info: dict, seed: int):
    with timed("mclust_run", sample=spec.key, seed=seed):
        return _mclust_one(spec, info, seed)


def consolidate(spec: SampleSpec, records, clustering_method: str, info: dict):
    recs = sorted(
        [r for r in records if r[0] == spec.key],
        key=lambda r: r[1],
    )

    if len(recs) != ITERATIONS:
        raise RuntimeError(
            f"{spec.key}: expected {ITERATIONS} {clustering_method} runs, "
            f"got {len(recs)}"
        )

    expected_seeds = list(range(CLUSTER_BASE_SEED, CLUSTER_BASE_SEED + ITERATIONS))
    if [int(r[1]) for r in recs] != expected_seeds:
        raise RuntimeError(f"{spec.key}: missing or duplicate {clustering_method} seed")

    results = [joblib.load(r[2]) for r in recs]

    if any(np.asarray(labels).shape != (info["n_obs"],) for labels in results):
        raise RuntimeError(f"{spec.key}: {clustering_method} label length mismatch")

    method_dir = OUT_ROOT / spec.output_group / METHOD_NAME

    method_dir.mkdir(parents=True, exist_ok=True)

    if clustering_method == "leiden":
        if spec.output_group == "dlpfc":
            section = spec.key.replace("dlpfc_", "")

            name = f"sedr_results_{section}_{ITERATIONS}iterations.pkl"

        else:
            name = f"sedr_results_{ITERATIONS}iterations.pkl"

    elif clustering_method == "mclust":
        if spec.output_group == "dlpfc":
            section = spec.key.replace("dlpfc_", "")

            name = f"sedr_mclust_results_{section}_{ITERATIONS}iterations.pkl"

        else:
            name = f"sedr_mclust_results_{ITERATIONS}iterations.pkl"

    else:
        raise ValueError(f"Unknown clustering_method={clustering_method!r}")

    joblib.dump(results, method_dir / name, compress=0)

    pd.DataFrame(
        {
            "seed": [int(r[1]) for r in recs],
            "n_clusters": [int(r[3]) for r in recs],
            "annotation_target": int(info["target_clusters"]),
            "leiden_resolution": (
                info.get("leiden_resolution")
                if clustering_method == "leiden"
                else np.nan
            ),
        }
    ).to_csv(method_dir / name.replace(".pkl", "_run_counts.csv"), index=False)


def print_sedr_settings():
    print(
        "SEDR fixed-representation experiment: Leiden + mclust on one shared embedding"
    )

    print("  preprocessing profiles:")

    print(
        "    Visium: filter_genes(min_cells="
        f"{SEDR_MIN_CELLS_PER_GENE}, min_counts={SEDR_MIN_COUNTS_PER_GENE}); "
        f"HVG={SEDR_N_HVG} ({SEDR_HVG_FLAVOR})"
    )

    print(
        "    Slide-seq: high-resolution released recipe; " "no gene filter/HVG subset"
    )

    print(
        "    Visium-HD: 10X-style expression preprocessing; "
        f"filter_genes + HVG={SEDR_N_HVG} ({SEDR_HVG_FLAVOR}); "
        "spatial k=6 is the HD extrapolation"
    )

    print(
        f"    all profiles: normalize_total={SEDR_NORMALIZE_TARGET_SUM:g}; "
        f"log1p={SEDR_LOG1P}; scale={SEDR_SCALE}; "
        f"zero_center={SEDR_ZERO_CENTER}; max_value={SEDR_SCALE_MAX_VALUE}"
    )

    print(f"    sklearn PCA={SEDR_PCA_N_COMPONENTS}, PCA seed={SEDR_PCA_SEED}")

    print("  SEDR spatial graph:")

    print(f"    Visium spatial neighbors={SEDR_SPATIAL_NEIGHBORS_VISIUM}")

    print(
        f"    Slide-seq / cropped Visium-HD spatial neighbors="
        f"{SEDR_SPATIAL_NEIGHBORS_HIGHRES}"
    )

    print(f"    graph mode={SEDR_SPATIAL_GRAPH_MODE}")

    print("  SEDR training:")

    print(f"    seed={SEDR_TRAINING_SEED} (FIXED; model trained once)")

    print(f"    mode={SEDR_MODE}; DEC={SEDR_USE_DEC}; N={SEDR_DEC_N}")

    print(
        f"    loss weights: rec={SEDR_REC_W}, gcn={SEDR_GCN_W}, "
        f"self={SEDR_SELF_W}, dec_kl={SEDR_DEC_KL_W}"
    )

    print(
        f"    hidden dims: feat=({SEDR_FEAT_HIDDEN1},{SEDR_FEAT_HIDDEN2}), "
        f"gcn=({SEDR_GCN_HIDDEN1},{SEDR_GCN_HIDDEN2})"
    )

    print(
        f"    dropout={SEDR_DROPOUT}, alpha={SEDR_ALPHA}, "
        f"internal DEC clusters={SEDR_INTERNAL_DEC_CLUSTERS}, "
        f"mask_rate={SEDR_MASK_RATE}, SCE alpha={SEDR_SCE_ALPHA}"
    )

    print(
        f"    pretrain: epochs={SEDR_PRETRAIN_EPOCHS}, lr={SEDR_PRETRAIN_LR}, "
        f"weight_decay={SEDR_PRETRAIN_WEIGHT_DECAY}"
    )

    print(
        f"    DEC phase: epochs={SEDR_DEC_EPOCHS}, "
        f"interval={SEDR_DEC_INTERVAL}, tol={SEDR_DEC_TOL}"
    )

    print("  shared representation:")

    print("    exact SEDR embedding is saved once and reused by BOTH Leiden and mclust")

    print("  downstream fixed graph for Leiden only:")

    print(
        f"    KNN={SEDR_CLUSTER_N_NEIGHBORS}, metric={SEDR_CLUSTER_METRIC}, "
        f"connectivity={SEDR_CLUSTER_CONNECTIVITY_METHOD}, "
        f"graph seed={SEDR_CLUSTER_GRAPH_SEED}"
    )

    print("  downstream Leiden:")

    print(
        f"    use_weights={SEDR_CLUSTER_USE_WEIGHTS}, "
        f"directed={SEDR_CLUSTER_DIRECTED}, "
        f"n_iterations={SEDR_CLUSTER_N_ITERATIONS}, "
        f"flavor={SEDR_CLUSTER_FLAVOR}"
    )

    print(
        f"    resolution search={SEDR_RES_START}..{SEDR_RES_END}, "
        f"increment={SEDR_RES_INCREMENT}, tolerance={SEDR_RES_TOLERANCE}, "
        f"seed={RESOLUTION_SEED}"
    )

    print("  downstream mclust:")

    print(
        f"    R mclust::Mclust, modelNames={MCLUST_MODEL}, "
        "G=annotation cluster count, no KNN/resolution"
    )

    print(
        f"    Leiden and mclust seeds={CLUSTER_BASE_SEED}.."
        f"{CLUSTER_BASE_SEED + ITERATIONS - 1}"
    )

    print(f"    mclust parallel workers={MCLUST_JOBS}")


def main():
    status = "error"
    try:
        write_hardware()
        print_sedr_settings()
        print(f"Timing events: {LOG_ROOT}", flush=True)
        with timed("r_worker_preflight"):
            if ALLOCATED_CPUS > 1:
                version_info = joblib.Parallel(n_jobs=2, backend="loky")(
                    joblib.delayed(mclust_worker_preflight)() for _ in range(2)
                )
            else:
                version_info = [mclust_worker_preflight()]
            print(f"R/mclust worker preflight: {version_info}")
        print(
            "\nPHASE 1/5: train SEDR once per sample; save embedding and Leiden graph"
        )
        with timed("phase_train_all"):
            if N_GPUS > 0:
                n_workers = min(N_GPUS, len(SAMPLES))
                batches = [SAMPLES[i::n_workers] for i in range(n_workers)]
                batch_results = joblib.Parallel(
                    n_jobs=n_workers,
                    backend="loky",
                    batch_size=1,
                    verbose=10,
                )(
                    joblib.delayed(train_gpu_batch)(batch, gpu_id)
                    for gpu_id, batch in enumerate(batches)
                )
                prepared = [item for batch in batch_results for item in batch]
            else:
                print("No CUDA GPU visible; SEDR training will run serially on CPU.")
                prepared = train_cpu_batch(SAMPLES)
        info_by_key = {x["key"]: x for x in prepared}
        if set(info_by_key) != {s.key for s in SAMPLES}:
            raise RuntimeError("SEDR preparation returned missing or duplicate samples")
        print("\nTargets derived from adata.obs['annotation']:")
        print({k: v["target_clusters"] for k, v in info_by_key.items()})

        print("\nPHASE 2/5: tune Leiden resolution on each frozen graph")
        with timed("phase_tune_all"):
            tuned = joblib.Parallel(
                n_jobs=min(RESOLUTION_JOBS, len(SAMPLES)),
                backend="loky",
                batch_size=1,
                verbose=10,
            )(
                joblib.delayed(tune_one_sample)(spec, info_by_key[spec.key])
                for spec in SAMPLES
            )
        resolutions = dict(tuned)
        if len(resolutions) != len(SAMPLES):
            raise RuntimeError("Missing Leiden resolution for at least one sample")
        for key, resolution in resolutions.items():
            info_by_key[key]["leiden_resolution"] = float(resolution)
        seed_tasks = [
            (spec, CLUSTER_BASE_SEED + i) for spec in SAMPLES for i in range(ITERATIONS)
        ]

        print(f"\nPHASE 3/5: {len(seed_tasks)} SEDR+Leiden fits; jobs={CLUSTER_JOBS}")
        with timed("phase_leiden_all"):
            leiden_records = joblib.Parallel(
                n_jobs=min(CLUSTER_JOBS, len(seed_tasks)),
                backend="loky",
                batch_size=1,
                verbose=10,
            )(
                joblib.delayed(leiden_one)(
                    spec, info_by_key[spec.key], resolutions[spec.key], seed
                )
                for spec, seed in seed_tasks
            )

        print(f"\nPHASE 4/5: {len(seed_tasks)} SEDR+mclust fits; jobs={MCLUST_JOBS}")
        with timed("phase_mclust_all"):
            mclust_records = joblib.Parallel(
                n_jobs=min(MCLUST_JOBS, len(seed_tasks)),
                backend="loky",
                batch_size=1,
                verbose=10,
            )(
                joblib.delayed(mclust_one)(spec, info_by_key[spec.key], seed)
                for spec, seed in seed_tasks
            )

        print("\nPHASE 5/5: consolidate Leiden and mclust hard labels")
        with timed("phase_consolidate_all"):
            for spec in SAMPLES:
                with timed("consolidate_leiden", sample=spec.key):
                    consolidate(spec, leiden_records, "leiden", info_by_key[spec.key])
                with timed("consolidate_mclust", sample=spec.key):
                    consolidate(spec, mclust_records, "mclust", info_by_key[spec.key])
        status = "ok"
        print("SEDR complete.")
    finally:
        log_timing("program_total", PROGRAM_STARTED, status=status)
        write_timing_summary()


if __name__ == "__main__":
    main()
