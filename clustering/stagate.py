#!/usr/bin/env python3
"Repeat Leiden and mclust clustering on fixed STAGATE representations."

from paths import DATA_ROOT as INPUT_DATA_ROOT, OUTPUT_ROOT

import os
import sys
import time

PROGRAM_STARTED = time.perf_counter()
from pathlib import Path

# Use the active environment's C++ runtime in worker processes.

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
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import gc
import inspect
import json
import platform
import resource
import socket
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
from scipy.spatial import cKDTree

import STAGATE_pyG as STAGATE

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from functions import (
    get_raw_counts,
    get_target_n_clusters,
    preprocessing,
    search_res,
)

DATA_ROOT = INPUT_DATA_ROOT
OUT_ROOT = OUTPUT_ROOT

METHOD_NAME = "stagate"
ANNOTATION_KEY = "annotation"

RUN_ID = os.environ.setdefault(
    "STAGATE_RUN_ID",
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}",
)
LOG_ROOT = OUT_ROOT / "run_logs" / METHOD_NAME / RUN_ID


def hardware_info():
    "Record the actual resources visible to this process/job."
    cpu_model = platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "available_cpus": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "total_ram_gib": round(
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30, 2
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node_mb": os.environ.get("SLURM_MEM_PER_NODE"),
        "slurm_mem_per_cpu_mb": os.environ.get("SLURM_MEM_PER_CPU"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gib": round(
                    torch.cuda.get_device_properties(i).total_memory / 2**30, 2
                ),
            }
            for i in range(torch.cuda.device_count())
        ],
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
        "scanpy_version": sc.__version__,
    }


def log_timing(
    stage, started, sample=None, seed=None, device="cpu", status="ok", **extra
):
    "One file per event avoids concurrent workers writing to the same log."
    if str(device).startswith("cuda") and status == "ok":
        torch.cuda.synchronize(torch.device(device))
    record = {
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
        **extra,
        "process_peak_rss_gib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3
        ),
    }
    if str(device).startswith("cuda") and status == "ok":
        record.update(
            gpu_name=torch.cuda.get_device_name(torch.device(device)),
            peak_gpu_allocated_gib=round(
                torch.cuda.max_memory_allocated(torch.device(device)) / 2**30, 3
            ),
            peak_gpu_reserved_gib=round(
                torch.cuda.max_memory_reserved(torch.device(device)) / 2**30, 3
            ),
        )
    directory = LOG_ROOT / (sample or "_run")
    directory.mkdir(parents=True, exist_ok=True)
    name = stage + (f"_seed{seed}" if seed is not None else "") + ".json"
    (directory / name).write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        f"TIME {stage} sample={sample or 'all'} seed={seed} "
        f"seconds={record['elapsed_seconds']:.3f} device={device} status={status}",
        flush=True,
    )


@contextmanager
def timed(stage, sample=None, seed=None, device="cpu"):
    started = time.perf_counter()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        log_timing(
            stage, started, sample=sample, seed=seed, device=device, status=status
        )


ITERATIONS = 50
CLUSTER_BASE_SEED = 42  # project-wide downstream clustering seeds: 42..91
STAGATE_TRAINING_SEED = 0  # released STAGATE_pyG train_STAGATE() default
RESOLUTION_SEED = 42  # fixed; Leiden resolution is tuned only once

MCLUST_MODEL = "EEE"

FORCE_RETRAIN = False

STAGATE_N_HVG = 3000
STAGATE_HVG_FLAVOR = "seurat_v3"
STAGATE_NORMALIZE_TARGET_SUM = 1e4
STAGATE_LOG1P = True  # report only; functions.preprocessing() calls sc.pp.log1p
STAGATE_SCALE = False  # official tutorials do not call sc.pp.scale()

STAGATE_SLIDE_MIN_CELLS_PER_GENE = 50

STAGATE_VISIUM_RADIUS_PITCH_MULTIPLIER = 1.2

STAGATE_SLIDE_TARGET_DEGREE = 10

STAGATE_HD_RADIUS_PITCH_MULTIPLIER = 1.5

SPATIAL_GRAPH_OVERRIDES = {}

STAGATE_HIDDEN_DIMS = [512, 30]
STAGATE_EPOCHS = 1000
STAGATE_LR = 0.001
STAGATE_GRADIENT_CLIPPING = 5.0
STAGATE_WEIGHT_DECAY = 0.0001
STAGATE_SAVE_LOSS = False
STAGATE_SAVE_RECONSTRUCTION = False
STAGATE_EMBEDDING_DIM = 30

STAGATE_GAT_HEADS = 1
STAGATE_GAT_DROPOUT = 0.0
STAGATE_GAT_ADD_SELF_LOOPS = False
STAGATE_GAT_BIAS = False
STAGATE_ACTIVATION = "ELU"

STAGATE_CLUSTER_N_NEIGHBORS = 15
STAGATE_CLUSTER_METRIC = "euclidean"
STAGATE_CLUSTER_CONNECTIVITY_METHOD = "umap"
STAGATE_CLUSTER_GRAPH_SEED = 0
STAGATE_CLUSTER_KNN = True

STAGATE_CLUSTER_USE_WEIGHTS = True
STAGATE_CLUSTER_DIRECTED = False
STAGATE_CLUSTER_N_ITERATIONS = -1
STAGATE_CLUSTER_FLAVOR = "leidenalg"

STAGATE_RES_START = 0.02
STAGATE_RES_END = 4.0
STAGATE_RES_INCREMENT = 0.01
STAGATE_RES_TOLERANCE = 2

RESOLUTION_OVERRIDES = {}

_available_cpus = (
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else (os.cpu_count() or 1)
)
ALLOCATED_CPUS = min(
    _available_cpus,
    max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", str(_available_cpus)))),
)
RESOLUTION_JOBS = min(18, ALLOCATED_CPUS)
CLUSTER_JOBS = min(192, ALLOCATED_CPUS)

MCLUST_JOBS = min(
    ALLOCATED_CPUS,
    max(1, int(os.environ.get("STAGATE_MCLUST_JOBS", "48"))),
)

VISIBLE_GPUS = torch.cuda.device_count()
REQUESTED_GPUS = int(os.environ.get("STAGATE_NUM_GPUS", str(VISIBLE_GPUS)))
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


def _seed_kw(func, seed):
    "Support Scanpy versions using either rng= or random_state=."
    params = inspect.signature(func).parameters
    if "rng" in params:
        return {"rng": int(seed)}
    return {"random_state": int(seed)}


def target_and_size(input_path: Path):
    backed = sc.read_h5ad(input_path, backed="r")
    try:
        target = get_target_n_clusters(backed, ANNOTATION_KEY)
        n_obs = int(backed.n_obs)
    finally:
        if backed.file is not None:
            backed.file.close()
    return int(target), n_obs


def spatial_graph_config(spec: SampleSpec):
    "Return a documented per-platform rule for STAGATE's radius graph."
    if spec.key in SPATIAL_GRAPH_OVERRIDES:
        cfg = dict(SPATIAL_GRAPH_OVERRIDES[spec.key])
        if (
            cfg.get("model") != "Radius"
            or not np.isfinite(cfg.get("rad_cutoff", np.nan))
            or float(cfg["rad_cutoff"]) <= 0
        ):
            raise ValueError(f"Invalid spatial graph override for {spec.key}: {cfg}")
        return cfg

    if spec.platform == "Slide-seq":
        return {
            "model": "Radius",
            "rule": "target_degree",
            "target_degree": STAGATE_SLIDE_TARGET_DEGREE,
        }

    if spec.platform == "Visium-HD":
        return {
            "model": "Radius",
            "rule": "square_first8",
            "target_degree": 8,
            "pitch_multiplier": STAGATE_HD_RADIUS_PITCH_MULTIPLIER,
        }

    return {
        "model": "Radius",
        "rule": "hex_first6",
        "target_degree": 6,
        "pitch_multiplier": STAGATE_VISIUM_RADIUS_PITCH_MULTIPLIER,
    }


def spatial_config_label(spec: SampleSpec) -> str:
    cfg = spatial_graph_config(spec)
    if "rad_cutoff" in cfg:
        return f"radius{cfg['rad_cutoff']:g}"
    if cfg["rule"] == "target_degree":
        return f"radius_targetdegree{cfg['target_degree']}"
    if cfg["rule"] == "square_first8":
        return f"radiusgrid8_pitchx{cfg['pitch_multiplier']:g}"
    return f"radiushex6_pitchx{cfg['pitch_multiplier']:g}"


def representation_tag(spec: SampleSpec) -> str:
    "Parameterized tag for the frozen STAGATE embedding shared by both branches."
    spatial_tag = spatial_config_label(spec)
    return (
        f"{spec.key}_stagate_fixed_"
        f"hvg{STAGATE_N_HVG}_spatial{spatial_tag}_"
        f"trainseed{STAGATE_TRAINING_SEED}_"
        f"hidden{STAGATE_HIDDEN_DIMS[0]}-{STAGATE_HIDDEN_DIMS[1]}_"
        f"epochs{STAGATE_EPOCHS}"
    )


def embedding_filename(spec: SampleSpec) -> str:
    return f"{representation_tag(spec)}_embedding.npy"


def graph_filename(spec: SampleSpec) -> str:
    "Leiden graph built from the exact saved STAGATE embedding."
    return (
        f"{representation_tag(spec)}_"
        f"clusterk{STAGATE_CLUSTER_N_NEIGHBORS}_{STAGATE_CLUSTER_METRIC}_"
        f"graphseed{STAGATE_CLUSTER_GRAPH_SEED}.npz"
    )


def preprocess_stagate(adata: ad.AnnData, spec: SampleSpec) -> ad.AnnData:
    "Apply the STAGATE paper/tutorial expression recipe once."
    if spec.platform == "Slide-seq":
        counts, source = get_raw_counts(adata, check_values=True)
        print(f"[{spec.key}] STAGATE raw-count source before Slide filter: {source}")
        if source != "adata.X":
            adata.X = counts.copy()
        sc.pp.filter_genes(adata, min_cells=STAGATE_SLIDE_MIN_CELLS_PER_GENE)

    adata = preprocessing(
        adata,
        highly_var=True,
        flavor=STAGATE_HVG_FLAVOR,
        n_hvg=STAGATE_N_HVG,
        check_raw_counts=True,
        normalize_target_sum=STAGATE_NORMALIZE_TARGET_SUM,
        scale=STAGATE_SCALE,
    )
    return adata


def radius_for_target_degree(tree: cKDTree, pitch: float, target: int) -> float:
    "Choose a radius from coordinates only, targeting mean graph degree."
    n_obs = tree.n
    if n_obs <= target:
        raise ValueError(
            f"Need more than {target} positions to target mean degree {target}"
        )

    zero_pairs = int(tree.count_neighbors(tree, 0.0))

    def mean_degree(radius):
        return (int(tree.count_neighbors(tree, radius)) - zero_pairs) / n_obs

    low, high = 0.0, float(pitch)
    for _ in range(40):
        if mean_degree(high) >= target:
            break
        high *= 2.0
    else:
        raise ValueError(
            "Could not find a finite spatial radius for the target graph density"
        )

    for _ in range(40):
        middle = (low + high) / 2.0
        if mean_degree(middle) < target:
            low = middle
        else:
            high = middle
    return float(np.nextafter(high, np.inf) * (1.0 + 1e-8))


def build_spatial_graph(adata: ad.AnnData, spec: SampleSpec):
    "Construct STAGATE's radius graph from the supplied spatial positions."
    cfg = spatial_graph_config(spec)
    coords = np.asarray(adata.obsm["spatial"])
    if coords.shape != (adata.n_obs, 2) or not np.isfinite(coords).all():
        raise ValueError(
            f"{spec.key}: spatial coordinates must be finite Nx2 positions"
        )
    if adata.n_obs < 2:
        raise ValueError(f"{spec.key}: need at least two spatial positions")

    pitch = None
    if "rad_cutoff" in cfg:
        radius = float(cfg["rad_cutoff"])
        print(f"[{spec.key}] documented manual spatial radius={radius:.6g}")
    else:
        tree = cKDTree(coords)
        unique_coords = np.unique(coords, axis=0)
        if len(unique_coords) < 2:
            raise ValueError(f"{spec.key}: fewer than two distinct spatial positions")
        if spec.platform != "Slide-seq" and len(unique_coords) != adata.n_obs:
            raise ValueError(f"{spec.key}: unexpected duplicate Visium/HD positions")
        if len(unique_coords) != adata.n_obs:
            print(
                f"[{spec.key}] {adata.n_obs - len(unique_coords):,} repeated "
                "Slide-seq coordinate entries; zero-distance edges excluded"
            )
        nearest = cKDTree(unique_coords).query(coords, k=2)[0][:, 1]
        if not np.isfinite(nearest).all() or np.any(nearest <= 0):
            raise ValueError(
                f"{spec.key}: cannot infer spacing between distinct positions"
            )
        pitch = float(np.median(nearest))
        if cfg["rule"] == "target_degree":
            radius = radius_for_target_degree(tree, pitch, cfg["target_degree"])
        else:
            radius = pitch * float(cfg["pitch_multiplier"])
        print(
            f"[{spec.key}] spatial radius rule={spatial_config_label(spec)}, "
            f"median nearest spacing={pitch:.6g}, radius={radius:.6g} "
            "(adata.obsm['spatial'] units)"
        )

    STAGATE.Cal_Spatial_Net(
        adata,
        rad_cutoff=radius,
        model="Radius",
        verbose=True,
    )

    if "Spatial_Net" not in adata.uns:
        raise RuntimeError(
            f"{spec.key}: STAGATE did not create adata.uns['Spatial_Net']"
        )

    n_edges = int(adata.uns["Spatial_Net"].shape[0])
    avg_neighbors = n_edges / float(adata.n_obs)
    print(
        f"[{spec.key}] STAGATE spatial graph: {spatial_config_label(spec)}, "
        f"radius={radius:.6g}, target neighbors={cfg.get('target_degree', 'manual')}, "
        f"edges={n_edges:,}, average neighbors={avg_neighbors:.3f}"
    )
    if spec.platform in {"Visium", "Visium-HD", "Slide-seq"}:
        degrees = (
            adata.uns["Spatial_Net"]["Cell1"]
            .value_counts()
            .reindex(adata.obs_names, fill_value=0)
            .to_numpy()
        )
        print(
            f"[{spec.key}] spatial graph QC: median degree={np.median(degrees):.0f}, "
            f"5th/95th percentile={np.percentile(degrees, 5):.0f}/{np.percentile(degrees, 95):.0f}, "
            f"fraction with 0 neighbors={np.mean(degrees == 0):.1%}"
        )

    if spec.platform == "Visium" and not (4.0 <= avg_neighbors <= 6.5):
        print(
            f"WARNING [{spec.key}]: radius={radius:.6g} gives "
            f"{avg_neighbors:.2f} neighbors/spot; STAGATE's paper describes "
            "about 6 for 10x Visium. Inspect the coordinates/graph."
        )
    elif spec.platform == "Slide-seq" and not (6.0 <= avg_neighbors <= 15.0):
        print(
            f"WARNING [{spec.key}]: radius={radius:.6g} gives "
            f"{avg_neighbors:.2f} neighbors/spot; STAGATE recommends roughly "
            "6-15 for non-Visium platforms."
        )
    elif spec.platform == "Visium-HD" and not (4.0 <= avg_neighbors <= 8.5):
        print(
            f"WARNING [{spec.key}]: HD grid rule gives {avg_neighbors:.2f} "
            "neighbors/bin; inspect coordinate scaling, filtering, and missing bins."
        )

    return {
        "sample": spec.key,
        "platform": spec.platform,
        "model": "Radius",
        "selection_rule": spatial_config_label(spec),
        "coordinate_key": "adata.obsm['spatial']",
        "median_nearest_spacing": pitch,
        "radius_in_coordinate_units": radius,
        "nominal_neighbor_target": cfg.get("target_degree"),
        "n_obs": adata.n_obs,
        "n_directed_edges": n_edges,
        "average_neighbors": avg_neighbors,
    }


def make_cluster_adata(embedding: np.ndarray) -> ad.AnnData:
    n_obs = embedding.shape[0]
    tmp = ad.AnnData(
        X=sparse.csr_matrix((n_obs, 0), dtype=np.float32),
        obs=pd.DataFrame(index=pd.RangeIndex(n_obs).astype(str)),
    )

    tmp.obsm["STAGATE"] = np.array(embedding, copy=True, order="C")
    return tmp


def build_leiden_graph_from_embedding(embedding: np.ndarray, graph_path: Path):
    "Build and persist the one fixed STAGATE->Leiden graph."
    cluster_adata = make_cluster_adata(embedding)
    neighbor_kwargs = dict(
        n_neighbors=STAGATE_CLUSTER_N_NEIGHBORS,
        use_rep="STAGATE",
        metric=STAGATE_CLUSTER_METRIC,
        method=STAGATE_CLUSTER_CONNECTIVITY_METHOD,
        knn=STAGATE_CLUSTER_KNN,
        key_added="shared_graph",
    )
    neighbor_kwargs.update(_seed_kw(sc.pp.neighbors, STAGATE_CLUSTER_GRAPH_SEED))
    sc.pp.neighbors(cluster_adata, **neighbor_kwargs)

    conn_key = cluster_adata.uns["shared_graph"]["connectivities_key"]
    graph = cluster_adata.obsp[conn_key].tocsr(copy=True)
    graph.sort_indices()
    sparse.save_npz(graph_path, graph, compressed=True)
    return graph


def train_and_build_one(spec: SampleSpec, device):
    sample_started = time.perf_counter()
    input_path = DATA_ROOT / spec.h5ad_name
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with timed("input_metadata", spec.key):
        target, n_obs = target_and_size(input_path)
    if n_obs < 1000:
        raise ValueError(f"{spec.key}: only {n_obs} observations; expected >=1000")

    method_dir = OUT_ROOT / spec.output_group / METHOD_NAME
    embedding_dir = method_dir / "shared_embedding"
    graph_dir = method_dir / "shared_graph"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = embedding_dir / embedding_filename(spec)
    graph_path = graph_dir / graph_filename(spec)

    if embedding_path.exists() and not FORCE_RETRAIN:
        embedding = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        if (
            embedding.ndim == 2
            and embedding.shape[0] == n_obs
            and embedding.shape[1] == STAGATE_EMBEDDING_DIM
        ):
            embedding_dim = int(embedding.shape[1])
            graph_ok = False
            if graph_path.exists():
                graph = sparse.load_npz(graph_path).tocsr()
                graph_ok = graph.shape == (n_obs, n_obs)
                if graph_ok:
                    print(
                        f"[{spec.key}] reuse exact STAGATE embedding + Leiden graph: "
                        f"n={n_obs:,}, dim={embedding_dim}, target={target}"
                    )
                else:
                    print(
                        f"[{spec.key}] cached Leiden graph shape={graph.shape}; "
                        "rebuilding from the saved STAGATE embedding"
                    )
                del graph

            if not graph_ok:
                with timed("leiden_graph", spec.key):
                    graph = build_leiden_graph_from_embedding(embedding, graph_path)
                print(
                    f"[{spec.key}] rebuilt Leiden graph from saved STAGATE embedding: "
                    f"nodes={graph.shape[0]:,}, nnz={graph.nnz:,}"
                )
                del graph

            del embedding
            log_timing(
                "sample_total",
                sample_started,
                sample=spec.key,
                device="cpu",
                cache="reused",
            )
            return {
                "key": spec.key,
                "embedding_path": str(embedding_path),
                "graph_path": str(graph_path),
                "n_obs": n_obs,
                "target_clusters": target,
                "platform": spec.platform,
                "spatial_config": spatial_config_label(spec),
                "embedding_dim": embedding_dim,
            }

        print(
            f"[{spec.key}] cached STAGATE embedding shape={embedding.shape} is invalid; "
            "retraining STAGATE once"
        )
        del embedding

    elif graph_path.exists() and not FORCE_RETRAIN:
        print(
            f"[{spec.key}] old Leiden graph exists but exact STAGATE embedding is missing; "
            "retraining once so Leiden and mclust share the same embedding"
        )

    print(
        f"[{spec.key}] STAGATE train ONCE on {device}; n={n_obs:,}, "
        f"target={target}, platform={spec.platform}, spatial={spatial_config_label(spec)}"
    )

    with timed("read_input", spec.key):
        adata = sc.read_h5ad(input_path)
    with timed("preprocessing", spec.key):
        adata = preprocess_stagate(adata, spec)
    with timed("spatial_graph", spec.key):
        spatial_report = build_spatial_graph(adata, spec)
    spatial_report_path = (
        embedding_dir / f"{representation_tag(spec)}_spatial_graph.json"
    )
    with timed("spatial_report_save", spec.key):
        with spatial_report_path.open("w", encoding="utf-8") as handle:
            json.dump(spatial_report, handle, indent=2)
            handle.write("\n")
    print(f"[{spec.key}] saved spatial radius and graph density: {spatial_report_path}")

    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    with timed("stagate_training", spec.key, device=device):
        adata = STAGATE.train_STAGATE(
            adata,
            hidden_dims=list(STAGATE_HIDDEN_DIMS),
            n_epochs=STAGATE_EPOCHS,
            lr=STAGATE_LR,
            key_added="STAGATE",
            gradient_clipping=STAGATE_GRADIENT_CLIPPING,
            weight_decay=STAGATE_WEIGHT_DECAY,
            verbose=True,
            random_seed=STAGATE_TRAINING_SEED,
            save_loss=STAGATE_SAVE_LOSS,
            save_reconstrction=STAGATE_SAVE_RECONSTRUCTION,
            device=torch.device(device),
        )

    embedding = np.asarray(adata.obsm["STAGATE"])
    if (
        embedding.ndim != 2
        or embedding.shape[0] != n_obs
        or embedding.shape[1] != STAGATE_EMBEDDING_DIM
    ):
        raise RuntimeError(
            f"{spec.key}: unexpected STAGATE embedding shape {embedding.shape}; "
            f"expected ({n_obs}, {STAGATE_EMBEDDING_DIM})"
        )

    with timed("embedding_save", spec.key):
        np.save(embedding_path, embedding, allow_pickle=False)
    embedding_shape = tuple(embedding.shape)
    embedding_dim = int(embedding.shape[1])
    del embedding

    with timed("leiden_graph", spec.key):
        frozen_embedding = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        graph = build_leiden_graph_from_embedding(frozen_embedding, graph_path)

    print(
        f"[{spec.key}] exact shared STAGATE embedding saved: {embedding_path.name}; "
        f"shape={embedding_shape}; Leiden graph nnz={graph.nnz:,}"
    )

    del graph, frozen_embedding, adata
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    log_timing(
        "sample_total",
        sample_started,
        sample=spec.key,
        device=f"cpu+{device}" if str(device).startswith("cuda") else "cpu",
        cache="trained",
        training_device=str(device),
    )

    return {
        "key": spec.key,
        "embedding_path": str(embedding_path),
        "graph_path": str(graph_path),
        "n_obs": n_obs,
        "target_clusters": target,
        "platform": spec.platform,
        "spatial_config": spatial_config_label(spec),
        "embedding_dim": embedding_dim,
    }


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


def check_cpu_worker_imports():
    "Catch SciPy/worker import errors before GPU training starts."
    import scipy.linalg
    from functions import search_res as project_search_res

    return scipy.__version__, project_search_res.__module__


def tune_one_sample(spec: SampleSpec, info: dict):
    if spec.key in RESOLUTION_OVERRIDES:
        started = time.perf_counter()
        res = float(RESOLUTION_OVERRIDES[spec.key])
        print(f"[{spec.key}] manual STAGATE-Leiden resolution={res:.4f}")
        log_timing("resolution_override", started, sample=spec.key, resolution=res)
        return spec.key, res

    with timed("resolution_search", spec.key):
        graph = sparse.load_npz(info["graph_path"]).tocsr()
        target = int(info["target_clusters"])

        res = search_res(
            adjacency=graph,
            n_clusters=target,
            method="leiden",
            start=STAGATE_RES_START,
            end=STAGATE_RES_END,
            increment=STAGATE_RES_INCREMENT,
            tolerance=STAGATE_RES_TOLERANCE,
            random_state=RESOLUTION_SEED,
            directed=STAGATE_CLUSTER_DIRECTED,
            use_weights=STAGATE_CLUSTER_USE_WEIGHTS,
            leiden_n_iterations=STAGATE_CLUSTER_N_ITERATIONS,
            leiden_flavor=STAGATE_CLUSTER_FLAVOR,
            verbose=False,
        )

    print(
        f"[{spec.key}] STAGATE fixed embedding/graph: target={target}; "
        f"Leiden resolution={res:.4f}"
    )
    return spec.key, float(res)


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


def leiden_one(spec: SampleSpec, info: dict, resolution: float, seed: int):
    started = time.perf_counter()
    graph = load_graph_cached(info["graph_path"])
    tmp = empty_adata(info["n_obs"])
    kwargs = dict(
        adjacency=graph,
        resolution=float(resolution),
        key_added="domain",
        directed=STAGATE_CLUSTER_DIRECTED,
        use_weights=STAGATE_CLUSTER_USE_WEIGHTS,
        n_iterations=STAGATE_CLUSTER_N_ITERATIONS,
        flavor=STAGATE_CLUSTER_FLAVOR,
    )
    kwargs.update(_seed_kw(sc.tl.leiden, seed))
    sc.tl.leiden(tmp, **kwargs)
    labels = tmp.obs["domain"].astype(int).to_numpy()

    method_dir = OUT_ROOT / spec.output_group / METHOD_NAME
    temp_dir = method_dir / "temp_leiden"
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_path = temp_dir / f"{spec.key}-stagate-leiden-seed{seed}.pkl"
    joblib.dump(labels, out_path, compress=0)
    log_timing(
        "leiden_run",
        started,
        sample=spec.key,
        seed=seed,
        clusters=int(np.unique(labels).size),
    )
    return spec.key, int(seed), str(out_path), int(np.unique(labels).size)


def _get_mclust_backend():
    "Lazy-load one R/mclust runtime per loky worker process."
    global _MCLUST_BACKEND
    if _MCLUST_BACKEND is not None:
        return _MCLUST_BACKEND

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import default_converter, numpy2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        importr("mclust")

        r_mclust = ro.r(
            "function(x, G, modelNames) mclust::Mclust(x, G=G, modelNames=modelNames)"
        )
        r_set_seed = ro.r["set.seed"]
    except Exception as exc:
        raise RuntimeError(
            "STAGATE+mclust requires a working R installation, rpy2, and the "
            "R package 'mclust' inside the STAGATE environment."
        ) from exc

    _MCLUST_BACKEND = (
        ro,
        default_converter,
        numpy2ri,
        localconverter,
        r_mclust,
        r_set_seed,
    )
    return _MCLUST_BACKEND


def mclust_one(spec: SampleSpec, info: dict, seed: int):
    "Run R mclust on the exact same native STAGATE embedding used by Leiden."
    started = time.perf_counter()
    embedding = load_embedding_cached(info["embedding_path"])
    target = int(info["target_clusters"])

    (
        ro,
        default_converter,
        numpy2ri,
        localconverter,
        r_mclust,
        r_set_seed,
    ) = _get_mclust_backend()

    r_set_seed(int(seed))

    r_embedding = ro.r["matrix"](
        ro.FloatVector(embedding.ravel(order="F")),
        nrow=int(embedding.shape[0]),
        ncol=int(embedding.shape[1]),
    )
    fit = r_mclust(
        r_embedding,
        ro.IntVector([target]),
        ro.StrVector([MCLUST_MODEL]),
    )
    classification = fit.rx2("classification")
    with localconverter(default_converter + numpy2ri.converter):
        labels = np.asarray(
            ro.conversion.rpy2py(classification),
            dtype=np.int64,
        ).reshape(-1)

    if labels.shape[0] != info["n_obs"]:
        raise RuntimeError(
            f"{spec.key}: mclust returned {labels.shape[0]} labels for "
            f"{info['n_obs']} observations"
        )

    method_dir = OUT_ROOT / spec.output_group / METHOD_NAME
    temp_dir = method_dir / "temp_mclust"
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_path = temp_dir / f"{spec.key}-stagate-mclust-seed{seed}.pkl"
    joblib.dump(labels, out_path, compress=0)
    log_timing(
        "mclust_run",
        started,
        sample=spec.key,
        seed=seed,
        clusters=int(np.unique(labels).size),
    )
    return spec.key, int(seed), str(out_path), int(np.unique(labels).size)


def consolidate(spec: SampleSpec, records, clustering_method: str):
    started = time.perf_counter()
    recs = sorted(
        [r for r in records if r[0] == spec.key],
        key=lambda r: r[1],
    )
    if len(recs) != ITERATIONS:
        raise RuntimeError(
            f"{spec.key}: expected {ITERATIONS} {clustering_method} runs, got {len(recs)}"
        )

    results = [joblib.load(r[2]) for r in recs]
    method_dir = OUT_ROOT / spec.output_group / METHOD_NAME
    method_dir.mkdir(parents=True, exist_ok=True)

    if clustering_method == "leiden":
        if spec.output_group == "dlpfc":
            section = spec.key.replace("dlpfc_", "")
            name = f"stagate_results_{section}_{ITERATIONS}iterations.pkl"
        else:
            name = f"stagate_results_{ITERATIONS}iterations.pkl"
    elif clustering_method == "mclust":
        if spec.output_group == "dlpfc":
            section = spec.key.replace("dlpfc_", "")
            name = f"stagate_mclust_results_{section}_{ITERATIONS}iterations.pkl"
        else:
            name = f"stagate_mclust_results_{ITERATIONS}iterations.pkl"
    else:
        raise ValueError(f"Unknown clustering_method={clustering_method!r}")

    joblib.dump(results, method_dir / name, compress=0)
    log_timing(f"consolidate_{clustering_method}", started, sample=spec.key)


def print_stagate_settings():
    print(
        "STAGATE fixed-representation experiment: Leiden + mclust on one shared embedding"
    )
    print("  preprocessing:")
    print(f"    HVGs={STAGATE_N_HVG} ({STAGATE_HVG_FLAVOR})")
    print(
        f"    normalize_total={STAGATE_NORMALIZE_TARGET_SUM:g}, log1p={STAGATE_LOG1P}, scale={STAGATE_SCALE}"
    )
    print(f"    Slide-seq gene filter: min_cells={STAGATE_SLIDE_MIN_CELLS_PER_GENE}")
    print("  STAGATE spatial graph: model=Radius for EVERY sample")
    print(
        "    standard Visium: radius=median nearest spacing "
        f"x {STAGATE_VISIUM_RADIUS_PITCH_MULTIPLIER:g} (hex first-ring rule)"
    )
    print(
        "    Slide-seq: coordinate-derived radius reaching mean degree "
        f"{STAGATE_SLIDE_TARGET_DEGREE} (within paper's 6-15 guidance)"
    )
    print(
        "    cropped Visium HD: radius=median nearest-bin spacing "
        f"x {STAGATE_HD_RADIUS_PITCH_MULTIPLIER:g} (square-grid project rule)"
    )
    print("  STAGATE_pyG training (ONCE):")
    print(f"    seed={STAGATE_TRAINING_SEED}, hidden_dims={STAGATE_HIDDEN_DIMS}")
    print(f"    epochs={STAGATE_EPOCHS}, lr={STAGATE_LR}")
    print(
        f"    weight_decay={STAGATE_WEIGHT_DECAY}, gradient_clipping={STAGATE_GRADIENT_CLIPPING}"
    )
    print("  shared representation:")
    print(
        "    exact native STAGATE embedding is saved once and reused by BOTH Leiden and mclust"
    )
    print("  Leiden-only fixed graph:")
    print(
        f"    KNN={STAGATE_CLUSTER_N_NEIGHBORS}, metric={STAGATE_CLUSTER_METRIC}, "
        f"connectivity={STAGATE_CLUSTER_CONNECTIVITY_METHOD}, graph seed={STAGATE_CLUSTER_GRAPH_SEED}"
    )
    print("  downstream Leiden:")
    print(
        f"    use_weights={STAGATE_CLUSTER_USE_WEIGHTS}, directed={STAGATE_CLUSTER_DIRECTED}, "
        f"n_iterations={STAGATE_CLUSTER_N_ITERATIONS}, flavor={STAGATE_CLUSTER_FLAVOR}"
    )
    print(
        f"    resolution search={STAGATE_RES_START}..{STAGATE_RES_END}, "
        f"increment={STAGATE_RES_INCREMENT}, "
        f"cluster-count tolerance={STAGATE_RES_TOLERANCE}, seed={RESOLUTION_SEED}"
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
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    hardware = hardware_info()
    hardware.update(
        run_id=RUN_ID,
        resolution_jobs=RESOLUTION_JOBS,
        leiden_jobs=CLUSTER_JOBS,
        mclust_jobs=MCLUST_JOBS,
        selected_gpus=N_GPUS,
        worker_libstdcxx=(str(WORKER_LIBSTDCXX) if USING_ENV_LIBSTDCXX else None),
    )
    (LOG_ROOT / "hardware.json").write_text(
        json.dumps(hardware, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Hardware and timing logs: {LOG_ROOT}")
    print(f"Hardware: {json.dumps(hardware, default=str)}", flush=True)
    print_stagate_settings()

    with timed("worker_import_preflight"):
        joblib.Parallel(n_jobs=min(2, ALLOCATED_CPUS), backend="loky")(
            joblib.delayed(check_cpu_worker_imports)()
            for _ in range(min(2, ALLOCATED_CPUS))
        )

    print(
        "\nPHASE 1/5: train STAGATE once per sample; save shared embedding + Leiden graph"
    )
    with timed("phase1_prepare", device="cpu+gpu" if N_GPUS else "cpu"):
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
            print("No CUDA GPU visible; STAGATE training will run serially on CPU.")
            prepared = train_cpu_batch(SAMPLES)

    info_by_key = {x["key"]: x for x in prepared}
    print("\nTargets derived from adata.obs['annotation']:")
    print({k: v["target_clusters"] for k, v in info_by_key.items()})

    print("\nPHASE 2/5: tune STAGATE+Leiden resolution on the frozen graph")
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

    seed_tasks = [
        (spec, CLUSTER_BASE_SEED + i) for spec in SAMPLES for i in range(ITERATIONS)
    ]

    leiden_tasks = [(spec, resolutions[spec.key], seed) for spec, seed in seed_tasks]
    print(
        f"\nPHASE 3/5: submitting {len(leiden_tasks)} STAGATE+Leiden jobs "
        f"with n_jobs={CLUSTER_JOBS}"
    )
    with timed("phase3_leiden"):
        leiden_records = joblib.Parallel(
            n_jobs=CLUSTER_JOBS,
            backend="loky",
            batch_size=1,
            verbose=10,
        )(
            joblib.delayed(leiden_one)(spec, info_by_key[spec.key], resolution, seed)
            for spec, resolution, seed in leiden_tasks
        )

    print(
        f"\nPHASE 4/5: submitting {len(seed_tasks)} STAGATE+mclust jobs "
        f"with n_jobs={MCLUST_JOBS}"
    )
    with timed("phase4_mclust"):
        mclust_records = joblib.Parallel(
            n_jobs=MCLUST_JOBS,
            backend="loky",
            batch_size=1,
            verbose=10,
        )(
            joblib.delayed(mclust_one)(spec, info_by_key[spec.key], seed)
            for spec, seed in seed_tasks
        )

    print("\nPHASE 5/5: consolidate Leiden and mclust outputs")
    with timed("phase5_consolidate"):
        for spec in SAMPLES:
            consolidate(spec, leiden_records, "leiden")
            consolidate(spec, mclust_records, "mclust")

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        log_timing(
            "total_wall",
            PROGRAM_STARTED,
            device="cpu+gpu" if N_GPUS else "cpu",
            status="error",
        )
        raise
    else:
        log_timing("total_wall", PROGRAM_STARTED, device="cpu+gpu" if N_GPUS else "cpu")
    finally:
        files = [p for p in LOG_ROOT.rglob("*.json") if p.name != "hardware.json"]
        if files:
            records = [json.loads(p.read_text(encoding="utf-8")) for p in files]
            pd.DataFrame(records).sort_values(["stage", "sample", "seed"]).to_csv(
                LOG_ROOT / "timings.csv", index=False
            )
            print(f"Timing summary: {LOG_ROOT / 'timings.csv'}", flush=True)
