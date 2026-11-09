#!/usr/bin/env python3
"Repeat Louvain clustering and spatial smoothing on fixed SpiceMix representations."

from paths import DATA_ROOT as INPUT_DATA_ROOT, OUTPUT_ROOT
import argparse
import os
import sys
import time

PROGRAM_START = time.perf_counter()
from pathlib import Path

# Use the active environment's C++ runtime in worker processes.

WORKER_LIBSTDCXX = Path(sys.prefix) / "lib" / "libstdc++.so.6"
USING_ENV_LIBSTDCXX = (
    WORKER_LIBSTDCXX.is_file() and b"GLIBCXX_3.4.26" in WORKER_LIBSTDCXX.read_bytes()
)
if USING_ENV_LIBSTDCXX:
    _paths = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    if _paths[0] == str(WORKER_LIBSTDCXX.parent):
        _paths.pop(0)
        if _paths:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(_paths)
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
import json
import logging
import multiprocessing
import platform
import resource
import socket
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack, contextmanager
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
from sklearn.preprocessing import StandardScaler

from spicemix_external import (
    SpiceMix,
    config_logger,
    smooth_continuous,
    smooth_discrete,
)

from functions import search_res, _cluster_count_from_adjacency

logger = config_logger(logging.getLogger(__name__))
torch.set_num_threads(1)

SPICEMIX_INPUT_ROOT = OUTPUT_ROOT

OUT_ROOT = OUTPUT_ROOT
METHOD_NAME = "spicemix"
ANNOTATION_KEY = "annotation"
RUN_ID = os.environ.setdefault(
    "SPICEMIX_RUN_ID",
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}",
)
LOG_ROOT = OUT_ROOT / "run_logs" / METHOD_NAME / RUN_ID


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
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "worker_libstdcxx": str(WORKER_LIBSTDCXX) if USING_ENV_LIBSTDCXX else None,
        "gpu_devices": [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_gib": round(
                    torch.cuda.get_device_properties(i).total_memory / 2**30, 2
                ),
            }
            for i in range(torch.cuda.device_count())
        ],
        "numpy_version": np.__version__,
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
        "scanpy_version": sc.__version__,
    }


def log_timing(
    stage, started, sample="all", seed=None, device="cpu", status="ok", **extra
):
    "Write each record independently, so parallel workers never overwrite."
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
        "process_peak_rss_gib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3
        ),
        **extra,
    }
    if str(device).startswith("cuda") and torch.cuda.is_initialized():
        dev = torch.device(device)
        record.update(
            gpu_name=torch.cuda.get_device_name(dev),
            peak_gpu_allocated_gib=round(
                torch.cuda.max_memory_allocated(dev) / 2**30, 3
            ),
            peak_gpu_reserved_gib=round(
                torch.cuda.max_memory_reserved(dev) / 2**30, 3
            ),
        )
    events = LOG_ROOT / "timing_events"
    events.mkdir(parents=True, exist_ok=True)
    event_file = (
        events / f"{sample}_{stage}_{seed}_{os.getpid()}_{uuid.uuid4().hex}.json"
    )
    event_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        f"TIME {stage} sample={sample} seed={seed} "
        f"seconds={record['elapsed_seconds']:.3f} device={device} status={status}",
        flush=True,
    )


@contextmanager
def timed(stage, sample="all", seed=None, device="cpu"):
    started = time.perf_counter()
    status = "ok"
    try:
        yield
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
    except BaseException:
        status = "error"
        raise
    finally:
        log_timing(
            stage, started, sample=sample, seed=seed, device=device, status=status
        )


def write_timing_summary():
    "Aggregate the durable worker records, including records after failures."
    paths = (LOG_ROOT / "timing_events").glob("*.json")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if rows:
        pd.DataFrame(rows).sort_values("finished_utc").to_csv(
            LOG_ROOT / "timings.csv", index=False
        )
        print("Timing summary:", LOG_ROOT / "timings.csv", flush=True)


ITERATIONS = 50
CLUSTER_BASE_SEED = 42  # final stochastic seeds: 42..91
RESOLUTION_SEED = 42  # fixed; resolution is selected only once
SPICEMIX_TRAINING_SEED = 0  # current CLI default / official DLPFC notebook

FORCE_RETRAIN = False
COORDINATE_SCALING_TAG = "spacing_v1_gene10_platformknn_v1"

INPUT_MIN_NONZERO_FRACTION = 0.10
INPUT_LIBRARY_SIZE = 1e4

SPICEMIX_DEFAULT_K = 15
SPICEMIX_DLPFC_K = 10
SPICEMIX_K_OVERRIDES = {}
SPICEMIX_LAMBDA_SIGMA_X_INV = 1e-6
SPICEMIX_POWER_SIGMA_X_INV = 2
SPICEMIX_INIT_METHOD = "louvain"
SPICEMIX_INIT_NUM_PCS = 50
SPICEMIX_INIT_N_NEIGHBORS = 20
# Use the wider initialization search only when training a new model.

SPICEMIX_INIT_RESOLUTION_BOUNDARIES = (0.5, 4.0)
SPICEMIX_INIT_NUM_RS = 10
SPICEMIX_INIT_NMF_ITER = 10
SPICEMIX_MAX_ITER = 200
SPICEMIX_DTYPE = torch.float64

SPICEMIX_CONTINUOUS_SMOOTH_RADIUS = 2.0
SPICEMIX_CONTINUOUS_SMOOTH_RADIUS_EPS = 1e-3
SPICEMIX_CONTINUOUS_SMOOTH_LAMBDA = 1.0
SPICEMIX_DISCRETE_SMOOTH_RADIUS = 2.0
SPICEMIX_DISCRETE_SMOOTH_RADIUS_EPS = 1e-3

SPICEMIX_CLUSTER_N_NEIGHBORS = 20
SPICEMIX_CLUSTER_METRIC = "euclidean"  # Scanpy default used by util.py
SPICEMIX_CLUSTER_METHOD = "umap"  # connectivity kernel, not UMAP plot
SPICEMIX_CLUSTER_GRAPH_SEED = 0  # Scanpy 1.8.2 default used by util.py
SPICEMIX_CLUSTER_KNN = True

SPICEMIX_LOUVAIN_DIRECTED = True
SPICEMIX_LOUVAIN_USE_WEIGHTS = False
SPICEMIX_LOUVAIN_FLAVOR = "vtraag"

SPICEMIX_RES_START = 0.1
SPICEMIX_RES_END = 4.0
SPICEMIX_RES_INCREMENT = 0.01
SPICEMIX_RES_TOLERANCE = 2
SPICEMIX_LOW_RES_START = 0.001
SPICEMIX_LOW_RES_INCREMENT = 0.001
RESOLUTION_OVERRIDES = {}

ALLOCATED_CPUS = (
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else (os.cpu_count() or 1)
)
if os.environ.get("SLURM_CPUS_PER_TASK"):
    ALLOCATED_CPUS = min(ALLOCATED_CPUS, int(os.environ["SLURM_CPUS_PER_TASK"]))

CPU_PIPELINE_JOBS = min(
    16,
    ALLOCATED_CPUS,
    max(1, int(os.environ.get("SPICEMIX_CLUSTER_JOBS", "8"))),
)
VISIBLE_GPUS = torch.cuda.device_count()
REQUESTED_GPUS = int(os.environ.get("SPICEMIX_NUM_GPUS", str(VISIBLE_GPUS)))
N_GPUS = min(VISIBLE_GPUS, max(0, REQUESTED_GPUS))
CPU_TRAIN_JOBS = max(1, int(os.environ.get("SPICEMIX_CPU_TRAIN_JOBS", "1")))


@dataclass(frozen=True)
class SampleSpec:
    key: str
    dataset: str
    output_group: str
    sample: str | None = None


SAMPLES = [
    SampleSpec("mouse_brain", "mouse_brain", "mouse_brain"),
    SampleSpec("human_breast_cancer", "human_breast_cancer", "human_breast_cancer"),
    SampleSpec(
        "mouse_brain_cerebellum", "mouse_brain_cerebellum", "mouse_brain_cerebellum"
    ),
    SampleSpec(
        "Visium_HD_Human_Colon_Cancer_cropped_square",
        "Visium_HD_Human_Colon_Cancer_cropped_square",
        "Visium_HD_Human_Colon_Cancer_cropped_square",
    ),
    SampleSpec("ov_ffpe", "ov_ffpe", "ov_ffpe"),
    SampleSpec("coad_ffpe", "coad_ffpe", "coad_ffpe"),
]
for _sample in [
    "151507",
    "151508",
    "151509",
    "151510",
    "151669",
    "151670",
    "151671",
    "151672",
    "151673",
    "151674",
    "151675",
    "151676",
]:
    SAMPLES.append(
        SampleSpec(
            key=f"dlpfc_{_sample}",
            dataset="dlpfc",
            output_group="dlpfc",
            sample=_sample,
        )
    )


def sample_input_dir(spec: SampleSpec) -> Path:
    base = SPICEMIX_INPUT_ROOT / spec.dataset / METHOD_NAME
    return base / spec.sample if spec.sample is not None else base


def repli_name(spec: SampleSpec) -> str:
    return f"{spec.dataset}_{spec.sample}" if spec.sample is not None else spec.dataset


def method_output_dir(spec: SampleSpec) -> Path:
    return OUT_ROOT / spec.output_group / METHOD_NAME


def spice_k(spec: SampleSpec) -> int:
    if spec.key in SPICEMIX_K_OVERRIDES:
        return int(SPICEMIX_K_OVERRIDES[spec.key])
    if spec.dataset == "dlpfc":
        return SPICEMIX_DLPFC_K
    return SPICEMIX_DEFAULT_K


def expected_input_spatial_k(spec: SampleSpec) -> int:
    if spec.key == "mouse_brain_cerebellum":
        return 10
    if spec.key in {
        "Visium_HD_Human_Colon_Cancer_cropped_square",
        "coad_ffpe",
        "ov_ffpe",
    }:
        return 8
    if spec.key in {"human_breast_cancer", "mouse_brain"} or spec.dataset == "dlpfc":
        return 6
    raise ValueError(f"No expected spatial kNN rule for {spec.key!r}")


def load_meta_only(spec: SampleSpec) -> pd.DataFrame:
    "Load the same meta table SpiceMix.load_dataset() would use."
    files_dir = sample_input_dir(spec) / "files"
    r = repli_name(spec)
    pkl_path = files_dir / f"meta_{r}.pkl"
    csv_path = files_dir / f"meta_{r}.csv"
    qc_path = files_dir / f"spicemix_coordinate_qc_{r}.json"
    if pkl_path.exists():
        raise FileExistsError(f"Stale .pkl could bypass scaled .csv: {pkl_path}")
    if not qc_path.exists():
        raise FileNotFoundError(
            f"Missing scaled-coordinate QC file {qc_path}; rerun spicemix_prepare_input.py"
        )
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    expected_k = expected_input_spatial_k(spec)
    if (
        qc.get("spatial_graph_type") != "undirected_union_of_euclidean_knn"
        or qc.get("requested_knn_neighbors") != expected_k
        or not np.isclose(
            qc.get("gene_filter_min_nonzero_fraction", np.nan),
            INPUT_MIN_NONZERO_FRACTION,
            rtol=0,
            atol=1e-12,
        )
        or not np.isclose(
            qc.get("normalization_library_size", np.nan),
            INPUT_LIBRARY_SIZE,
            rtol=0,
            atol=1e-9,
        )
    ):
        raise ValueError(
            f"{r}: prepared input does not match the 10% gene filter, "
            f"library size 1e4, and spatial kNN={expected_k}; "
            "rerun spicemix_prepare_input.py before training"
        )
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Could not find {pkl_path.name} or {csv_path.name} in {files_dir}"
        )
    df = df.copy()
    if qc.get("n_observations") != len(df):
        raise ValueError(f"{r}: metadata and exporter QC observation counts disagree")
    coords = df[["coor X", "coor Y"]].to_numpy(dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError(f"{r}: nonfinite or invalid smoothing coordinates")
    unique_coords = np.unique(coords, axis=0)
    if len(unique_coords) < 2:
        raise ValueError(f"{r}: fewer than two unique smoothing positions")
    pitch = float(np.median(cKDTree(unique_coords).query(coords, k=2)[0][:, 1]))
    if not 0.99 <= pitch <= 1.01:
        raise ValueError(
            f"{r}: smoothing position spacing={pitch:g}; expected 1 after export"
        )
    df["repli"] = r
    return df


def target_from_meta(df_meta: pd.DataFrame) -> int:
    if ANNOTATION_KEY not in df_meta.columns:
        raise KeyError(
            f"SpiceMix meta table does not contain {ANNOTATION_KEY!r}; "
            "cannot derive target cluster count."
        )
    labels = df_meta[ANNOTATION_KEY]
    labels = labels[labels.notna()].astype(str).str.strip()
    labels = labels[labels != ""]
    n = int(labels.nunique())
    if n <= 0:
        raise ValueError(f"No observed labels in meta[{ANNOTATION_KEY!r}].")
    return n


def validate_spicemix_expression(obj: SpiceMix, spec: SampleSpec):
    "SpiceMix expects finite, nonnegative normalized expression."
    for j, Y in enumerate(obj.Ys):
        if not torch.isfinite(Y).all().item():
            raise ValueError(
                f"{spec.key}: SpiceMix expression replicate {j} has non-finite values"
            )
        if torch.any(Y < 0).item():
            raise ValueError(
                f"{spec.key}: SpiceMix expression replicate {j} contains negative values; "
                "SpiceMix expects nonnegative normalized expression."
            )


def cache_paths(spec: SampleSpec):
    K = spice_k(spec)
    fixed_dir = method_output_dir(spec) / "shared_graph"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{spec.key}_spicemix_fixed_"
        f"K{K}_lambda{SPICEMIX_LAMBDA_SIGMA_X_INV:g}_"
        f"initpc{SPICEMIX_INIT_NUM_PCS}_initk{SPICEMIX_INIT_N_NEIGHBORS}_"
        f"initrs{SPICEMIX_INIT_NUM_RS}_nmf{SPICEMIX_INIT_NMF_ITER}_"
        f"iter{SPICEMIX_MAX_ITER}_trainseed{SPICEMIX_TRAINING_SEED}_"
        f"clusterk{SPICEMIX_CLUSTER_N_NEIGHBORS}_"
        f"graphseed{SPICEMIX_CLUSTER_GRAPH_SEED}_{COORDINATE_SCALING_TAG}"
    )

    return fixed_dir / f"{stem}.npz", fixed_dir / f"{stem}_coords.npy"


def set_training_seed(seed: int):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def make_downstream_graph(latent_state: np.ndarray, df_meta: pd.DataFrame):
    "Reproduce SpiceMix's official latent-state clustering preprocessing ONCE."
    if latent_state.ndim != 2:
        raise ValueError(f"latent_state must be 2-D, got {latent_state.shape}")
    if latent_state.shape[0] != len(df_meta):
        raise ValueError(
            f"latent/meta row mismatch: {latent_state.shape[0]} vs {len(df_meta)}"
        )
    coords = df_meta[["coor X", "coor Y"]].to_numpy(dtype=float, copy=True)
    rep = df_meta["repli"].to_numpy(copy=True)

    z = StandardScaler().fit_transform(latent_state)
    z = smooth_continuous(
        coords,
        z,
        rep,
        radius=SPICEMIX_CONTINUOUS_SMOOTH_RADIUS,
        radius_eps=SPICEMIX_CONTINUOUS_SMOOTH_RADIUS_EPS,
        lam=SPICEMIX_CONTINUOUS_SMOOTH_LAMBDA,
    )

    cluster_adata = ad.AnnData(X=np.asarray(z, dtype=np.float32))
    sc.pp.neighbors(
        cluster_adata,
        use_rep="X",
        n_neighbors=SPICEMIX_CLUSTER_N_NEIGHBORS,
        metric=SPICEMIX_CLUSTER_METRIC,
        method=SPICEMIX_CLUSTER_METHOD,
        knn=SPICEMIX_CLUSTER_KNN,
        random_state=SPICEMIX_CLUSTER_GRAPH_SEED,
    )
    graph = cluster_adata.obsp["connectivities"].tocsr(copy=True)
    graph.sort_indices()
    return graph, coords


def train_and_build_one(spec: SampleSpec, device: str, cache_only=False):
    sample_started = time.perf_counter()
    input_dir = sample_input_dir(spec)
    if not (input_dir / "files").exists():
        raise FileNotFoundError(f"Missing SpiceMix input folder: {input_dir / 'files'}")
    with timed("sample_metadata_and_cache_check", spec.key):
        df_meta = load_meta_only(spec)
        target = target_from_meta(df_meta)
        n_obs = int(len(df_meta))
        K = spice_k(spec)
        graph_path, coords_path = cache_paths(spec)
        files_dir = input_dir / "files"
        r = repli_name(spec)
        for old_pkl in (files_dir / f"expression_{r}.pkl",):
            if old_pkl.exists():
                raise FileExistsError(
                    f"Stale .pkl could bypass exported text: {old_pkl}"
                )
        input_paths = [
            files_dir / f"{prefix}_{r}{suffix}"
            for prefix, suffix in (
                ("expression", ".txt"),
                ("neighborhood", ".txt"),
                ("genes", ".txt"),
                ("meta", ".csv"),
                ("spicemix_coordinate_qc", ".json"),
            )
        ]
        for path in input_paths:
            if not path.exists():
                raise FileNotFoundError(path)
    if graph_path.exists() and coords_path.exists() and not FORCE_RETRAIN:
        graph = sparse.load_npz(graph_path).tocsr()
        coords = np.load(coords_path, mmap_mode="r")
        if graph.shape != (n_obs, n_obs):
            raise ValueError(
                f"{spec.key}: cached graph shape {graph.shape} != {(n_obs, n_obs)}; "
                "set FORCE_RETRAIN=True"
            )
        if coords.shape != (n_obs, 2):
            raise ValueError(
                f"{spec.key}: cached coords shape {coords.shape} != {(n_obs, 2)}; "
                "set FORCE_RETRAIN=True"
            )
        if any(
            path.stat().st_mtime_ns > graph_path.stat().st_mtime_ns
            for path in input_paths
        ) or not np.allclose(
            coords, df_meta[["coor X", "coor Y"]].to_numpy(), rtol=0, atol=1e-9
        ):
            print(f"[{spec.key}] input or smoothing coordinates changed; retraining")
            del graph, coords
        else:
            print(
                f"[{spec.key}] reuse fixed SpiceMix graph: n={n_obs:,}, target={target}, "
                f"K={K}, downstream_k={SPICEMIX_CLUSTER_N_NEIGHBORS}"
            )
            del graph, coords
            log_timing(
                "sample_total_reused", sample_started, sample=spec.key, device="cpu"
            )
            return {
                "key": spec.key,
                "graph_path": str(graph_path),
                "coords_path": str(coords_path),
                "n_obs": n_obs,
                "target_clusters": target,
                "K": K,
            }
    if cache_only:
        return None
    print(
        f"[{spec.key}] train SpiceMix ONCE on {device}; "
        f"n={n_obs:,}, target={target}, K={K}"
    )
    if device != "cpu":
        torch.cuda.set_device(device)
    set_training_seed(SPICEMIX_TRAINING_SEED)
    context = dict(device=device, dtype=SPICEMIX_DTYPE)

    with timed("model_load_dataset", spec.key, device=device):
        obj = SpiceMix(
            K=K,
            lambda_Sigma_x_inv=SPICEMIX_LAMBDA_SIGMA_X_INV,
            power_Sigma_x_inv=SPICEMIX_POWER_SIGMA_X_INV,
            repli_list=[repli_name(spec)],
            context=context,
            context_Y=context,
            path2result=None,
        )
        obj.load_dataset(input_dir)
        validate_spicemix_expression(obj, spec)

    target_loaded = target_from_meta(obj.meta)
    if target_loaded != target:
        raise RuntimeError(
            f"{spec.key}: meta target changed between reads: {target} -> {target_loaded}"
        )
    avg_model_neighbors = sum(len(x) for x in obj.Es[0]) / float(obj.Ns[0])
    print(
        f"[{spec.key}] prepared SpiceMix spatial graph: "
        f"average adjacency-list neighbors={avg_model_neighbors:.3f}"
    )
    init_kwargs = dict(
        num_pcs=SPICEMIX_INIT_NUM_PCS,
        n_neighbors=SPICEMIX_INIT_N_NEIGHBORS,
        resolution_boundaries=SPICEMIX_INIT_RESOLUTION_BOUNDARIES,
        num_rs=SPICEMIX_INIT_NUM_RS,
    )
    with timed("louvain_initialization", spec.key, device=device):
        obj.initialize(
            method=SPICEMIX_INIT_METHOD,
            random_state=SPICEMIX_TRAINING_SEED,
            kwargs=init_kwargs,
        )

    with timed("nonspatial_initialization", spec.key, device=device):
        for iiter in range(SPICEMIX_INIT_NMF_ITER):
            checkpoint_index = 0 if iiter == SPICEMIX_INIT_NMF_ITER - 1 else -1
            obj.estimate_weights(
                iiter=checkpoint_index, use_spatial=[False] * obj.num_repli
            )
            obj.estimate_parameters(
                iiter=checkpoint_index, use_spatial=[False] * obj.num_repli
            )

    with timed("initialize_spatial_affinity", spec.key, device=device):
        obj.initialize_Sigma_x_inv()

    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    with timed("spatial_training", spec.key, device=device):
        for iiter in range(1, SPICEMIX_MAX_ITER + 1):
            logger.info(
                f"[{spec.key}] SpiceMix spatial iteration {iiter}/{SPICEMIX_MAX_ITER}"
            )
            obj.estimate_parameters(iiter=iiter, use_spatial=[True] * obj.num_repli)
            obj.estimate_weights(iiter=iiter, use_spatial=[True] * obj.num_repli)
    latent_state = np.concatenate([X.detach().cpu().numpy() for X in obj.Xs], axis=0)
    if latent_state.shape != (n_obs, K):
        raise RuntimeError(
            f"{spec.key}: unexpected SpiceMix latent shape {latent_state.shape}; "
            f"expected {(n_obs, K)}"
        )
    with timed("continuous_smoothing_and_leiden_graph", spec.key):
        graph, coords = make_downstream_graph(latent_state, obj.meta)
    with timed("save_fixed_graph", spec.key):
        np.save(coords_path, coords, allow_pickle=False)
        sparse.save_npz(graph_path, graph, compressed=True)
    print(
        f"[{spec.key}] fixed SpiceMix downstream graph saved: "
        f"nodes={graph.shape[0]:,}, nnz={graph.nnz:,}, K={K}, "
        f"cluster_k={SPICEMIX_CLUSTER_N_NEIGHBORS}"
    )
    del graph, coords, latent_state, obj, df_meta
    gc.collect()
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_timing("sample_total", sample_started, sample=spec.key, device=device)
    return {
        "key": spec.key,
        "graph_path": str(graph_path),
        "coords_path": str(coords_path),
        "n_obs": n_obs,
        "target_clusters": target,
        "K": K,
    }


def check_cpu_worker_imports():
    "Fail before GPU training if fresh workers cannot import the CPU stack."
    import scipy.linalg
    from functions import search_res as worker_search_res
    from spicemix_external import SpiceMix as WorkerSpiceMix

    return scipy.__version__, worker_search_res.__name__, WorkerSpiceMix.__name__


def tune_one_sample(spec: SampleSpec, info: dict):
    if spec.key in RESOLUTION_OVERRIDES:
        res = float(RESOLUTION_OVERRIDES[spec.key])
        print(f"[{spec.key}] manual SpiceMix-Louvain resolution={res:.4f}")
        log_timing(
            "resolution_override", time.perf_counter(), sample=spec.key, resolution=res
        )
        return spec.key, res
    with timed("resolution_search", spec.key):
        graph = sparse.load_npz(info["graph_path"]).tocsr()
        target = int(info["target_clusters"])
        settings = dict(
            method="louvain",
            random_state=RESOLUTION_SEED,
            directed=SPICEMIX_LOUVAIN_DIRECTED,
            use_weights=SPICEMIX_LOUVAIN_USE_WEIGHTS,
            louvain_flavor=SPICEMIX_LOUVAIN_FLAVOR,
        )
        probe_count = _cluster_count_from_adjacency(
            adjacency=graph,
            resolution=SPICEMIX_RES_START,
            **settings,
        )
        use_low_range = probe_count > target
        start = SPICEMIX_LOW_RES_START if use_low_range else SPICEMIX_RES_START
        end = SPICEMIX_RES_START if use_low_range else SPICEMIX_RES_END
        increment = (
            SPICEMIX_LOW_RES_INCREMENT if use_low_range else SPICEMIX_RES_INCREMENT
        )
        print(
            f"[{spec.key}] resolution={SPICEMIX_RES_START}: {probe_count} clusters; "
            f"target={target}; search=[{start}, {end}], fine increment={increment}",
            flush=True,
        )
        res = search_res(
            adjacency=graph,
            n_clusters=target,
            start=start,
            end=end,
            increment=increment,
            tolerance=SPICEMIX_RES_TOLERANCE,
            verbose=False,
            **settings,
        )

        selection = dict(
            sample=spec.key,
            target_clusters=target,
            probe_resolution=SPICEMIX_RES_START,
            probe_clusters=probe_count,
            search_start=start,
            search_end=end,
            fine_increment=increment,
            tolerance=SPICEMIX_RES_TOLERANCE,
            selected_resolution=float(res),
            graph_path=info["graph_path"],
            **settings,
        )
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        (LOG_ROOT / f"{spec.key}_resolution.json").write_text(
            json.dumps(selection, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"[{spec.key}] fixed SpiceMix representation/graph: target={target}; "
        f"Louvain resolution={res:.4f}"
    )
    return spec.key, float(res)


_GRAPH_CACHE = {}
_COORD_CACHE = {}


def load_graph_cached(path: str):
    graph = _GRAPH_CACHE.get(path)
    if graph is None:
        _GRAPH_CACHE.clear()  # retain at most one sample's graph per CPU worker
        graph = sparse.load_npz(path).tocsr()
        _GRAPH_CACHE[path] = graph
    return graph


def load_coords_cached(path: str):
    coords = _COORD_CACHE.get(path)
    if coords is None:
        _COORD_CACHE.clear()
        coords = np.load(path, mmap_mode="r")
        _COORD_CACHE[path] = coords
    return coords


def smooth_labels(coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    "Official SpiceMix iterative discrete spatial refinement."
    labels = np.asarray(labels)
    rep = np.zeros(labels.shape[0], dtype=np.int8)  # one FOV per benchmark sample
    while True:
        prev = labels
        labels = smooth_discrete(
            coords,
            labels,
            rep,
            radius=SPICEMIX_DISCRETE_SMOOTH_RADIUS,
            radius_eps=SPICEMIX_DISCRETE_SMOOTH_RADIUS_EPS,
        )
        if np.array_equal(prev, labels):
            break
    return np.asarray(labels, dtype=int)


def cluster_one(spec: SampleSpec, info: dict, resolution: float, seed: int):
    "One final run: ONLY the Louvain random_state differs."
    run_started = time.perf_counter()
    graph = load_graph_cached(info["graph_path"])
    coords = load_coords_cached(info["coords_path"])
    n_obs = int(info["n_obs"])
    tmp = ad.AnnData(
        X=sparse.csr_matrix((n_obs, 0), dtype=np.float32),
        obs=pd.DataFrame(index=pd.RangeIndex(n_obs).astype(str)),
    )
    with timed("louvain_run", spec.key, seed):
        sc.tl.louvain(
            tmp,
            adjacency=graph,
            resolution=float(resolution),
            random_state=int(seed),
            key_added="cluster",
            directed=SPICEMIX_LOUVAIN_DIRECTED,
            use_weights=SPICEMIX_LOUVAIN_USE_WEIGHTS,
            flavor=SPICEMIX_LOUVAIN_FLAVOR,
        )
        raw_labels = tmp.obs["cluster"].astype(int).to_numpy()
    with timed("discrete_smoothing", spec.key, seed):
        labels = smooth_labels(coords, raw_labels)
    with timed("save_labels", spec.key, seed):
        method_dir = method_output_dir(spec)
        temp_dir = method_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        out_path = temp_dir / f"{spec.key}-{METHOD_NAME}-seed{seed}.pkl"
        joblib.dump(labels, out_path, compress=0)
    log_timing(
        "cluster_run_total",
        run_started,
        sample=spec.key,
        seed=seed,
        raw_clusters=int(np.unique(raw_labels).size),
        smoothed_clusters=int(np.unique(labels).size),
    )
    return (
        spec.key,
        int(seed),
        str(out_path),
        int(np.unique(raw_labels).size),
        int(np.unique(labels).size),
    )


def consolidated_result_path(spec: SampleSpec) -> Path:
    section = f"{spec.sample}_" if spec.dataset == "dlpfc" else ""
    return (
        method_output_dir(spec)
        / f"{METHOD_NAME}_results_{section}{ITERATIONS}iterations.pkl"
    )


def completed_result_present(spec: SampleSpec) -> bool:
    "Skip verified results; refuse to overwrite an inconsistent final file."
    path = consolidated_result_path(spec)
    if not path.exists():
        return False
    results = joblib.load(path, mmap_mode="r")
    n_obs = len(load_meta_only(spec))
    seeds = range(CLUSTER_BASE_SEED, CLUSTER_BASE_SEED + ITERATIONS)
    seed_paths = [
        method_output_dir(spec) / "temp" / f"{spec.key}-{METHOD_NAME}-seed{seed}.pkl"
        for seed in seeds
    ]
    if (
        len(results) != ITERATIONS
        or any(
            np.asarray(labels).shape != (n_obs,)
            or not np.isfinite(np.asarray(labels)).all()
            for labels in results
        )
        or any(not p.is_file() or p.stat().st_size == 0 for p in seed_paths)
    ):
        raise RuntimeError(
            f"{spec.key}: existing consolidated output or seed files are incomplete; "
            "inspect them before rerunning. Existing results were not overwritten."
        )
    return True


def consolidate(spec: SampleSpec, records):
    started = time.perf_counter()
    recs = sorted(
        [r for r in records if r[0] == spec.key],
        key=lambda r: r[1],
    )
    if len(recs) != ITERATIONS:
        raise RuntimeError(
            f"{spec.key}: expected {ITERATIONS} SpiceMix runs, got {len(recs)}"
        )
    results = [joblib.load(r[2]) for r in recs]
    method_dir = method_output_dir(spec)
    method_dir.mkdir(parents=True, exist_ok=True)
    output_path = consolidated_result_path(spec)
    name = output_path.name
    joblib.dump(results, output_path, compress=0)
    raw_counts = [r[3] for r in recs]
    smoothed_counts = [r[4] for r in recs]
    print(
        f"[{spec.key}] saved {name}; clusters before smoothing "
        f"range={min(raw_counts)}..{max(raw_counts)}, after smoothing "
        f"range={min(smoothed_counts)}..{max(smoothed_counts)}"
    )
    log_timing("sample_consolidation", started, sample=spec.key)


def downstream_one_sample(spec: SampleSpec, info: dict):
    "Finish one sample independently as soon as its fixed graph is ready."
    started = time.perf_counter()
    status = "error"
    try:
        key, resolution = tune_one_sample(spec, info)
        if key != spec.key:
            raise RuntimeError(f"Unexpected resolution result for {spec.key}: {key}")
        records = [
            cluster_one(spec, info, resolution, CLUSTER_BASE_SEED + i)
            for i in range(ITERATIONS)
        ]
        consolidate(spec, records)
        status = "ok"
        return spec.key
    finally:
        log_timing("sample_downstream_total", started, sample=spec.key, status=status)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--samples",
        nargs="+",
        choices=[s.key for s in SAMPLES],
        help="Process only these samples.",
    )
    selection.add_argument(
        "--all-samples",
        action="store_true",
        help="Select all 18 samples (the Python runner's default).",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Recompute downstream clustering for every selected sample, "
        "replacing existing seed labels and consolidated results. "
        "Compatible trained graphs are still reused.",
    )
    parser.add_argument(
        "--require-cached-graphs",
        action="store_true",
        help="Fail if a pending sample needs training or graph rebuilding.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate output/cache status and write the execution plan only.",
    )
    args = parser.parse_args(argv)
    if args.samples and len(args.samples) != len(set(args.samples)):
        parser.error("--samples must not contain duplicates")
    return args


def main(argv=None):
    args = parse_args(argv)
    selected = [s for s in SAMPLES if args.samples is None or s.key in args.samples]
    skipped = (
        []
        if args.rerun_completed
        else [s for s in selected if completed_result_present(s)]
    )
    skipped_keys = {s.key for s in skipped}
    pending = [s for s in selected if s.key not in skipped_keys]
    for spec in skipped:
        print(
            f"[{spec.key}] all {ITERATIONS} results already present; skipping",
            flush=True,
        )
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    hardware = hardware_info()
    hardware.update(
        samples=len(selected),
        pending_samples=len(pending),
        skipped_samples=len(skipped),
        rerun_completed=args.rerun_completed,
        plan_only=args.plan_only,
        training_gpus=N_GPUS,
        cpu_sample_pipeline_jobs=CPU_PIPELINE_JOBS,
        cpu_training_jobs=CPU_TRAIN_JOBS,
    )
    (LOG_ROOT / "hardware.json").write_text(
        json.dumps(hardware, indent=2) + "\n", encoding="utf-8"
    )
    print("Hardware and timing logs:", LOG_ROOT, flush=True)
    print("Hardware:", json.dumps(hardware), flush=True)
    print("SpiceMix fixed-representation seed experiment")
    print(
        f"  selected samples={len(selected)}; scheduled={len(pending)}; skipped complete={len(skipped)}"
    )
    print(f"  rerun completed samples={args.rerun_completed}")
    print(f"  concurrent CPU sample pipelines={CPU_PIPELINE_JOBS}")
    print(
        f"  initialization resolution boundaries={SPICEMIX_INIT_RESOLUTION_BOUNDARIES}"
    )
    print(f"  downstream resolution end={SPICEMIX_RES_END}")
    print(f"  model seed={SPICEMIX_TRAINING_SEED} (fixed)")
    print(f"  K default={SPICEMIX_DEFAULT_K}; DLPFC K={SPICEMIX_DLPFC_K}")
    print(
        f"  initialization={SPICEMIX_INIT_METHOD}, PCA={SPICEMIX_INIT_NUM_PCS}, "
        f"KNN={SPICEMIX_INIT_N_NEIGHBORS}, num_rs={SPICEMIX_INIT_NUM_RS}"
    )
    print(
        f"  model iterations: non-spatial={SPICEMIX_INIT_NMF_ITER}, "
        f"spatial={SPICEMIX_MAX_ITER}"
    )
    print(
        f"  downstream graph: k={SPICEMIX_CLUSTER_N_NEIGHBORS}, "
        f"metric={SPICEMIX_CLUSTER_METRIC}, seed={SPICEMIX_CLUSTER_GRAPH_SEED}"
    )
    print(
        "  Louvain interpretation: "
        f"weights={SPICEMIX_LOUVAIN_USE_WEIGHTS}, "
        f"directed={SPICEMIX_LOUVAIN_DIRECTED}, "
        f"flavor={SPICEMIX_LOUVAIN_FLAVOR}"
    )
    print(
        f"  final clustering seeds={CLUSTER_BASE_SEED}.."
        f"{CLUSTER_BASE_SEED + ITERATIONS - 1}"
    )

    with timed("cached_graph_preflight"):
        cached = []
        missing = []
        for spec in pending:
            info = train_and_build_one(spec, "cpu", cache_only=True)
            if info is None:
                missing.append(spec)
            else:
                cached.append((spec, info))
    plan = dict(
        selected=[s.key for s in selected],
        skipped_complete=[s.key for s in skipped],
        excluded=[s.key for s in SAMPLES if s not in selected],
        cached=[info for _, info in cached],
        needs_training=[s.key for s in missing],
        require_cached_graphs=args.require_cached_graphs,
        plan_only=args.plan_only,
        rerun_completed=args.rerun_completed,
        seed_start=CLUSTER_BASE_SEED,
        seed_end=CLUSTER_BASE_SEED + ITERATIONS - 1,
        pending_seed_runs=len(pending) * ITERATIONS,
    )
    (LOG_ROOT / "execution_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.require_cached_graphs and missing:
        raise RuntimeError(
            "Recovery requires existing compatible graphs; these samples need "
            f"training/rebuilding: {[s.key for s in missing]}. No training was started."
        )
    print(
        f"Reusing {len(cached)} fixed graphs; training {len(missing)} samples.",
        flush=True,
    )
    if args.plan_only:
        print("Plan validated; no clustering or training requested.", flush=True)
        return
    if not pending:
        print("All selected samples are already complete.", flush=True)
        return

    if N_GPUS:
        with timed("gpu_architecture_preflight"):
            supported = torch.cuda.get_arch_list()
            for i in range(N_GPUS):
                major, minor = torch.cuda.get_device_capability(i)
                if (
                    f"sm_{major}{minor}" not in supported
                    and f"compute_{major}{minor}" not in supported
                ):
                    raise RuntimeError(
                        f"GPU {i} ({torch.cuda.get_device_name(i)}) requires "
                        f"sm_{major}{minor}, but PyTorch supports {supported}. "
                        "Use a PyTorch build for this GPU, or set "
                        "SPICEMIX_NUM_GPUS=0 to run training on CPU."
                    )

    with timed("worker_import_preflight"):
        joblib.Parallel(n_jobs=2, backend="loky")(
            joblib.delayed(check_cpu_worker_imports)() for _ in range(2)
        )

    with timed("streaming_pipeline_total", device="cpu+gpu" if N_GPUS else "cpu"):
        context = multiprocessing.get_context("spawn")
        worker_count = min(N_GPUS if N_GPUS else CPU_TRAIN_JOBS, len(missing))
        failures = []
        complete = []
        with ExitStack() as stack:
            cpu_pool = stack.enter_context(
                ProcessPoolExecutor(max_workers=CPU_PIPELINE_JOBS, mp_context=context)
            )
            train_pools = [
                stack.enter_context(
                    ProcessPoolExecutor(max_workers=1, mp_context=context)
                )
                for _ in range(worker_count)
            ]
            cpu_futures = {
                cpu_pool.submit(downstream_one_sample, spec, info): spec
                for spec, info in cached
            }
            train_futures = {}
            missing_iter = iter(missing)

            def submit_training(worker_index):
                spec = next(missing_iter, None)
                if spec is not None:
                    device = f"cuda:{worker_index}" if N_GPUS else "cpu"
                    future = train_pools[worker_index].submit(
                        train_and_build_one, spec, device
                    )
                    train_futures[future] = (spec, worker_index)
                    print(f"[{spec.key}] submitted training on {device}", flush=True)

            for worker_index in range(worker_count):
                submit_training(worker_index)

            while train_futures or cpu_futures:
                done, _ = wait(
                    set(train_futures) | set(cpu_futures), return_when=FIRST_COMPLETED
                )
                for future in done:
                    if future in train_futures:
                        spec, worker_index = train_futures.pop(future)
                        try:
                            info = future.result()
                            if info["key"] != spec.key:
                                raise RuntimeError(f"Wrong trained sample: {spec.key}")
                            cpu_futures[
                                cpu_pool.submit(downstream_one_sample, spec, info)
                            ] = spec
                        except Exception as exc:
                            failures.append((spec.key, "training", str(exc)))
                            traceback.print_exception(type(exc), exc, exc.__traceback__)
                        submit_training(worker_index)
                    else:
                        spec = cpu_futures.pop(future)
                        try:
                            result_key = future.result()
                            if result_key != spec.key:
                                raise RuntimeError(
                                    f"Wrong clustered sample: {spec.key}"
                                )
                            complete.append(spec.key)
                            print(
                                f"[{spec.key}] 50 seed runs complete "
                                f"({len(complete)}/{len(pending)} pending samples)",
                                flush=True,
                            )
                        except Exception as exc:
                            failures.append((spec.key, "downstream", str(exc)))
                            traceback.print_exception(type(exc), exc, exc.__traceback__)
        if failures:
            raise RuntimeError(f"SpiceMix sample failures: {failures}")
        if len(complete) != len(pending):
            raise RuntimeError(
                f"Only {len(complete)}/{len(pending)} pending samples completed"
            )
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        log_timing(
            "total_wall",
            PROGRAM_START,
            device="cpu+gpu" if N_GPUS else "cpu",
            status="error",
        )
        raise
    else:
        log_timing("total_wall", PROGRAM_START, device="cpu+gpu" if N_GPUS else "cpu")
    finally:
        write_timing_summary()
