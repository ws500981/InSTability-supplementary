#!/usr/bin/env python3
"Repeat Leiden and mclust clustering on fixed GraphST representations."

from paths import DATA_ROOT as INPUT_DATA_ROOT, OUTPUT_ROOT
import os
import sys
import time
from pathlib import Path

PROGRAM_STARTED = time.perf_counter()

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
import hashlib
import json
import importlib.metadata
import inspect
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
from GraphST import GraphST
from functions import get_target_n_clusters, preprocessing

DATA_ROOT = INPUT_DATA_ROOT

OUT_ROOT = OUTPUT_ROOT

METHOD_NAME = "graphst"

RUN_ID = os.environ.setdefault(
    "GRAPHST_RUN_ID",
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}",
)
LOG_ROOT = OUT_ROOT / "run_logs" / METHOD_NAME / RUN_ID


def hardware_info():
    "Record resources visible to this job, following the other methods."
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
        "available_cpus": (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
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
    "A separate JSON per event is safe when multiple loky workers finish together."
    gpu_device = str(device).startswith("cuda")
    if gpu_device and status == "ok":
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
    if gpu_device and status == "ok":
        device_obj = torch.device(device)
        record.update(
            gpu_name=torch.cuda.get_device_name(device_obj),
            peak_gpu_allocated_gib=round(
                torch.cuda.max_memory_allocated(device_obj) / 2**30, 3
            ),
            peak_gpu_reserved_gib=round(
                torch.cuda.max_memory_reserved(device_obj) / 2**30, 3
            ),
        )
    events = LOG_ROOT / "timing_events"
    events.mkdir(parents=True, exist_ok=True)
    (events / f"{uuid.uuid4().hex}.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        f"TIME {stage} sample={sample or 'all'} seed={seed} "
        f"seconds={record['elapsed_seconds']:.3f} device={device} "
        f"status={status}",
        flush=True,
    )


@contextmanager
def timed(stage, sample=None, seed=None, device="cpu"):
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        log_timing(
            stage, started, sample=sample, seed=seed, device=device, status="error"
        )
        raise
    else:
        log_timing(stage, started, sample=sample, seed=seed, device=device)


def write_timing_summary():
    event_dir = LOG_ROOT / "timing_events"
    records = (
        [json.loads(p.read_text(encoding="utf-8")) for p in event_dir.glob("*.json")]
        if event_dir.exists()
        else []
    )
    if records:
        pd.DataFrame(records).sort_values("finished_utc").to_csv(
            LOG_ROOT / "timings.csv", index=False
        )
        print(f"Timing summary: {LOG_ROOT / 'timings.csv'}", flush=True)


ANNOTATION_KEY = "annotation"

ITERATIONS = 50

CLUSTER_BASE_SEED = 42  # same 42..91 seed series as Leiden/Louvain

GRAPHST_TRAINING_SEED = 41  # GraphST released default

RESOLUTION_SEED = 0  # GraphST's released resolution-search seed

MCLUST_MODEL = "EEE"

FORCE_RETRAIN = False

GRAPHST_N_HVG = 3000

GRAPHST_HVG_FLAVOR = "seurat_v3"

GRAPHST_NORMALIZE_TARGET_SUM = 1e4

GRAPHST_SCALE = True

GRAPHST_ZERO_CENTER = False

GRAPHST_SCALE_MAX_VALUE = 10

GRAPHST_SPATIAL_N_NEIGHBORS = 3

GRAPHST_LEARNING_RATE = 0.001

GRAPHST_WEIGHT_DECAY = 0.0

GRAPHST_EPOCHS = 600

GRAPHST_DIM_OUTPUT = 64

GRAPHST_ALPHA = 10

GRAPHST_BETA = 1

GRAPHST_DECONVOLUTION = False

GRAPHST_CLUSTER_PCS = 20

GRAPHST_CLUSTER_PCA_SEED = 42

GRAPHST_CLUSTER_N_NEIGHBORS = 50

GRAPHST_CLUSTER_GRAPH_SEED = 0  # Scanpy neighbors default in GraphST's env

GRAPHST_CLUSTER_METRIC = "euclidean"

GRAPHST_CLUSTER_CONNECTIVITY_METHOD = "umap"

GRAPHST_CLUSTER_USE_WEIGHTS = True

GRAPHST_CLUSTER_DIRECTED = True

GRAPHST_CLUSTER_N_ITERATIONS = -1

GRAPHST_CLUSTER_FLAVOR = "leidenalg"

GRAPHST_RES_START = 0.1

GRAPHST_RES_END = 3.0

GRAPHST_RES_INCREMENT = 0.01

GRAPHST_REFINEMENT = False

RESOLUTION_OVERRIDES = {}

ALLOCATED_CPUS = max(
    1, int(os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 1)))
)

RESOLUTION_JOBS = min(18, ALLOCATED_CPUS)

CLUSTER_JOBS = min(8, ALLOCATED_CPUS)

MCLUST_JOBS = min(
    ALLOCATED_CPUS,
    max(1, int(os.environ.get("GRAPHST_MCLUST_JOBS", "4"))),
)

VISIBLE_GPUS = torch.cuda.device_count()

REQUESTED_GPUS = int(os.environ.get("GRAPHST_NUM_GPUS", str(VISIBLE_GPUS)))

N_GPUS = min(VISIBLE_GPUS, max(0, REQUESTED_GPUS))


@dataclass(frozen=True)
class SampleSpec:
    key: str

    h5ad_name: str

    output_group: str

    datatype: str = "10X"


SAMPLES = [
    SampleSpec("mouse_brain", "mouse_brain_filtered.h5ad", "mouse_brain", "10X"),
    SampleSpec(
        "mouse_brain_cerebellum",
        "mouse_brain_cerebellum_filtered.h5ad",
        "mouse_brain_cerebellum",
        "Slide",
    ),
    SampleSpec(
        "human_breast_cancer",
        "human_breast_cancer_filtered.h5ad",
        "human_breast_cancer",
        "10X",
    ),
    SampleSpec(
        "Visium_HD_Human_Colon_Cancer_cropped_square",
        "Visium_HD_Human_Colon_Cancer_cropped_square_filtered.h5ad",
        "Visium_HD_Human_Colon_Cancer_cropped_square",
        "10X",
    ),
    SampleSpec("ov_ffpe", "ov_ffpe_filtered.h5ad", "ov_ffpe", "10X"),
    SampleSpec("coad_ffpe", "coad_ffpe_filtered.h5ad", "coad_ffpe", "10X"),
    SampleSpec("dlpfc_151507", "dlpfc_151507_filtered.h5ad", "dlpfc", "10X"),
    SampleSpec("dlpfc_151508", "dlpfc_151508_filtered.h5ad", "dlpfc", "10X"),
    SampleSpec("dlpfc_151509", "dlpfc_151509_filtered.h5ad", "dlpfc", "10X"),
    SampleSpec("dlpfc_151510", "dlpfc_151510_filtered.h5ad", "dlpfc", "10X"),
    SampleSpec("dlpfc_151669", "dlpfc_151669_filtered.h5ad", "dlpfc", "10X"),
    SampleSpec("dlpfc_151670", "dlpfc_151670_filtered.h5ad", "dlpfc", "10X"),
    SampleSpec("dlpfc_151671", "dlpfc_151671_filtered.h5ad", "dlpfc", "10X"),
    SampleSpec("dlpfc_151672", "dlpfc_151672_filtered.h5ad", "dlpfc", "10X"),
    SampleSpec("dlpfc_151673", "dlpfc_151673_filtered.h5ad", "dlpfc", "10X"),
    SampleSpec("dlpfc_151674", "dlpfc_151674_filtered.h5ad", "dlpfc", "10X"),
    SampleSpec("dlpfc_151675", "dlpfc_151675_filtered.h5ad", "dlpfc", "10X"),
    SampleSpec("dlpfc_151676", "dlpfc_151676_filtered.h5ad", "dlpfc", "10X"),
]


def representation_tag(spec: SampleSpec) -> str:
    "Parameterized tag for the frozen GraphST clustering representation."

    datatype = spec.datatype.lower().replace("-", "")

    return (
        f"{spec.key}_graphst_fixed_"
        f"dtype{datatype}_trainseed{GRAPHST_TRAINING_SEED}_"
        f"hvg{GRAPHST_N_HVG}_out{GRAPHST_DIM_OUTPUT}_"
        f"pca{GRAPHST_CLUSTER_PCS}_pcaseed{GRAPHST_CLUSTER_PCA_SEED}"
    )


def embedding_filename(spec: SampleSpec) -> str:
    return f"{representation_tag(spec)}_embedding.npy"


def graph_filename(spec: SampleSpec) -> str:
    "Leiden graph built from the exact saved shared representation."

    return (
        f"{representation_tag(spec)}_"
        f"k{GRAPHST_CLUSTER_N_NEIGHBORS}_{GRAPHST_CLUSTER_METRIC}_"
        f"graphseed{GRAPHST_CLUSTER_GRAPH_SEED}.npz"
    )


def target_and_size(input_path: Path):
    "Read only what is needed to derive target k and n_obs."

    backed = sc.read_h5ad(input_path, backed="r")

    try:
        target = get_target_n_clusters(backed, ANNOTATION_KEY)

        n_obs = int(backed.n_obs)

    finally:
        if backed.file is not None:
            backed.file.close()

    if target < 2:
        raise ValueError(f"{input_path}: expected at least two annotated classes")
    return target, n_obs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "source-install"


def leiden_backend_kwargs():
    "Scanpy 1.9.1 has only the leidenalg backend, without a flavor argument."
    if "flavor" in inspect.signature(sc.tl.leiden).parameters:
        return {"flavor": GRAPHST_CLUSTER_FLAVOR}
    if GRAPHST_CLUSTER_FLAVOR != "leidenalg":
        raise ValueError("Installed Scanpy only supports leidenalg")
    return {}


def search_graphst_resolution(graph, target):
    "GraphST's descending resolution search, on the fixed saved graph."
    tmp = empty_adata(graph.shape[0])
    for res in sorted(
        np.arange(GRAPHST_RES_START, GRAPHST_RES_END, GRAPHST_RES_INCREMENT),
        reverse=True,
    ):
        sc.tl.leiden(
            tmp,
            adjacency=graph,
            resolution=float(res),
            random_state=RESOLUTION_SEED,
            key_added="domain",
            directed=GRAPHST_CLUSTER_DIRECTED,
            use_weights=GRAPHST_CLUSTER_USE_WEIGHTS,
            n_iterations=GRAPHST_CLUSTER_N_ITERATIONS,
            **leiden_backend_kwargs(),
        )
        if tmp.obs["domain"].nunique() == target:
            return float(res)
    raise ValueError(
        f"No Leiden resolution from {GRAPHST_RES_START} to "
        f"{GRAPHST_RES_END} produced {target} clusters"
    )


def make_cluster_adata(embedding_pca: np.ndarray) -> ad.AnnData:
    n_obs = embedding_pca.shape[0]

    tmp = ad.AnnData(
        X=sparse.csr_matrix((n_obs, 0), dtype=np.float32),
        obs=pd.DataFrame(index=pd.RangeIndex(n_obs).astype(str)),
    )

    tmp.obsm["emb_pca"] = np.array(embedding_pca, copy=True, order="C")

    return tmp


def build_leiden_graph_from_embedding(embedding_pca: np.ndarray, graph_path: Path):
    "Build the one fixed GraphST->Leiden graph from the shared PCA-20 input."

    cluster_adata = make_cluster_adata(embedding_pca)

    sc.pp.neighbors(
        cluster_adata,
        n_neighbors=GRAPHST_CLUSTER_N_NEIGHBORS,
        use_rep="emb_pca",
        metric=GRAPHST_CLUSTER_METRIC,
        method=GRAPHST_CLUSTER_CONNECTIVITY_METHOD,
        random_state=GRAPHST_CLUSTER_GRAPH_SEED,
    )

    graph = cluster_adata.obsp["connectivities"].tocsr(copy=True)

    graph.sort_indices()

    tmp_path = graph_path.with_name(graph_path.stem + ".tmp.npz")
    sparse.save_npz(tmp_path, graph, compressed=True)
    os.replace(tmp_path, graph_path)

    return graph


def _train_and_build_one(spec: SampleSpec, device: torch.device):
    input_path = DATA_ROOT / spec.h5ad_name

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    target, n_obs = target_and_size(input_path)

    if n_obs <= GRAPHST_CLUSTER_N_NEIGHBORS:
        raise ValueError(
            f"{spec.key}: {n_obs} spots are too few for 50-neighbor clustering"
        )
    if spec.datatype == "10X" and n_obs > 10000:
        print(
            f"[{spec.key}] GraphST 10X uses dense spatial matrices: "
            f"one float64 N x N matrix is ~{8 * n_obs**2 / 1e9:.1f} GB"
        )

    method_dir = OUT_ROOT / spec.output_group / METHOD_NAME

    embedding_dir = method_dir / "shared_embedding"

    graph_dir = method_dir / "shared_graph"

    embedding_dir.mkdir(parents=True, exist_ok=True)

    graph_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = embedding_dir / embedding_filename(spec)

    graph_path = graph_dir / graph_filename(spec)

    names_path = embedding_dir / f"{representation_tag(spec)}_obs_names.npy"
    manifest_path = embedding_dir / f"{representation_tag(spec)}_manifest.json"
    settings = {
        "input_sha256": sha256_file(input_path),
        "datatype": spec.datatype,
        "graphst_version": installed_version("GraphST"),
        "graphst_source_sha256": sha256_file(Path(inspect.getfile(GraphST.GraphST))),
        "shared_functions_sha256": sha256_file(Path(inspect.getfile(preprocessing))),
        "scanpy_version": sc.__version__,
        "torch_version": torch.__version__,
        "sklearn_version": installed_version("scikit-learn"),
        "training_seed": GRAPHST_TRAINING_SEED,
        "preprocess": [
            GRAPHST_N_HVG,
            GRAPHST_HVG_FLAVOR,
            GRAPHST_NORMALIZE_TARGET_SUM,
            GRAPHST_SCALE,
            GRAPHST_ZERO_CENTER,
            GRAPHST_SCALE_MAX_VALUE,
            "batch=False",
            "check_raw_counts=True",
        ],
        "model_defaults": [
            GRAPHST_LEARNING_RATE,
            GRAPHST_WEIGHT_DECAY,
            GRAPHST_EPOCHS,
            GRAPHST_DIM_OUTPUT,
            GRAPHST_ALPHA,
            GRAPHST_BETA,
            GRAPHST_DECONVOLUTION,
        ],
        "pca": [GRAPHST_CLUSTER_PCS, GRAPHST_CLUSTER_PCA_SEED],
        "neighbors": [
            GRAPHST_CLUSTER_N_NEIGHBORS,
            GRAPHST_CLUSTER_METRIC,
            GRAPHST_CLUSTER_CONNECTIVITY_METHOD,
            GRAPHST_CLUSTER_GRAPH_SEED,
        ],
    }
    manifest = {}
    if manifest_path.exists() and not FORCE_RETRAIN:
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            pass

    if (
        embedding_path.exists()
        and names_path.exists()
        and not FORCE_RETRAIN
        and manifest.get("settings") == settings
        and sha256_file(embedding_path) == manifest.get("embedding_sha256")
        and sha256_file(names_path) == manifest.get("names_sha256")
    ):
        embedding = np.load(embedding_path, mmap_mode="r", allow_pickle=False)

        expected_dim = min(GRAPHST_CLUSTER_PCS, GRAPHST_DIM_OUTPUT, n_obs - 1)

        if (
            embedding.ndim == 2
            and embedding.shape[0] == n_obs
            and embedding.shape[1] == expected_dim
            and np.isfinite(embedding).all()
        ):
            embedding_dim = int(embedding.shape[1])

            graph_ok = False

            if graph_path.exists() and sha256_file(graph_path) == manifest.get(
                "graph_sha256"
            ):
                graph = sparse.load_npz(graph_path).tocsr()

                graph_ok = graph.shape == (n_obs, n_obs)

                if graph_ok:
                    print(
                        f"[{spec.key}] reuse exact GraphST PCA representation + Leiden graph: "
                        f"n={n_obs:,}, dim={embedding_dim}, target={target}"
                    )

                else:
                    print(
                        f"[{spec.key}] cached Leiden graph shape={graph.shape}; "
                        "rebuilding from the saved GraphST representation"
                    )

                del graph

            if not graph_ok:
                graph = build_leiden_graph_from_embedding(embedding, graph_path)

                manifest["graph_sha256"] = sha256_file(graph_path)
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

                print(
                    f"[{spec.key}] rebuilt Leiden graph from saved GraphST representation: "
                    f"nodes={graph.shape[0]:,}, nnz={graph.nnz:,}"
                )

                del graph

            del embedding

            return {
                "key": spec.key,
                "reused_embedding": True,
                "embedding_path": str(embedding_path),
                "graph_path": str(graph_path),
                "n_obs": n_obs,
                "target_clusters": target,
                "datatype": spec.datatype,
                "embedding_dim": embedding_dim,
                "names_path": str(names_path),
                "embedding_sha256": manifest["embedding_sha256"],
                "graph_sha256": manifest["graph_sha256"],
            }

        print(
            f"[{spec.key}] cached GraphST representation shape={embedding.shape} is invalid; "
            "retraining GraphST once"
        )

        del embedding

    elif graph_path.exists() and not FORCE_RETRAIN:
        print(
            f"[{spec.key}] old Leiden graph exists but exact GraphST PCA representation "
            "is missing; retraining once so Leiden and mclust use the same input"
        )

    print(
        f"[{spec.key}] GraphST train ONCE on {device}; "
        f"n={n_obs:,}, target={target}, datatype={spec.datatype}"
    )

    preprocessing_started = time.perf_counter()
    adata = sc.read_h5ad(input_path)

    if not adata.obs_names.is_unique:
        raise ValueError(f"{spec.key}: nonunique obs_names prevent label alignment")
    if "spatial" not in adata.obsm or adata.obsm["spatial"].shape != (n_obs, 2):
        raise ValueError(f"{spec.key}: missing (n_obs, 2) spatial coordinates")
    if not np.isfinite(adata.obsm["spatial"]).all():
        raise ValueError(f"{spec.key}: nonfinite spatial coordinates")
    adata.var_names_make_unique()
    for key in ("adj", "graph_neigh", "distance_matrix", "feat", "feat_a", "label_CSL"):
        if key in adata.obsm:
            del adata.obsm[key]
    adata.uns.pop("log1p", None)
    adata = preprocessing(
        adata,
        highly_var=True,
        flavor=GRAPHST_HVG_FLAVOR,
        n_hvg=GRAPHST_N_HVG,
        batch=False,
        check_raw_counts=True,
        normalize_target_sum=GRAPHST_NORMALIZE_TARGET_SUM,
        scale=GRAPHST_SCALE,
        zero_center=GRAPHST_ZERO_CENTER,
        max_value=GRAPHST_SCALE_MAX_VALUE,
    )
    log_timing("preprocessing", preprocessing_started, sample=spec.key)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with timed("graphst_training", sample=spec.key, device=device):
        model = GraphST.GraphST(
            adata,
            device=device,
            random_seed=GRAPHST_TRAINING_SEED,
            datatype=spec.datatype,
        )

        trained = model.train()

    postprocess_started = time.perf_counter()

    if "emb" not in trained.obsm:
        raise KeyError(f"{spec.key}: GraphST training did not produce obsm['emb']")

    emb = np.asarray(trained.obsm["emb"])

    n_pcs = min(GRAPHST_CLUSTER_PCS, emb.shape[1], emb.shape[0] - 1)

    pca = PCA(n_components=n_pcs, random_state=GRAPHST_CLUSTER_PCA_SEED)

    emb_pca = pca.fit_transform(emb.copy())

    if emb_pca.ndim != 2 or emb_pca.shape[0] != n_obs or not np.isfinite(emb_pca).all():
        raise RuntimeError(
            f"{spec.key}: unexpected GraphST PCA representation shape {emb_pca.shape}"
        )

    manifest_path.unlink(missing_ok=True)
    tmp_embedding = embedding_path.with_name(embedding_path.stem + ".tmp.npy")
    np.save(tmp_embedding, emb_pca, allow_pickle=False)
    os.replace(tmp_embedding, embedding_path)
    tmp_names = names_path.with_name(names_path.stem + ".tmp.npy")
    np.save(tmp_names, np.asarray(adata.obs_names, dtype=str), allow_pickle=False)
    os.replace(tmp_names, names_path)

    embedding_dim = int(emb_pca.shape[1])

    embedding_shape = tuple(emb_pca.shape)

    del emb_pca

    frozen_embedding = np.load(embedding_path, mmap_mode="r", allow_pickle=False)

    graph = build_leiden_graph_from_embedding(frozen_embedding, graph_path)

    manifest = {
        "settings": settings,
        "embedding_sha256": sha256_file(embedding_path),
        "names_sha256": sha256_file(names_path),
        "graph_sha256": sha256_file(graph_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"[{spec.key}] exact shared GraphST PCA representation saved: "
        f"{embedding_path.name}; shape={embedding_shape}; Leiden graph nnz={graph.nnz:,}"
    )
    log_timing("pca_and_graph", postprocess_started, sample=spec.key)

    del graph, frozen_embedding, emb, trained, model, adata

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "key": spec.key,
        "reused_embedding": False,
        "embedding_path": str(embedding_path),
        "graph_path": str(graph_path),
        "n_obs": n_obs,
        "target_clusters": target,
        "datatype": spec.datatype,
        "embedding_dim": embedding_dim,
        "names_path": str(names_path),
        "embedding_sha256": manifest["embedding_sha256"],
        "graph_sha256": manifest["graph_sha256"],
    }


def train_and_build_one(spec: SampleSpec, device: torch.device):
    started = time.perf_counter()
    try:
        result = _train_and_build_one(spec, device)
    except BaseException:
        log_timing(
            "sample_prepare", started, sample=spec.key, device=device, status="error"
        )
        raise
    else:
        log_timing(
            "sample_prepare",
            started,
            sample=spec.key,
            device=device,
            reused_embedding=result["reused_embedding"],
        )
        return result


def train_gpu_batch(specs, gpu_id):
    "One joblib worker owns one GPU and trains its assigned samples serially."

    device = torch.device(f"cuda:{gpu_id}")

    torch.cuda.set_device(gpu_id)

    out = []

    for spec in specs:
        out.append(train_and_build_one(spec, device))

    return out


def train_cpu_batch(specs):
    device = torch.device("cpu")

    return [train_and_build_one(spec, device) for spec in specs]


def _tune_one_sample(spec: SampleSpec, info: dict):
    if spec.key in RESOLUTION_OVERRIDES:
        res = float(RESOLUTION_OVERRIDES[spec.key])

        print(f"[{spec.key}] manual GraphST-Leiden resolution={res:.4f}")

        return spec.key, res

    graph = sparse.load_npz(info["graph_path"]).tocsr()

    target = int(info["target_clusters"])

    res = search_graphst_resolution(graph, target)

    print(
        f"[{spec.key}] GraphST fixed embedding/graph: target={target}; "
        f"Leiden resolution={res:.4f}"
    )

    return spec.key, float(res)


def tune_one_sample(spec: SampleSpec, info: dict):
    with timed("resolution_search", sample=spec.key):
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

    sc.tl.leiden(
        tmp,
        adjacency=graph,
        resolution=float(resolution),
        random_state=int(seed),
        key_added="domain",
        directed=GRAPHST_CLUSTER_DIRECTED,
        use_weights=GRAPHST_CLUSTER_USE_WEIGHTS,
        n_iterations=GRAPHST_CLUSTER_N_ITERATIONS,
        **leiden_backend_kwargs(),
    )

    labels = tmp.obs["domain"].astype(int).to_numpy()
    if labels.size != info["n_obs"]:
        raise RuntimeError(f"{spec.key}: Leiden label count differs from spot count")

    method_dir = OUT_ROOT / spec.output_group / METHOD_NAME

    temp_dir = method_dir / "temp_leiden"

    temp_dir.mkdir(parents=True, exist_ok=True)

    out_path = temp_dir / f"{spec.key}-graphst-leiden-seed{seed}.pkl"

    joblib.dump(labels, out_path, compress=0)

    return spec.key, int(seed), str(out_path), int(np.unique(labels).size)


def leiden_one(spec: SampleSpec, info: dict, resolution: float, seed: int):
    with timed("leiden_run", sample=spec.key, seed=seed):
        return _leiden_one(spec, info, resolution, seed)


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
            "GraphST+mclust requires a working R installation, rpy2, and the "
            "R package 'mclust' inside the GraphST environment."
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


def _mclust_one(spec: SampleSpec, info: dict, seed: int):
    "Run R mclust on the exact same saved PCA-20 input used for Leiden."

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
    if np.any(labels < 1):
        raise RuntimeError(f"{spec.key}: mclust returned invalid cluster IDs")

    method_dir = OUT_ROOT / spec.output_group / METHOD_NAME

    temp_dir = method_dir / "temp_mclust"

    temp_dir.mkdir(parents=True, exist_ok=True)

    out_path = temp_dir / f"{spec.key}-graphst-mclust-seed{seed}.pkl"

    joblib.dump(labels, out_path, compress=0)

    return spec.key, int(seed), str(out_path), int(np.unique(labels).size)


def mclust_one(spec: SampleSpec, info: dict, seed: int):
    with timed("mclust_run", sample=spec.key, seed=seed):
        return _mclust_one(spec, info, seed)


def check_cpu_worker_imports():
    "Catch fresh-worker C++/SciPy/R loading problems before model training."
    import scipy
    from GraphST import GraphST as _worker_graphst
    from functions import preprocessing as _worker_preprocessing

    _get_mclust_backend()
    return scipy.__version__


def consolidate(
    spec: SampleSpec, records, clustering_method: str, info: dict, resolution=None
):
    recs = sorted(
        [r for r in records if r[0] == spec.key],
        key=lambda r: r[1],
    )

    if len(recs) != ITERATIONS:
        raise RuntimeError(
            f"{spec.key}: expected {ITERATIONS} {clustering_method} runs, got {len(recs)}"
        )

    expected_seeds = list(range(CLUSTER_BASE_SEED, CLUSTER_BASE_SEED + ITERATIONS))
    if [int(r[1]) for r in recs] != expected_seeds:
        raise RuntimeError(
            f"{spec.key}: missing or duplicated {clustering_method} seed"
        )

    results = [joblib.load(r[2]) for r in recs]
    if any(np.asarray(x).shape != (info["n_obs"],) for x in results):
        raise RuntimeError(f"{spec.key}: mismatched {clustering_method} label lengths")

    method_dir = OUT_ROOT / spec.output_group / METHOD_NAME

    method_dir.mkdir(parents=True, exist_ok=True)

    if clustering_method == "leiden":
        if spec.output_group == "dlpfc":
            section = spec.key.replace("dlpfc_", "")

            name = f"graphst_results_{section}_{ITERATIONS}iterations.pkl"

        else:
            name = f"graphst_results_{ITERATIONS}iterations.pkl"

    elif clustering_method == "mclust":
        if spec.output_group == "dlpfc":
            section = spec.key.replace("dlpfc_", "")

            name = f"graphst_mclust_results_{section}_{ITERATIONS}iterations.pkl"

        else:
            name = f"graphst_mclust_results_{ITERATIONS}iterations.pkl"

    else:
        raise ValueError(f"Unknown clustering_method={clustering_method!r}")

    joblib.dump(results, method_dir / name, compress=0)
    from sklearn.metrics import adjusted_rand_score

    ari_to_first = [float(adjusted_rand_score(results[0], x)) for x in results]
    counts = [int(np.unique(x).size) for x in results]
    diagnostics = {
        "sample": spec.key,
        "method": clustering_method,
        "n_obs": info["n_obs"],
        "target_clusters": info["target_clusters"],
        "seeds": expected_seeds,
        "cluster_counts": counts,
        "ari_to_first": ari_to_first,
        "all_same_partition_as_first": all(np.isclose(x, 1.0) for x in ari_to_first),
        "labels_file": name,
        "obs_names_file": str(info["names_path"]),
        "embedding_sha256": info["embedding_sha256"],
        "graph_sha256": info["graph_sha256"] if clustering_method == "leiden" else None,
        "resolution": (float(resolution) if resolution is not None else None),
        "mclust_default_subset_note": (
            "Check mclust.options('subset'): its default 2000-spot random "
            "hierarchical initialization applies only when n_obs > 2000"
            if clustering_method == "mclust"
            else None
        ),
    }
    (method_dir / f"{spec.key}_{clustering_method}_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    print(
        f"[{spec.key}] {clustering_method}: cluster counts "
        f"{sorted(set(counts))}, ARI to first {min(ari_to_first):.4f}.."
        f"{max(ari_to_first):.4f}"
    )


def print_graphst_settings():
    print(
        "GraphST fixed-representation experiment: Leiden + mclust on one shared PCA-20 input"
    )

    print("  preprocessing:")

    print(f"    HVGs={GRAPHST_N_HVG} ({GRAPHST_HVG_FLAVOR})")

    print(f"    normalize_total={GRAPHST_NORMALIZE_TARGET_SUM:g}; log1p=True")

    print(
        f"    scale={GRAPHST_SCALE}, zero_center={GRAPHST_ZERO_CENTER}, "
        f"max_value={GRAPHST_SCALE_MAX_VALUE}"
    )

    print(
        f"    GraphST spatial neighbors={GRAPHST_SPATIAL_N_NEIGHBORS} (package default)"
    )

    print("  GraphST model:")

    print(f"    training seed={GRAPHST_TRAINING_SEED} (FIXED; trained once)")

    print(
        f"    epochs={GRAPHST_EPOCHS}, lr={GRAPHST_LEARNING_RATE}, weight_decay={GRAPHST_WEIGHT_DECAY}"
    )

    print(
        f"    dim_output={GRAPHST_DIM_OUTPUT}, alpha={GRAPHST_ALPHA}, beta={GRAPHST_BETA}"
    )

    print("  shared downstream representation:")

    print(
        f"    PCA={GRAPHST_CLUSTER_PCS}, PCA seed={GRAPHST_CLUSTER_PCA_SEED}; "
        "saved once and used by BOTH Leiden and mclust"
    )

    print("  Leiden-only fixed graph:")

    print(
        f"    KNN={GRAPHST_CLUSTER_N_NEIGHBORS}, metric={GRAPHST_CLUSTER_METRIC}, "
        f"connectivity={GRAPHST_CLUSTER_CONNECTIVITY_METHOD}, graph seed={GRAPHST_CLUSTER_GRAPH_SEED}"
    )

    print("  downstream Leiden:")

    print(
        f"    use_weights={GRAPHST_CLUSTER_USE_WEIGHTS}, directed={GRAPHST_CLUSTER_DIRECTED}, "
        f"n_iterations={GRAPHST_CLUSTER_N_ITERATIONS}, flavor={GRAPHST_CLUSTER_FLAVOR}"
    )

    print(
        f"    resolution search={GRAPHST_RES_START}..{GRAPHST_RES_END}, "
        f"increment={GRAPHST_RES_INCREMENT}, seed={RESOLUTION_SEED}"
    )

    print("  downstream mclust:")

    print(
        f"    R mclust::Mclust, modelNames={MCLUST_MODEL}, "
        "G=annotation cluster count, no KNN/resolution"
    )

    print(
        f"    refinement={GRAPHST_REFINEMENT} for GraphST helper; no refinement applied here"
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
    print(f"Hardware and timing logs: {LOG_ROOT}", flush=True)
    print(f"Hardware: {json.dumps(hardware, default=str)}", flush=True)
    print_graphst_settings()

    with timed("worker_import_preflight"):
        joblib.Parallel(n_jobs=min(2, ALLOCATED_CPUS), backend="loky")(
            joblib.delayed(check_cpu_worker_imports)()
            for _ in range(min(2, ALLOCATED_CPUS))
        )

    print(
        "\nPHASE 1/5: train GraphST once per sample; save shared representation + Leiden graph"
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
            print("No CUDA GPU visible; GraphST training will run serially on CPU.")
            prepared = train_cpu_batch(SAMPLES)

    info_by_key = {x["key"]: x for x in prepared}
    print("\nTargets derived from adata.obs['annotation']:")
    print({k: v["target_clusters"] for k, v in info_by_key.items()})

    print("\nPHASE 2/5: tune GraphST+Leiden resolution on the frozen graph")
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
        f"\nPHASE 3/5: submitting {len(leiden_tasks)} GraphST+Leiden jobs "
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
        f"\nPHASE 4/5: submitting {len(seed_tasks)} GraphST+mclust jobs "
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
            with timed("leiden_consolidate", sample=spec.key):
                consolidate(
                    spec,
                    leiden_records,
                    "leiden",
                    info_by_key[spec.key],
                    resolutions[spec.key],
                )
            with timed("mclust_consolidate", sample=spec.key):
                consolidate(spec, mclust_records, "mclust", info_by_key[spec.key])
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
        write_timing_summary()
