"Shared count validation, preprocessing, and clustering helpers."

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from scipy import sparse


def _check_nonnegative_integer_counts(X, atol=1e-6, chunk_size=1_000_000):
    "Return True only if all stored values are finite, non-negative integers."
    values = X.data if sparse.issparse(X) else np.asarray(X).reshape(-1)

    for i in range(0, values.size, chunk_size):
        chunk = np.asarray(values[i : i + chunk_size])
        if not np.all(np.isfinite(chunk)):
            return False
        if np.any(chunk < -atol):
            return False
        if np.any(np.abs(chunk - np.rint(chunk)) > atol):
            return False
    return True


def get_raw_counts(adata, prefer=("counts", "raw", "X"), check_values=True):
    "Find raw counts aligned to ``adata.var_names`` without storing a new layer."
    candidates = []

    if "counts" in adata.layers:
        candidates.append(("adata.layers['counts']", adata.layers["counts"]))

    candidates.append(("adata.X", adata.X))

    if adata.raw is not None:
        raw_names = pd.Index(adata.raw.var_names)
        loc = raw_names.get_indexer(adata.var_names)
        if np.all(loc >= 0):
            candidates.append(("adata.raw.X", adata.raw.X[:, loc]))

    order = {
        "counts": "adata.layers['counts']",
        "X": "adata.X",
        "raw": "adata.raw.X",
    }
    candidates_by_name = {name: X for name, X in candidates}

    for key in prefer:
        name = order[key]
        if name not in candidates_by_name:
            continue
        X = candidates_by_name[name]
        if X.shape != adata.shape:
            continue
        if (not check_values) or _check_nonnegative_integer_counts(X):
            return X, name

    available = ", ".join(candidates_by_name) or "none"
    raise ValueError(
        "Could not find a matrix that looks like raw non-negative integer "
        f"counts for Seurat-v3 HVG selection. Checked: {available}."
    )


def get_target_n_clusters(adata, annotation_key="annotation"):
    "Count observed, non-missing annotation labels in ``adata.obs``."
    if annotation_key not in adata.obs:
        raise KeyError(
            f"adata.obs does not contain {annotation_key!r}; cannot derive "
            "the target number of clusters."
        )

    labels = adata.obs[annotation_key]
    labels = labels[labels.notna()].astype(str).str.strip()
    labels = labels[labels != ""]
    unique_labels = pd.Index(labels.unique())

    if len(unique_labels) == 0:
        raise ValueError(
            f"adata.obs[{annotation_key!r}] contains no non-missing labels."
        )

    return int(len(unique_labels))


def get_n_neighbors(sample, default=15, overrides=None):
    "Resolve KNN ``k`` with an optional per-dataset override dictionary."
    overrides = {} if overrides is None else overrides
    k = int(overrides.get(sample, default))
    if k < 2:
        raise ValueError(f"n_neighbors must be >= 2, got {k} for {sample}.")
    return k


def preprocessing(
    adata,
    highly_var=True,
    flavor="seurat_v3",
    n_hvg=3000,
    batch=False,
    check_raw_counts=True,
    normalize_target_sum=1e4,
    scale=True,
    zero_center=True,
    max_value=10,
):
    "General expression preprocessing with explicit, reproducible options."
    if adata.n_obs <= 0:
        print(f"Data not preprocessed because there are {adata.n_obs} spots!")
        return adata

    batch_key = "batch" if batch else None

    # Normalize verified counts, not a previously transformed matrix.

    counts, count_source = get_raw_counts(
        adata,
        check_values=check_raw_counts,
    )
    print(f"Raw-count source: {count_source}")

    if count_source != "adata.X":
        adata.X = counts.copy()

    if highly_var and flavor == "seurat_v3":
        # Select Seurat-v3 HVGs before normalization.

        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=min(n_hvg, adata.n_vars),
            batch_key=batch_key,
            check_values=check_raw_counts,
        )

    sc.pp.normalize_total(adata, target_sum=normalize_target_sum)
    sc.pp.log1p(adata)

    if highly_var and flavor != "seurat_v3":
        sc.pp.highly_variable_genes(
            adata,
            flavor=flavor,
            n_top_genes=min(n_hvg, adata.n_vars),
            batch_key=batch_key,
        )

    if highly_var:
        print(f"Selected {int(adata.var['highly_variable'].sum())} HVGs")
        adata.uns["hvg_info"] = adata.var[["highly_variable"]].copy()
        adata = adata[:, adata.var["highly_variable"]].copy()

    if scale:
        sc.pp.scale(
            adata,
            zero_center=zero_center,
            max_value=max_value,
        )

    sample_list = (
        list(adata.uns["spatial"].keys()) if "spatial" in adata.uns else ["Unknown"]
    )
    all_label = "Sample: " + ",".join(sample_list)
    print(f"{all_label} preprocessed: {adata.n_obs} spots x {adata.n_vars} genes")
    return adata


def _cluster_count_from_adjacency(
    adjacency,
    resolution,
    method="leiden",
    random_state=0,
    directed=False,
    use_weights=True,
    leiden_n_iterations=-1,
    leiden_flavor="leidenalg",
    louvain_flavor="vtraag",
):
    "Run one clustering on an explicitly supplied graph and return #clusters."
    n_obs = adjacency.shape[0]
    tmp = ad.AnnData(
        X=sparse.csr_matrix((n_obs, 0), dtype=np.float32),
        obs=pd.DataFrame(index=pd.RangeIndex(n_obs).astype(str)),
    )

    if method == "leiden":
        kwargs = dict(
            adjacency=adjacency,
            resolution=float(resolution),
            key_added="_cluster",
            directed=directed,
            use_weights=use_weights,
            n_iterations=leiden_n_iterations,
            flavor=leiden_flavor,
        )

        try:
            sc.tl.leiden(tmp, random_state=int(random_state), **kwargs)
        except TypeError:
            sc.tl.leiden(tmp, rng=int(random_state), **kwargs)

    elif method == "louvain":
        sc.tl.louvain(
            tmp,
            adjacency=adjacency,
            resolution=float(resolution),
            random_state=int(random_state),
            key_added="_cluster",
            directed=directed,
            use_weights=use_weights,
            flavor=louvain_flavor,
        )
    else:
        raise ValueError("method must be 'leiden' or 'louvain'")

    return int(tmp.obs["_cluster"].nunique())


def search_res(
    adjacency,
    n_clusters,
    method="leiden",
    start=0.1,
    end=3.0,
    increment=0.01,
    tolerance=2,
    random_state=0,
    directed=False,
    use_weights=True,
    leiden_n_iterations=-1,
    leiden_flavor="leidenalg",
    louvain_flavor="vtraag",
    verbose=False,
):
    "Adaptively search resolution on the exact graph used for clustering."
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if start <= 0 or end <= start:
        raise ValueError("Require 0 < start < end.")
    if increment <= 0:
        raise ValueError("increment must be > 0.")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0.")
    if method not in {"leiden", "louvain"}:
        raise ValueError("method must be 'leiden' or 'louvain'")

    evaluated = {}

    def evaluate(res):
        res = round(float(res), 10)
        if res not in evaluated:
            count = _cluster_count_from_adjacency(
                adjacency=adjacency,
                resolution=res,
                method=method,
                random_state=random_state,
                directed=directed,
                use_weights=use_weights,
                leiden_n_iterations=leiden_n_iterations,
                leiden_flavor=leiden_flavor,
                louvain_flavor=louvain_flavor,
            )
            evaluated[res] = count
            if verbose:
                print(f"{method}: resolution={res:.4f}, clusters={count}")
        return evaluated[res]

    if verbose:
        print(
            f"Searching {method} resolution for target={n_clusters} clusters "
            f"on the supplied adjacency..."
        )

    res = float(end)
    best_res = None
    best_diff = float("inf")
    best_count = None

    prev_res = None
    prev_count = None
    coarse = True

    while res >= start - 1e-12:
        res = round(max(res, start), 10)
        count = evaluate(res)

        if count == n_clusters:
            if verbose:
                print(
                    f"{method}: exact match at resolution={res:.4f}, "
                    f"clusters={count}"
                )
            return float(res)

        diff = count - n_clusters
        abs_diff = abs(diff)

        if abs_diff < best_diff:
            best_diff = abs_diff
            best_res = res
            best_count = count

        if (
            prev_res is not None
            and prev_count is not None
            and prev_count > n_clusters
            and count < n_clusters
        ):
            res = prev_res - increment
            prev_res = None
            prev_count = None
            coarse = False
            continue

        if coarse and diff >= 5:
            step = 30 * increment
        elif coarse and diff >= 3:
            step = 10 * increment
        elif coarse and diff >= 2:
            step = 5 * increment
        else:
            step = increment
            coarse = False

        if step > increment:
            prev_res = res
            prev_count = count

        next_res = res - step
        if next_res < start and res > start:
            res = start
        else:
            res = next_res

    if best_res is not None and best_diff <= tolerance:
        print(
            f"Warning: {method} found no exact resolution for target={n_clusters}. "
            f"Using resolution={best_res:.4f}, which produced {best_count} clusters "
            f"at random_state={random_state}."
        )
        return float(best_res)

    raise ValueError(
        f"{method}: no resolution in [{start}, {end}] reached target={n_clusters} "
        f"within tolerance={tolerance}. Best was resolution={best_res}, "
        f"clusters={best_count}."
    )
