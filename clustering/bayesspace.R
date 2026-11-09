#!/usr/bin/env Rscript
# BayesSpace: fixed preprocessing and 50 MCMC seeds (42..91).
# Zhao et al., Nature Biotechnology (2021), doi:10.1038/s41587-021-00935-2.
# The mouse cerebellum uses a project-specific symmetric spatial kNN adaptation.
# Usage: Rscript clustering/bayesspace.R
# Paths: INSTABILITY_DATA_ROOT and INSTABILITY_OUTPUT_ROOT; relative to this repository.

suppressPackageStartupMessages({
    library(zellkonverter)
    library(BayesSpace)
    library(SingleCellExperiment)
    library(SummarizedExperiment)
    library(S4Vectors)
    library(Matrix)
    library(BiocSingular)
    library(BiocParallel)
    library(BiocNeighbors)
    library(parallel)
})
if (!requireNamespace("clue", quietly=TRUE)) {
    stop("Install the R package 'clue' for optimal alignment of MCMC cluster labels.")
}

script_args <- grep("^--file=", commandArgs(trailingOnly=FALSE), value=TRUE)
if (length(script_args) != 1L) {
    stop("Run this script with Rscript clustering/bayesspace.R.")
}
SCRIPT_PATH <- normalizePath(sub("^--file=", "", script_args[[1L]]), mustWork=TRUE)
REPO_ROOT <- dirname(dirname(SCRIPT_PATH))

resolve_repo_path <- function(value) {
    value <- path.expand(value)
    if (!startsWith(value, "/") && !grepl("^[A-Za-z]:", value)) {
        value <- file.path(REPO_ROOT, value)
    }
    normalizePath(value, winslash="/", mustWork=FALSE)
}

DATA_ROOT <- resolve_repo_path(Sys.getenv("INSTABILITY_DATA_ROOT", unset="data"))
OUT_ROOT <- resolve_repo_path(Sys.getenv("INSTABILITY_OUTPUT_ROOT", unset="results"))

# Fixed experiment settings.
ANNOTATION_KEY <- "annotation"
ITERATIONS <- 50L
BASE_SEED <- 42L
SEEDS <- BASE_SEED + seq_len(ITERATIONS) - 1L
N_HVGS <- 2000L
N_PCS <- 15L
LOG_NORMALIZE <- TRUE

# Use seeded approximate PCA only above this sample size.
FAST_PCA_MIN_SPOTS <- 10000L
PREPROCESS_SEED <- 42L
INIT_METHOD <- "mclust"
ERROR_MODEL <- "t"
PRECISION_MODEL <- "equal"
NREP <- 50000L
BURN_IN <- 1000L
THIN <- 100L
ALPHA <- 1
BETA <- 0.01

# Retain native chains temporarily to estimate posterior memberships.
SAVE_CHAIN_TEMPORARILY <- TRUE

# Project adaptation for irregular Slide-seq beads: symmetric kNN, not a native lattice.
SLIDESEQ_K <- 10L
SLIDESEQ_GAMMA <- 3

# Limit concurrent MCMC chains to the available CPUs.
ALLOCATED_CPUS <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "")))
if (is.na(ALLOCATED_CPUS) || ALLOCATED_CPUS < 1L) {
    ALLOCATED_CPUS <- parallel::detectCores(logical=FALSE)
}
if (is.na(ALLOCATED_CPUS) || ALLOCATED_CPUS < 1L) ALLOCATED_CPUS <- 1L
DEFAULT_JOBS <- min(120L, ALLOCATED_CPUS, ITERATIONS)
REQUESTED_JOBS <- suppressWarnings(as.integer(Sys.getenv(
    "BAYESSPACE_JOBS", unset=as.character(DEFAULT_JOBS)
)))
if (is.na(REQUESTED_JOBS) || REQUESTED_JOBS < 1L) {
    stop("BAYESSPACE_JOBS must be a positive integer.")
}
N_JOBS <- min(REQUESTED_JOBS, ALLOCATED_CPUS, ITERATIONS)

# The 18-sample benchmark; full, uncropped Visium HD datasets are excluded.
samples <- data.frame(
    key = c(
        "Visium_HD_Human_Colon_Cancer_cropped_square",
        "mouse_brain",
        "mouse_brain_cerebellum",
        "human_breast_cancer",
        "ov_ffpe",
        "coad_ffpe",
        paste0("dlpfc_", c(
            "151507", "151508", "151509", "151510",
            "151669", "151670", "151671", "151672",
            "151673", "151674", "151675", "151676"
        ))
    ),
    h5ad = c(
        "Visium_HD_Human_Colon_Cancer_cropped_square_filtered.h5ad",
        "mouse_brain_filtered.h5ad",
        "mouse_brain_cerebellum_filtered.h5ad",
        "human_breast_cancer_filtered.h5ad",
        "ov_ffpe_filtered.h5ad",
        "coad_ffpe_filtered.h5ad",
        paste0("dlpfc_", c(
            "151507", "151508", "151509", "151510",
            "151669", "151670", "151671", "151672",
            "151673", "151674", "151675", "151676"
        ), "_filtered.h5ad")
    ),
    output_group = c(
        "Visium_HD_Human_Colon_Cancer_cropped_square",
        "mouse_brain",
        "mouse_brain_cerebellum",
        "human_breast_cancer",
        "ov_ffpe",
        "coad_ffpe",
        rep("dlpfc", 12)
    ),
    platform = c(
        "VisiumHD",
        "Visium",
        "SlideSeq",
        "Visium",
        "VisiumHD",
        "VisiumHD",
        rep("Visium", 12)
    ),
    reader_r = rep(TRUE, 18),
    stringsAsFactors = FALSE
)

# Resume these samples only when all expected artifacts are present.
COMPLETED_ON_PREVIOUS_RUN <- c(
    "Visium_HD_Human_Colon_Cancer_cropped_square",
    "mouse_brain", "mouse_brain_cerebellum", "human_breast_cancer"
)

is_integer_count_block <- function(x, atol=1e-6) {
    vals <- if (inherits(x, "sparseMatrix")) x@x else as.numeric(x)
    if (length(vals) == 0L) return(TRUE)
    all(is.finite(vals)) && all(vals >= -atol) &&
        all(abs(vals - round(vals)) <= atol)
}

# Validate delayed matrices blockwise to limit memory use.
looks_like_raw_counts <- function(x, atol=1e-6) {
    if (inherits(x, "DelayedArray")) {
        checks <- DelayedArray::blockApply(
            x,
            FUN=function(block) is_integer_count_block(block, atol=atol)
        )
        return(all(unlist(checks, use.names=FALSE)))
    }
    is_integer_count_block(x, atol=atol)
}

ensure_counts_assay <- function(spe) {
    names_now <- assayNames(spe)
    candidates <- unique(c(
        if ("counts" %in% names_now) "counts" else character(0),
        if ("X" %in% names_now) "X" else character(0),
        names_now
    ))
    for (nm in candidates) {
        x <- assay(spe, nm)
        if (looks_like_raw_counts(x)) {
            if (nm != "counts") {
                assay(spe, "counts") <- x
            }
            message("Raw-count source: assay '", nm, "'")
            return(spe)
        }
    }
    stop(
        "No assay contains finite, non-negative, integer-valued raw counts. ",
        "Checked assays: ", paste(names_now, collapse=", ")
    )
}

get_target_n_clusters <- function(spe, annotation_key=ANNOTATION_KEY) {
    if (!(annotation_key %in% colnames(colData(spe)))) {
        stop("colData(spe) does not contain '", annotation_key, "'.")
    }
    x <- as.character(colData(spe)[[annotation_key]])
    x <- trimws(x[!is.na(x)])
    x <- x[nzchar(x)]
    labs <- unique(x)
    if (length(labs) < 2L) {
        stop("Need at least two observed annotation labels; found ", length(labs), ".")
    }
    length(labs)
}

# Native Visium/VisiumHD require integer lattice coordinates.
ensure_array_coordinates <- function(spe, platform) {
    cd <- colData(spe)
    if (!all(c("array_row", "array_col") %in% colnames(cd))) {
        if (all(c("row", "col") %in% colnames(cd))) {
            cd$array_row <- cd$row
            cd$array_col <- cd$col
        } else if (platform == "VisiumHD") {
            ids <- colnames(spe)
            pattern <- "^s_([0-9]{3})um_([0-9]+)_([0-9]+)(-[0-9]+)?$"
            if (is.null(ids) || length(ids) != ncol(spe) ||
                anyNA(ids) || anyDuplicated(ids) ||
                !all(grepl(pattern, ids))) {
                stop("VisiumHD array coordinates are missing and spot IDs ",
                     "cannot be safely parsed as 10x square-bin IDs. ",
                     "Check source tissue_positions.parquet and spot IDs.")
            }
            fields <- regmatches(ids, regexec(pattern, ids))
            bin_sizes <- vapply(fields, `[[`, character(1), 2L)
            if (length(unique(bin_sizes)) != 1L) {
                stop("VisiumHD sample contains mixed bin sizes; do not ",
                     "construct a single square-bin neighbor graph.")
            }
            cd$array_row <- as.numeric(vapply(fields, `[[`, character(1), 3L))
            cd$array_col <- as.numeric(vapply(fields, `[[`, character(1), 4L))
            message("[VisiumHD] parsed ", length(ids), " array positions from ",
                    bin_sizes[[1L]], "um spot IDs")
        } else {
            stop(
                "BayesSpace requires array_row/array_col lattice coordinates ",
                "for native platform=", platform, "."
            )
        }
    }
    coord_numeric <- function(x) {
        if (is.factor(x)) x <- as.character(x)
        suppressWarnings(as.numeric(x))
    }
    rr <- coord_numeric(cd$array_row)
    cc <- coord_numeric(cd$array_col)
    if (any(!is.finite(rr)) || any(!is.finite(cc))) {
        stop("array_row/array_col contain non-finite values.")
    }
    if (any(abs(rr - round(rr)) > 1e-8) || any(abs(cc - round(cc)) > 1e-8)) {
        stop("array_row/array_col must be integer lattice coordinates for native BayesSpace.")
    }
    cd$array_row <- as.integer(round(rr))
    cd$array_col <- as.integer(round(cc))
    if (anyDuplicated(paste(cd$array_row, cd$array_col, sep=":"))) {
        stop("Duplicate array_row/array_col pairs would create invalid lattice neighbors.")
    }
    colData(spe) <- cd
    spe
}

# Reject coordinate/platform combinations that produce no spatial edges.
check_native_neighbors <- function(spe, platform) {
    rr <- as.integer(colData(spe)$array_row)
    cc <- as.integer(colData(spe)$array_col)
    offsets <- if (platform == "Visium") {
        cbind(c(0L, 0L, -1L, -1L, 1L, 1L),
              c(-2L, 2L, -1L, 1L, -1L, 1L))
    } else {
        cbind(c(-1L, 0L, 1L, 0L), c(0L, 1L, 0L, -1L))
    }
    spots <- paste(rr, cc, sep=":")
    has_neighbor <- rep(FALSE, length(rr))
    for (j in seq_len(nrow(offsets))) {
        has_neighbor <- has_neighbor |
            paste(rr + offsets[j, 1], cc + offsets[j, 2], sep=":") %in% spots
    }
    n_linked <- sum(has_neighbor)
    message("[", platform, "] spots with native lattice neighbors: ",
            n_linked, "/", length(rr))
    if (n_linked == 0L) {
        stop("No native lattice neighbors found; check the sample's platform ",
             "and that array_row/array_col are array indices, not pixel coordinates.")
    }
    if (n_linked < 0.1 * length(rr)) {
        warning("Fewer than 10% of spots have lattice neighbors; verify ",
                "platform and coordinate scale for this sample.")
    }
    invisible(n_linked)
}

# Use true continuous coordinates for the Slide-seq adaptation.
get_slideseq_coordinates <- function(spe) {
    rd_names <- reducedDimNames(spe)
    spatial_rd <- rd_names[grepl("spatial", tolower(rd_names), fixed=TRUE)]
    if (length(spatial_rd) >= 1L) {
        coords <- as.matrix(reducedDim(spe, spatial_rd[[1L]]))
    } else {
        cd <- as.data.frame(colData(spe))
        candidates <- list(
            c("x", "y"),
            c("X", "Y"),
            c("coor X", "coor Y"),
            c("spatial_x", "spatial_y")
        )
        hit <- NULL
        for (pair in candidates) {
            if (all(pair %in% colnames(cd))) {
                hit <- pair
                break
            }
        }
        if (is.null(hit)) {
            stop(
                "Slide-seq adaptation requires true continuous spatial coordinates. ",
                "Expected reducedDim(spe, 'spatial') from adata.obsm['spatial'] ",
                "or an explicit x/y coordinate pair in colData."
            )
        }
        coords <- as.matrix(cd[, hit, drop=FALSE])
    }
    if (ncol(coords) < 2L) {
        stop("Slide-seq spatial coordinates must have at least two columns.")
    }
    coords <- coords[, 1:2, drop=FALSE]
    storage.mode(coords) <- "double"
    if (nrow(coords) != ncol(spe) || any(!is.finite(coords))) {
        stop("Invalid Slide-seq spatial-coordinate matrix.")
    }
    coords
}

# Symmetrize exact Euclidean kNN edges and return zero-based neighbor IDs.
build_slideseq_neighbors <- function(spe, k=SLIDESEQ_K) {
    coords <- get_slideseq_coordinates(spe)
    n <- nrow(coords)
    if (n <= k) {
        stop("Slide-seq k must be smaller than the number of observations.")
    }
    knn <- BiocNeighbors::findKNN(
        coords,
        k=as.integer(k),
        get.index=TRUE,
        get.distance=FALSE
    )$index
    ii <- rep(seq_len(n), each=ncol(knn))
    jj <- as.vector(t(knn))
    A <- Matrix::sparseMatrix(
        i=ii,
        j=jj,
        x=1,
        dims=c(n, n),
        giveCsparse=TRUE
    )
    A <- (A + t(A)) > 0
    diag(A) <- FALSE
    A <- methods::as(A, "dgCMatrix")
    edges <- summary(A)
    df_j <- unname(split(
        as.integer(edges$j - 1L),
        factor(edges$i, levels=seq_len(n)),
        drop=FALSE
    ))
    degree <- lengths(df_j)
    if (any(degree == 0L)) {
        stop("Slide-seq adaptation produced isolated observations; cannot continue.")
    }
    message(
        "[Slide-seq adaptation] exact symmetric Euclidean ", k, "-NN graph: ",
        "mean degree=", sprintf("%.3f", mean(degree)),
        ", median=", sprintf("%.1f", stats::median(degree)),
        ", range=", min(degree), "..", max(degree)
    )
    list(df_j=df_j, degree=degree)
}

gamma_for_platform <- function(platform) {
    if (platform == "Visium") return(3)
    if (platform %in% c("VisiumHD", "ST")) return(2)
    if (platform == "SlideSeq") return(SLIDESEQ_GAMMA)
    stop("Unknown platform label: ", platform)
}

read_sample_h5ad <- function(path, reader_r=FALSE) {
    if (!file.exists(path)) stop("Input file not found: ", path)
    if (reader_r) {
        readH5AD(path, reader="R")
    } else {
        readH5AD(path)
    }
}

timed_phase <- function(phase, expr, record) {
    start <- Sys.time()
    tick <- unname(proc.time()[["elapsed"]])
    status <- "error"
    on.exit(record(phase, start,
                   unname(proc.time()[["elapsed"]]) - tick, status), add=TRUE)
    value <- eval.parent(substitute(expr))
    status <- "ok"
    value
}

timing_row <- function(sample, seed, phase, start, duration, status) {
    data.frame(sample=sample, seed=as.integer(seed), phase=phase,
               started_utc=format(start, "%Y-%m-%dT%H:%M:%OS3Z", tz="UTC"),
               elapsed_seconds=as.numeric(duration), status=status,
               stringsAsFactors=FALSE)
}

# Align each MCMC state to its final hard labels before estimating membership frequencies.
posterior_memberships <- function(zs, hard_labels, q) {
    n <- length(hard_labels)
    if (!is.matrix(zs) || nrow(zs) < 1L || ncol(zs) != n) {
        stop("MCMC assignment history must be iterations x spots.")
    }
    if (anyNA(zs) || any(zs < 1 | zs > q) ||
        any(abs(zs - round(zs)) > 0)) {
        stop("MCMC history has invalid cluster IDs.")
    }
    counts <- matrix(0L, nrow=n, ncol=q)
    n_aligned <- 0L
    for (i in seq_len(nrow(zs))) {
        z <- as.integer(zs[i, ])
        overlaps <- matrix(tabulate((z - 1L) * q + hard_labels,
                                    nbins=q*q), nrow=q, byrow=TRUE)
        mapping <- as.integer(clue::solve_LSAP(overlaps, maximum=TRUE))
        if (any(mapping != seq_len(q))) n_aligned <- n_aligned + 1L
        assigned <- mapping[z]
        ix <- (assigned - 1L) * n + seq_len(n)
        counts[ix] <- counts[ix] + 1L
    }
    probs <- counts / nrow(zs)
    colnames(probs) <- paste0("cluster_", seq_len(q))
    if (any(abs(rowSums(probs) - 1) > 1e-10)) {
        stop("Posterior memberships fail the row-sum check.")
    }
    list(probabilities=probs, n_retained=nrow(zs),
         n_iterations_relabelled=n_aligned)
}

# Account for readers that retain the initialization state before the thinned samples.
post_burn_z <- function(z, nspots) {
    z <- as.matrix(z)
    expected <- NREP %/% THIN
    burn <- BURN_IN %/% THIN
    if (ncol(z) != nspots || burn >= expected) {
        stop("Unexpected MCMC z dimensions or burn-in settings.")
    }
    if (nrow(z) == expected + 1L) {
        keep <- seq.int(burn + 2L, expected + 1L)
    } else if (nrow(z) == expected) {
        keep <- seq.int(burn + 1L, expected)
    } else {
        stop("Unexpected number of thinned MCMC states: ", nrow(z),
             " (expected ", expected, " or ", expected + 1L, ").")
    }
    z[keep, , drop=FALSE]
}

# Freeze preprocessing and spatial neighbors before varying the clustering seed.
prepare_sample <- function(spec, record) {
    input_path <- file.path(DATA_ROOT, spec$h5ad)
    message("\n[", spec$key, "] loading ", input_path)
    spe <- timed_phase("read_h5ad",
                       read_sample_h5ad(input_path, isTRUE(spec$reader_r)), record)
    spe <- timed_phase("raw_counts_check", ensure_counts_assay(spe), record)
    if (spec$platform != "SlideSeq") {
        spe <- timed_phase("validate_spatial_graph", {
            spe <- ensure_array_coordinates(spe, spec$platform)
            check_native_neighbors(spe, spec$platform)
            spe
        }, record)
    } else {
        timed_phase("validate_spatial_graph", {
            invisible(get_slideseq_coordinates(spe))
        }, record)
    }
    target <- get_target_n_clusters(spe)
    n_spots <- ncol(spe)
    if (nrow(spe) < N_HVGS) {
        stop("Only ", nrow(spe), " genes remain; cannot select ", N_HVGS,
             " HVGs for the declared preprocessing recipe.")
    }
    if (n_spots <= N_PCS || target >= n_spots) {
        stop("Insufficient spots for ", N_PCS, " PCs and target q=", target, ".")
    }
    use_fast_pca <- n_spots > FAST_PCA_MIN_SPOTS
    pca_method <- if (use_fast_pca) "FastAutoParam" else "ExactParam"
    message(
        "[", spec$key, "] spots=", format(n_spots, big.mark=","),
        ", target q=", target,
        ", platform=", spec$platform,
        ", PCA=", pca_method
    )
    set.seed(PREPROCESS_SEED)
    bsparam <- if (use_fast_pca) {
        BiocSingular::FastAutoParam()
    } else {
        BiocSingular::ExactParam()
    }
    preprocess_platform <- if (spec$platform == "SlideSeq") "ST" else spec$platform
    spe <- timed_phase("spatial_preprocess", spatialPreprocess(
        spe,
        platform=preprocess_platform,
        n.PCs=N_PCS,
        n.HVGs=N_HVGS,
        skip.PCA=FALSE,
        log.normalize=LOG_NORMALIZE,
        assay.type="logcounts",
        BSPARAM=bsparam,
        BPPARAM=BiocParallel::SerialParam()
    ), record)
    if (spec$platform == "SlideSeq") {
        metadata(spe)$BayesSpace.data$platform <- "SlideSeq"
    }
    if (!("PCA" %in% reducedDimNames(spe))) {
        stop("BayesSpace preprocessing did not create reducedDim(spe, 'PCA').")
    }
    if (ncol(reducedDim(spe, "PCA")) < N_PCS) {
        stop("BayesSpace PCA contains fewer than ", N_PCS, " PCs.")
    }
    slide_neighbors <- NULL
    if (spec$platform == "SlideSeq") {
        slide_neighbors <- timed_phase("build_slideseq_graph",
            build_slideseq_neighbors(spe, k=SLIDESEQ_K)$df_j, record)
    }
    spe_cluster <- spe[integer(0), ]
    rm(spe)
    gc(verbose=FALSE)
    list(
        spe=spe_cluster,
        target=target,
        n_spots=n_spots,
        pca_method=pca_method,
        slide_neighbors=slide_neighbors
    )
}

# Supply the fixed Slide-seq graph to the unchanged BayesSpace MCMC core.
spatialCluster_slideseq_adapted <- function(sce, q, df_j) {
    Y <- reducedDim(sce, "PCA")
    d <- min(ncol(Y), N_PCS)
    Y <- Y[, seq_len(d), drop=FALSE]
    init <- BayesSpace:::.init_cluster(
        Y,
        q,
        init=NULL,
        init.method=INIT_METHOD
    )
    if (is.null(init)) {
        stop("Empty BayesSpace initialization for Slide-seq adaptation.")
    }
    mu0 <- colMeans(Y)
    lambda0 <- diag(0.01, ncol(Y))
    results <- BayesSpace:::cluster(
        Y,
        q,
        df_j,
        init=init,
        model=ERROR_MODEL,
        precision=PRECISION_MODEL,
        mu0=mu0,
        lambda0=lambda0,
        gamma=SLIDESEQ_GAMMA,
        alpha=ALPHA,
        beta=BETA,
        nrep=NREP,
        thin=THIN
    )
    burn_index <- BURN_IN %/% THIN
    kept_index <- NREP %/% THIN
    zs <- results$z[seq(burn_index + 2L, kept_index + 1L), , drop=FALSE]
    if (burn_index + 1L == kept_index) {
        labels <- as.vector(zs)
    } else {
        labels <- apply(zs, 2, BayesSpace:::Mode)
    }
    list(labels=as.integer(labels), z=zs)
}

# Save hard labels, aligned posterior memberships, and timings for one chain.
run_one_seed <- function(seed, spe_fixed, q, platform, sample_key,
                         per_run_dir, slide_neighbors=NULL) {
    events <- list()
    record <- function(phase, start, duration, status) {
        events[[length(events) + 1L]] <<- timing_row(
            sample_key, seed, phase, start, duration, status)
    }
    run_start <- Sys.time()
    run_tick <- unname(proc.time()[["elapsed"]])
    completed <- FALSE
    timings_path <- file.path(per_run_dir, paste0("seed_", seed, "_timings.csv"))
    on.exit({
        record("run_total", run_start,
               unname(proc.time()[["elapsed"]]) - run_tick,
               if (completed) "ok" else "error")
        write.csv(do.call(rbind, events), timings_path, row.names=FALSE)
    }, add=TRUE)
    set.seed(seed)
    if (platform == "SlideSeq") {
        if (is.null(slide_neighbors)) {
            stop("Missing fixed Slide-seq neighbor list.")
        }
        result <- timed_phase("mcmc_fit", spatialCluster_slideseq_adapted(
            spe_fixed,
            q=q,
            df_j=slide_neighbors
        ), record)
        labels <- result$labels
        z <- result$z
    } else {
        chain_path <- tempfile(pattern=paste0("seed_", seed, "_chain_"),
                               tmpdir=per_run_dir, fileext=".h5")
        on.exit(unlink(chain_path), add=TRUE)
        fit <- timed_phase("mcmc_fit", spatialCluster(
            spe_fixed,
            q=q,
            use.dimred="PCA",
            d=N_PCS,
            platform=platform,
            init=NULL,
            init.method=INIT_METHOD,
            model=ERROR_MODEL,
            precision=PRECISION_MODEL,
            nrep=NREP,
            burn.in=BURN_IN,
            thin=THIN,
            gamma=gamma_for_platform(platform),
            mu0=NULL,
            lambda0=NULL,
            alpha=ALPHA,
            beta=BETA,
            save.chain=SAVE_CHAIN_TEMPORARILY,
            chain.fname=chain_path
        ), record)
        labels <- as.integer(as.character(fit$spatial.cluster))
        z <- timed_phase("read_mcmc_chain", {
            chain <- BayesSpace::mcmcChain(fit, params="z")
            post_burn_z(as.matrix(chain), ncol(spe_fixed))
        }, record)
        unlink(chain_path)
    }
    if (length(labels) != ncol(spe_fixed)) {
        stop("Cluster-label length mismatch for seed ", seed, ".")
    }
    if (anyNA(labels) || any(labels < 1L | labels > q)) {
        stop("Invalid BayesSpace cluster IDs for seed ", seed, ".")
    }
    soft <- timed_phase("estimate_memberships", {
        posterior_memberships(z, labels, q)
    }, record)
    rm(z)
    membership_path <- file.path(per_run_dir,
                                 paste0("seed_", seed, "_memberships.rds"))
    spot_ids <- colnames(spe_fixed)
    if (is.null(spot_ids) || anyNA(spot_ids) || anyDuplicated(spot_ids)) {
        stop("Cannot write memberships without unique spot IDs.")
    }
    timed_phase("save_memberships", {
        saveRDS(list(
            sample=sample_key, seed=seed, spot_id=spot_ids,
            hard_label=labels, probabilities=soft$probabilities,
            clusters=seq_len(q),
            interpretation=paste(
                "Per-spot relative frequencies of post-burn-in thinned MCMC",
                "labels, each iteration aligned to the run's final hard labels",
                "using optimal maximum-overlap one-to-one label assignment."
            ),
            n_retained=soft$n_retained,
            n_iterations_relabelled=soft$n_iterations_relabelled,
            nrep=NREP, burn_in=BURN_IN, thin=THIN,
            platform=platform,
            slide_seq_k=if (platform == "SlideSeq") SLIDESEQ_K else NA_integer_,
            gamma=gamma_for_platform(platform)
        ), file=membership_path, compress=TRUE)
    }, record)
    completed <- TRUE
    labels
}

output_path_for <- function(spec) {
    out_dir <- file.path(OUT_ROOT, spec$output_group, "bayesspace")
    dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
    if (spec$output_group == "dlpfc") {
        section <- sub("^dlpfc_", "", spec$key)
        file.path(
            out_dir,
            paste0(section, "_bayesspace_results_", ITERATIONS, "iterations.csv")
        )
    } else {
        file.path(
            out_dir,
            paste0("bayesspace_results_", ITERATIONS, "iterations.csv")
        )
    }
}

# Check all 50 membership files and combined outputs before resuming.
completed_artifacts_present <- function(spec) {
    out_path <- output_path_for(spec)
    per_run_dir <- file.path(dirname(out_path), paste0(spec$key, "_per_run"))
    files <- c(
        out_path,
        sub("\\.csv$", "_spot_ids.csv", out_path),
        sub("\\.csv$", "_timings.csv", out_path),
        file.path(per_run_dir, paste0("seed_", SEEDS, "_memberships.rds"))
    )
    if (!all(file.exists(files)) || anyNA(file.info(files)$size) ||
        any(file.info(files)$size <= 0)) return(FALSE)
    columns <- tryCatch(names(read.csv(out_path, nrows=0L,
                                       check.names=FALSE)),
                        error=function(e) NULL)
    identical(columns, paste0("seed_", SEEDS))
}

run_sample <- function(spec) {
    started <- Sys.time()
    tick <- unname(proc.time()[["elapsed"]])
    out_path <- output_path_for(spec)
    per_run_dir <- file.path(dirname(out_path),
                             paste0(spec$key, "_per_run"))
    dir.create(per_run_dir, recursive=TRUE, showWarnings=FALSE)
    unlink(file.path(per_run_dir, paste0("seed_", SEEDS, "_timings.csv")))
    sample_events <- list()
    sample_ok <- FALSE
    record <- function(phase, start, duration, status) {
        sample_events[[length(sample_events) + 1L]] <<- timing_row(
            spec$key, NA_integer_, phase, start, duration, status)
    }
    on.exit({
        record("sample_total", started,
               unname(proc.time()[["elapsed"]]) - tick,
               if (sample_ok) "ok" else "error")
        run_files <- file.path(per_run_dir,
                               paste0("seed_", SEEDS, "_timings.csv"))
        run_files <- run_files[file.exists(run_files)]
        rows <- c(sample_events, lapply(run_files, function(path) {
            read.csv(path, stringsAsFactors=FALSE)
        }))
        timings_path <- sub("\\.csv$", "_timings.csv", out_path)
        write.csv(do.call(rbind, rows), timings_path, row.names=FALSE)
        message("[", spec$key, "] timings saved: ", timings_path)
    }, add=TRUE)
    prepared <- timed_phase("preparation_wall",
                            prepare_sample(spec, record), record)
    message(
        "[", spec$key, "] running ", ITERATIONS,
        " independent BayesSpace MCMC chains with ", N_JOBS, " workers"
    )
    results <- timed_phase("parallel_runs_wall", parallel::mclapply(
        SEEDS,
        FUN=run_one_seed,
        spe_fixed=prepared$spe,
        q=prepared$target,
        platform=spec$platform,
        sample_key=spec$key,
        per_run_dir=per_run_dir,
        slide_neighbors=prepared$slide_neighbors,
        mc.cores=N_JOBS,
        mc.preschedule=FALSE,
        mc.set.seed=FALSE
    ), record)
    failed <- vapply(results, inherits, logical(1), what="try-error")
    if (any(failed)) {
        stop(
            "BayesSpace failed for seeds: ",
            paste(SEEDS[failed], collapse=", ")
        )
    }
    results_mat <- do.call(cbind, results)
    colnames(results_mat) <- paste0("seed_", SEEDS)
    spot_ids <- colnames(prepared$spe)
    if (is.null(spot_ids) || length(spot_ids) != nrow(results_mat) ||
        anyNA(spot_ids) || anyDuplicated(spot_ids)) {
        stop("Missing or duplicated spot IDs: cannot document CSV row alignment.")
    }
    timed_phase("save_hard_labels", {
        write.csv(results_mat, out_path, row.names=FALSE)
        write.csv(data.frame(spot_id=spot_ids),
                  sub("\\.csv$", "_spot_ids.csv", out_path), row.names=FALSE)
    }, record)
    elapsed <- as.numeric(difftime(Sys.time(), started, units="secs"))
    message(
        "[", spec$key, "] saved ", out_path,
        " (", nrow(results_mat), " spots x ", ncol(results_mat),
        " seeds) in ", sprintf("%.1f", elapsed), " s"
    )
    sample_ok <- TRUE
    rm(results, results_mat, prepared)
    gc(verbose=FALSE)
    invisible(out_path)
}

validate_bayesspace_api <- function() {
    preprocess_args <- names(formals(BayesSpace::spatialPreprocess))
    cluster_args <- names(formals(BayesSpace::spatialCluster))
    internal_args <- names(formals(BayesSpace:::cluster))
    if (!all(c("BSPARAM", "BPPARAM") %in% preprocess_args) ||
        !"thin" %in% cluster_args || !"thin" %in% internal_args ||
        !"VisiumHD" %in% eval(formals(BayesSpace::spatialCluster)$platform)) {
        stop("Installed BayesSpace API does not support this runner's ",
             "VisiumHD/thin/BSPARAM/BPPARAM arguments. Check packageVersion('BayesSpace').")
    }
    invisible(TRUE)
}

main <- function() {
    validate_bayesspace_api()
    cat("BayesSpace fixed-preprocessing seed experiment\n")
    cat("  BayesSpace version:", as.character(packageVersion("BayesSpace")), "\n")
    cat("  HVGs:", N_HVGS, "[BayesSpace default/paper]\n")
    cat("  PCs:", N_PCS, "[BayesSpace default/paper]\n")
    cat("  log.normalize:", LOG_NORMALIZE, "[BayesSpace default]\n")
    cat("  init.method:", INIT_METHOD, "[BayesSpace default]\n")
    cat("  model:", ERROR_MODEL, "[BayesSpace default]\n")
    cat("  precision:", PRECISION_MODEL, "[BayesSpace default]\n")
    cat("  nrep:", NREP, "[BayesSpace default]\n")
    cat("  burn.in:", BURN_IN, "[BayesSpace default]\n")
    cat("  thin:", THIN, "[BayesSpace default]\n")
    cat("  alpha:", ALPHA, "[BayesSpace default]\n")
    cat("  beta:", BETA, "[BayesSpace default]\n")
    cat("  gamma: 3 for Visium; 2 for VisiumHD [native BayesSpace defaults]\n")
    cat("  VisiumHD neighborhood: 4 cardinal square-bin neighbors [native default]\n")
    cat("  Slide-seq adaptation: symmetric exact Euclidean kNN, k=", SLIDESEQ_K,
        "; gamma=", SLIDESEQ_GAMMA, " [project adaptation; not BayesSpace default]\n", sep="")
    cat("  preprocessing seed:", PREPROCESS_SEED, "[fixed]\n")
    cat("  clustering seeds:", min(SEEDS), "..", max(SEEDS), "[only varying quantity]\n")
    cat("  workers per sample:", N_JOBS, "\n")
    cat("  save.chain:", SAVE_CHAIN_TEMPORARILY,
        "[temporary native chains for soft-membership extraction; deleted per run]\n")
    cat("  memberships: aligned post-burn-in MCMC z frequencies; 'clue' required\n")
    cat("  output: combined hard-label CSV + spot IDs, per-seed membership RDS, per-seed and per-sample timing CSV\n")
    cat("  samples:", nrow(samples), "[18 samples; MBC uses a project Slide-seq adaptation]\n")
    for (i in seq_len(nrow(samples))) {
        spec <- as.list(samples[i, , drop=FALSE])
        if (spec$key %in% COMPLETED_ON_PREVIOUS_RUN &&
            completed_artifacts_present(spec)) {
            message("[", spec$key, "] existing 50-run labels and memberships ",
                    "found; skipping completed sample")
            next
        }
        run_sample(spec)
    }
    cat("\nDone.\n")
}

if (sys.nframe() == 0L) {
    main()
}
