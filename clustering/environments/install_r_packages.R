#!/usr/bin/env Rscript
options(repos=c(CRAN="https://cloud.r-project.org"))
lib <- .Library
args <- commandArgs(trailingOnly=TRUE)
if (length(setdiff(args, "--bayesspace"))) stop("Usage: install_r_packages.R [--bayesspace]")

if ("--bayesspace" %in% args) {
    if (getRversion() < "4.5" || getRversion() >= "4.6") {
        stop("BayesSpace setup requires R 4.5.x; the recorded package build used R 4.5.2.")
    }
    if (!requireNamespace("BiocManager", quietly=TRUE)) {
        install.packages("BiocManager", lib=lib)
    }
    BiocManager::install(version="3.22", lib=lib, ask=FALSE, update=FALSE)
    BiocManager::install(c(
        "BayesSpace", "zellkonverter", "SingleCellExperiment",
        "SummarizedExperiment", "S4Vectors", "BiocSingular",
        "BiocParallel", "BiocNeighbors", "clue"
    ), lib=lib, ask=FALSE, update=FALSE)
    installed <- tryCatch(as.character(packageVersion("BayesSpace", lib.loc=lib)),
                          error=function(e) NA_character_)
    if (!identical(installed, "1.20.2")) {
        install.packages(
            "https://bioconductor.posit.co/packages/3.22/bioc/src/contrib/BayesSpace_1.20.2.tar.gz",
            repos=NULL, type="source", lib=lib
        )
    }
    stopifnot(as.character(packageVersion("BayesSpace", lib.loc=lib)) == "1.20.2")
}

if (!requireNamespace("mclust", quietly=TRUE, lib.loc=lib) ||
    as.character(packageVersion("mclust", lib.loc=lib)) != "6.1.2") {
    install.packages(
        "https://cran.r-project.org/src/contrib/Archive/mclust/mclust_6.1.2.tar.gz",
        repos=NULL, type="source", lib=lib
    )
}
stopifnot(as.character(packageVersion("mclust", lib.loc=lib)) == "6.1.2")
cat("R:", as.character(getRversion()), " mclust:",
    as.character(packageVersion("mclust", lib.loc=lib)), "\n")
