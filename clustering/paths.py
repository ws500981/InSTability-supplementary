"""Resolve input and output paths relative to the repository."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def repository_path(variable, default):
    """Resolve an environment override or a repository-relative default."""
    path = Path(os.environ.get(variable, default)).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


DATA_ROOT = repository_path("INSTABILITY_DATA_ROOT", "data")
OUTPUT_ROOT = repository_path("INSTABILITY_OUTPUT_ROOT", "results")
