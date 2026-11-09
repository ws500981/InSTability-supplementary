"""Load the external SpiceMix implementation without changing its source files."""

import importlib
import sys
from pathlib import Path

from paths import repository_path

SOURCE_DIR = repository_path("SPICEMIX_SOURCE_DIR", "not-included/SpiceMix/SpiceMix")
MODULES = (
    "model",
    "util",
    "load_data",
    "initialization",
    "estimate_parameters",
    "estimate_weights",
    "sample_for_integral",
)

missing = [
    name + ".py" for name in MODULES if not (SOURCE_DIR / (name + ".py")).is_file()
]
if missing:
    raise ImportError(
        "SpiceMix source is not available at " + str(SOURCE_DIR) + ". "
        "Clone the pinned upstream version described in clustering/THIRD_PARTY.md, "
        "or set SPICEMIX_SOURCE_DIR to its SpiceMix/ source subdirectory. "
        "Missing: " + ", ".join(missing)
    )

for name in MODULES:
    loaded = sys.modules.get(name)
    if loaded is not None:
        origin = getattr(loaded, "__file__", None)
        if origin is None or Path(origin).resolve().parent != SOURCE_DIR:
            raise ImportError(
                f"Module {name!r} is already loaded from another location; use a fresh Python process."
            )

sys.path.insert(0, str(SOURCE_DIR))
_load_data = importlib.import_module("load_data")


class _NumpyCompatibility:
    """Restore removed dtype aliases only for the upstream input loader."""

    float = float
    int = int

    def __init__(self, numpy_module):
        self._numpy = numpy_module

    def __getattr__(self, name):
        return getattr(self._numpy, name)


_load_data.np = _NumpyCompatibility(_load_data.np)
_model = importlib.import_module("model")
_util = importlib.import_module("util")

SpiceMix = _model.SpiceMix
config_logger = _util.config_logger
smooth_continuous = _util.smooth_continuous
smooth_discrete = _util.smooth_discrete
