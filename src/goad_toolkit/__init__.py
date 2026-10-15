"""GOAD: three ways into an analysis.

- `DataProcessor`: a file in, an analysis-ready frame out, through a `Pipeline`.
- `Pipeline`: a frame in, a transformed frame out, one `TransformBase` step at a time.
- `BasePlot`: a frame in, a figure out, composed from layers, saved with `save(path)`.
- `DistributionFitter`: values in, fitted distributions out, ranked.

Each is imported on first use, so `goad` on the command line does not load matplotlib.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from goad_toolkit.analytics import DistributionFitter
    from goad_toolkit.dataprocessor import DataProcessor
    from goad_toolkit.datatransforms import Pipeline
    from goad_toolkit.visualizer import BasePlot

__all__ = ["DataProcessor", "Pipeline", "BasePlot", "DistributionFitter"]

_HOME = {
    "DataProcessor": "goad_toolkit.dataprocessor",
    "Pipeline": "goad_toolkit.datatransforms",
    "BasePlot": "goad_toolkit.visualizer",
    "DistributionFitter": "goad_toolkit.analytics",
}


def __getattr__(name: str) -> Any:
    if name in _HOME:
        return getattr(import_module(_HOME[name]), name)
    raise AttributeError(f"module 'goad_toolkit' has no attribute {name!r}")


try:
    __version__ = version("goad-toolkit")
except PackageNotFoundError:  # source tree without an installed distribution
    __version__ = "unknown"
