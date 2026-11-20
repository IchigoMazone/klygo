"""Modular, backend-neutral Python toolkit for end-to-end Computer Vision workflows.

Public Modules (15 packages + Config):
    - ``archive``        : Create, inspect, extract, and transform archives.
    - ``config`` / ``Config``: Functional and stateful configuration management.
    - ``datasets``       : Load, inspect, partition, merge, split, and remap datasets.
    - ``files``          : Path, filesystem, checksum, download, and 14-format I/O helpers.
    - ``media``          : Lazy images (``LazyImage``), ``MediaFrames``, and ``VideoReader``.
    - ``metrics``        : Composable object-detection evaluation metrics.
    - ``mining``         : Dataset analysis, deduplication, clustering, and balancing.
    - ``models``         : Multi-backend model wrappers (Ultralytics, HF, Torchvision, Keras).
    - ``outputs``        : Standardized detection containers (``Box``, ``Detection``, ``Detections``, ``DetectionGroup``).
    - ``postprocessing`` : Composable NMS, thresholding, clipping, and IoU helpers.
    - ``preprocessing``  : Lazy image transforms, quality metrics, and coordinate trace restoration.
    - ``utils``          : Mapping helpers, binary size formatting, table rendering, and ``ProgressBar``.
    - ``validators``     : Standardized runtime type and count validation helpers.
    - ``graphics``       : Non-destructive box rendering, image display, grids, and report charts.
    - ``reports``        : Experiment log loading, multi-seed aggregation, and audit tables.
"""

import importlib

from . import archive
from . import config
from . import datasets
from . import files
from . import media
from . import metrics
from . import outputs
from . import postprocessing
from . import preprocessing
from . import mining
from . import reports
from . import utils
from . import validators
from . import graphics
from .config import Config

__version__ = "2.1.2"
__author__ = "IchigoMazone"

_LAZY_MODULES = {"models"}


def __getattr__(name: str):
    """Load optional high-level packages only when they are requested."""
    if name in _LAZY_MODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY_MODULES)

__all__ = [
    "archive",
    "config",
    "datasets",
    "files",
    "media",
    "metrics",
    "mining",
    "models",
    "outputs",
    "postprocessing",
    "preprocessing",
    "reports",
    "utils",
    "validators",
    "graphics",
    "Config",
]
