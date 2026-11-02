import importlib
import sys

from . import archive
from . import config
from . import datasets
from . import files
from . import media
from . import outputs
from . import postprocessing
from . import processing
from . import utils
from . import validators
from . import visual
from .config import Config

# Alias tương thích ngược
visualize = visual
sys.modules["klygo.visualize"] = visual

__version__ = "2.1.0"
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
    "models",
    "outputs",
    "postprocessing",
    "processing",
    "utils",
    "validators",
    "visual",
    "visualize",
    "Config",
]
