import sys

from . import archive
from . import config
from . import cuda
from . import datasets
from . import files
from . import media
from . import models
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

__version__ = "2.0.20"
__author__ = "IchigoMazone"

__all__ = [
    "archive",
    "config",
    "cuda",
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
