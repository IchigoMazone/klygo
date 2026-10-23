"""Configure and load model wrappers across optional AI backends."""

from . import utils

utils.suppress_ai_warnings()

from . import base
from . import errors
from .base import BaseModel
from .detection.base import Detector
from .load import load, configure, set_backend, get_backend
from .metadata import details, flags, metadata, priority
from .utils import suppress_warnings, suppress_ai_warnings

__all__ = [
    "base", "BaseModel", "Detector",
    "load", "configure", "set_backend", "get_backend",
    "metadata", "priority", "details", "flags",
    "errors", "utils", "suppress_warnings", "suppress_ai_warnings",
]
