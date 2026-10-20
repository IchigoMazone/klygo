"""Post-process normalized Klygo results without coupling to model backends."""

from . import core as _core
from . import detect as _detect
from .core import *
from .detect import *

__all__ = list(dict.fromkeys(_core.__all__ + _detect.__all__))
