"""
Bộ công cụ Quản lý & Nạp Mô hình AI Klygo (`klygo.models`).
Kiến trúc 3 tầng: BaseModel -> Task Base Class (Detector) -> Concrete Models.
"""

from . import utils

utils.suppress_ai_warnings()

from . import base
from . import errors
from . import utils
from .base import BaseModel, DetectorModel
from .detection.base import Detector
from .load import load, register
from .utils import suppress_warnings, suppress_ai_warnings

__all__ = [
    "load",
    "register",
    "base",
    "errors",
    "utils",
    "BaseModel",
    "Detector",
    "DetectorModel",
    "suppress_warnings",
    "suppress_ai_warnings",
]
