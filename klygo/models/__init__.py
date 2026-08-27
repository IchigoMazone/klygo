"""
Bộ công cụ Quản lý & Nạp Mô hình AI Klygo (`klygo.models`).
Kiến trúc 3 tầng: BaseModel -> Task Base Class (Detector) -> Concrete Models.
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", message=".*The key `labels` is will return integer ids.*")
warnings.filterwarnings("ignore", message=".*text_labels.*")

from . import base
from . import errors
from . import utils
from .base import BaseModel, DetectorModel
from .detection.base import Detector
from .load import load, register

__all__ = [
    "load",
    "register",
    "base",
    "errors",
    "utils",
    "BaseModel",
    "Detector",
    "DetectorModel",
]
