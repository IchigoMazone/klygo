"""
Bộ công cụ Quản lý & Nạp Mô hình AI Nhận diện Đối tượng (`klygo.models`).

Hướng dẫn sử dụng:
- Nạp mô hình từ registry mẫu: `models.load("grounding-dino-tiny")`
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", message=".*The key `labels` is will return integer ids.*")
warnings.filterwarnings("ignore", message=".*text_labels.*")

from . import base
from . import utils
from .load import load, register
from .base import Detector, DetectorModel

__all__ = ["load", "register", "base", "utils", "Detector", "DetectorModel"]
