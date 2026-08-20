"""
Bộ công cụ Quản lý & Nạp Mô hình AI Nhận diện Đối tượng (`klygo.models`).

Hướng dẫn sử dụng:
- Nạp mô hình từ registry mẫu: `models.load("grounding-dino-tiny")`
- Nạp mô hình từ thư mục tối ưu đã xuất: `models.load("./my_exported_model")`
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", message=".*The key `labels` is will return integer ids.*")
warnings.filterwarnings("ignore", message=".*text_labels.*")

from .load import load

__all__ = ["load"]
