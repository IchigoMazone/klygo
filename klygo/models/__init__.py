"""
Bộ công cụ Quản lý & Nạp Mô hình AI Nhận diện Đối tượng (`klygo.models`).

Hướng dẫn sử dụng:
- Nạp mô hình từ registry mẫu: `models.load("grounding-dino-tiny")`
- Nạp mô hình từ thư mục tối ưu đã xuất: `models.load("./my_exported_model")`
"""

from .load import load

__all__ = ["load"]
