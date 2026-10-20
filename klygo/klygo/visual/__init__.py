"""
Bộ công cụ Đồ Họa & Trực Quan Hóa Dữ Liệu AI (`klygo.visual`).

Các chức năng cốt lõi:
- `draw_bboxes` (alias: `draw`)      → Vẽ Bounding Box & Badge nhãn 20 màu chuẩn Ultralytics YOLO.
- `show_image` (alias: `show`)        → Hiển thị ảnh thông minh (Tự động nhận diện Notebook / Colab / Desktop).
- `plot_dataset_stats` (alias: `plot_stats`) → Vẽ biểu đồ phân tích thống kê Dataset.
"""

from .draw_bboxes import draw_bboxes
from .show_image import show_image
from .plot_dataset_stats import plot_dataset_stats

# Aliases ngắn gọn
draw = draw_bboxes
show = show_image
plot_stats = plot_dataset_stats

__all__ = [
    "draw_bboxes",
    "show_image",
    "plot_dataset_stats",
    "draw",
    "show",
    "plot_stats",
]
