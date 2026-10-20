"""
Hệ thống kết quả đầu ra chuẩn hóa của Klygo (`klygo.outputs`).

Các module theo từng tác vụ thị giác máy tính:
- `klygo.outputs.detect`  : Nhận diện đối tượng (Box, Crops, Detection, Detections)
- `klygo.outputs.segment` : Phân vùng đối tượng (Mask, Crops, Segmentation, Segmentations) [tương lai]
- `klygo.outputs.classify`: Phân loại ảnh (Class, TopK, Classification, Classifications) [tương lai]
"""

from . import detect

# Export trực tiếp submodule detect
__all__ = [
    "detect",
]
