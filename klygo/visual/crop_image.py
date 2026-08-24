from pathlib import Path
from typing import List, Union

from PIL import Image

from klygo.utils._crop_boxes import _crop_boxes
from klygo.utils._read_yolo_boxes import _read_yolo_boxes
from klygo.utils._save_crops import _save_crops


def crop_image(
    image_path: Union[str, Path],
    label_path: Union[str, Path],
    classes: List[str],
    target: Union[str, Path, None] = None,
    padding: int = 0,
) -> List[Image.Image]:
    """
    Tác dụng:
    - Cắt các đối tượng trong một ảnh theo nhãn YOLO

    Đầu vào:
    - image_path: Đường dẫn file ảnh
    - label_path: Đường dẫn file nhãn YOLO
    - classes: Danh sách tên class
    - target: Thư mục lưu ảnh đã cắt
    - padding: Số pixel mở rộng xung quanh bounding box

    Đầu ra:
    - Danh sách ảnh PIL của các đối tượng đã cắt

    Ngoại lệ:
    - TypeError: Kiểu dữ liệu của tham số không hợp lệ
    - ValueError: Đường dẫn hoặc padding không hợp lệ
    - FileNotFoundError: File ảnh không tồn tại

    Nguồn: TrinhNhuNhat_13072026.
    """
    if not isinstance(image_path, (str, Path)):
        raise TypeError("image_path must be str or Path")
    if not isinstance(label_path, (str, Path)):
        raise TypeError("label_path must be str or Path")
    if not isinstance(classes, list):
        raise TypeError("classes must be a list")
    if not isinstance(padding, int):
        raise TypeError("padding must be int")
    if padding < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"image_path must be a file, got directory: {image_path}")

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    boxes = _read_yolo_boxes(label_path, image.size, classes, padding)
    crops = _crop_boxes(image, boxes)
    if target is not None:
        _save_crops(crops, target, image_path.stem)
    return [crop for crop, _ in crops]
