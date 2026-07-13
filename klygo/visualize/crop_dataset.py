import shutil
import tempfile
from pathlib import Path
from typing import List, Union

from PIL import Image

from klygo.archive import extract
from klygo.utils.dataset import (
    _find_dataset_root,
    _read_class_names,
    _scan_dataset_files,
)
from klygo.utils._crop_boxes import _crop_boxes
from klygo.utils._read_yolo_boxes import _read_yolo_boxes
from klygo.utils._save_crops import _save_crops


def crop_dataset(
    source: Union[str, Path],
    target: Union[str, Path, None] = None,
    padding: int = 0,
) -> List[Image.Image]:
    """
    Tác dụng:
    - Cắt toàn bộ đối tượng từ dataset YOLO dạng phẳng hoặc train/val/test

    Đầu vào:
    - source: File ZIP hoặc thư mục dataset YOLO
    - target: Thư mục lưu ảnh crop; None để chỉ trả ảnh PIL
    - padding: Số pixel mở rộng xung quanh bounding box

    Đầu ra:
    - Danh sách ảnh PIL của tất cả đối tượng đã cắt

    Ngoại lệ:
    - TypeError: Kiểu dữ liệu của tham số không hợp lệ
    - ValueError: Dataset hoặc padding không hợp lệ
    - FileNotFoundError: Source hoặc thư mục images không tồn tại

    Nguồn: TrinhNhuNhat_12072026.
    """
    if not isinstance(source, (str, Path)):
        raise TypeError(f"source must be str or Path, got {type(source).__name__}")
    if not isinstance(padding, int):
        raise TypeError(f"padding must be int, got {type(padding).__name__}")
    if padding < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"source does not exist: {source}")

    target_dir = Path(target) if target is not None else None
    if target_dir is not None:
        if target_dir.exists() and not target_dir.is_dir():
            raise ValueError(f"target must be a directory, got file: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = None
    try:
        if source.is_file():
            temp_dir = Path(tempfile.mkdtemp())
            extract(source, temp_dir, overwrite=True, verbose=False)
            dataset_dir = _find_dataset_root(temp_dir)
        else:
            dataset_dir = _find_dataset_root(source)

        classes = _read_class_names(dataset_dir)
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        if not images_dir.exists() or not images_dir.is_dir():
            raise FileNotFoundError(f"images directory not found: {images_dir}")

        crops: List[Image.Image] = []
        for image_path, label_path, relative_image, _ in _scan_dataset_files(
            images_dir,
            labels_dir,
        ):
            if label_path is None:
                continue

            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            boxes = _read_yolo_boxes(
                label_path,
                image.size,
                classes,
                padding,
            )
            image_crops = _crop_boxes(image, boxes)
            crops.extend(crop for crop, _ in image_crops)
            if target_dir is not None:
                image_name = "_".join(relative_image.with_suffix("").parts)
                _save_crops(image_crops, target_dir, image_name)

        return crops
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir)
