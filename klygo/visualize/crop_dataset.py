import shutil
import tempfile
from pathlib import Path
from typing import List, Union

from PIL import Image

from klygo.archive import extract
from klygo.datasets._utils import (
    _find_dataset_root,
    _read_class_names,
    _scan_dataset_files,
)


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
        crop_counts = {}
        for image_path, label_path, relative_image, _ in _scan_dataset_files(
            images_dir,
            labels_dir,
        ):
            if label_path is None:
                continue

            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            with open(label_path, "r", encoding="utf-8") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    box_width = float(parts[3]) * width
                    box_height = float(parts[4]) * height
                    x1 = max(0, int(x_center - box_width / 2) - padding)
                    y1 = max(0, int(y_center - box_height / 2) - padding)
                    x2 = min(width, int(x_center + box_width / 2) + padding)
                    y2 = min(height, int(y_center + box_height / 2) + padding)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop_image = image.crop((x1, y1, x2, y2))
                    crops.append(crop_image)
                    if target_dir is not None:
                        label = (
                            str(classes[class_id])
                            if class_id < len(classes)
                            else f"class_{class_id}"
                        )
                        label_dir = target_dir / label
                        label_dir.mkdir(parents=True, exist_ok=True)
                        key = (str(relative_image), class_id)
                        crop_index = crop_counts.get(key, 0)
                        crop_counts[key] = crop_index + 1
                        image_name = "_".join(
                            relative_image.with_suffix("").parts
                        )
                        crop_image.save(
                            label_dir / f"{image_name}_crop_{crop_index}.jpg"
                        )

        return crops
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir)
