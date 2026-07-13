import re
from pathlib import Path
from typing import List, Union

from ._crop_boxes import Crop


def _save_crops(
    crops: List[Crop],
    target: Union[str, Path],
    stem: str,
) -> List[Path]:
    """
    Tác dụng:
    - Lưu ảnh đã cắt vào các thư mục theo class

    Đầu vào:
    - crops: Danh sách ảnh đã cắt và tên class
    - target: Thư mục đầu ra
    - stem: Tên cơ sở dùng để đặt tên file ảnh

    Đầu ra:
    - Danh sách đường dẫn của các ảnh đã lưu

    Ngoại lệ:
    - ValueError: target là file thay vì thư mục

    Nguồn: TrinhNhuNhat_13072026.
    """
    target = Path(target)
    if target.exists() and not target.is_dir():
        raise ValueError(f"target must be a directory, got file: {target}")
    target.mkdir(parents=True, exist_ok=True)

    counts = {}
    paths: List[Path] = []
    for image, label in crops:
        label_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_")
        label_name = label_name or "unknown"
        index = counts.get(label_name, 0)
        counts[label_name] = index + 1
        label_dir = target / label_name
        label_dir.mkdir(parents=True, exist_ok=True)
        image_path = label_dir / f"{stem}_crop_{index}.jpg"
        image.save(image_path)
        paths.append(image_path)
    return paths
