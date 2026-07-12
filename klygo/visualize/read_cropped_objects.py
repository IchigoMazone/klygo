import io
from pathlib import Path
from typing import Dict, List, Union
from zipfile import ZipFile

from PIL import Image


def read_cropped_objects(
    zip_path: Union[str, Path],
) -> Dict[str, List[Image.Image]]:
    """
    Tác dụng:
    - Đọc các ảnh đối tượng đã cắt từ file ZIP

    Đầu vào:
    - zip_path: Đường dẫn file ZIP

    Đầu ra:
    - Dictionary ánh xạ tên class với danh sách ảnh đã cắt

    Ngoại lệ:
    - ValueError: Đường dẫn trỏ đến thư mục thay vì file
    - FileNotFoundError: File ZIP không tồn tại

    Nguồn: TrinhNhuNhat_12072026.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    if not zip_path.is_file():
        raise ValueError(f"zip_path must be a file, got directory: {zip_path}")

    result: Dict[str, List[Image.Image]] = {}
    with ZipFile(zip_path, mode="r") as zip_file:
        for name in zip_file.namelist():
            if not name.endswith((".jpg", ".jpeg", ".png")):
                continue
            parts = name.split("_crop_")
            if len(parts) != 2:
                continue
            image = Image.open(io.BytesIO(zip_file.read(name)))
            image.load()
            result.setdefault(parts[0], []).append(image)
    return result
