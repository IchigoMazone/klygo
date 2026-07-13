from pathlib import Path
from typing import List, Union

import cv2 as cv
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def read_images(
    source: Union[str, Path],
    recursive: bool = False,
    backend: str = "pil",
) -> List[Union[Image.Image, np.ndarray]]:
    """
    Tác dụng:
    - Đọc một file ảnh hoặc toàn bộ ảnh trong thư mục

    Đầu vào:
    - source: File ảnh hoặc thư mục chứa ảnh
    - recursive: Trạng thái cho phép đọc ảnh trong các thư mục con
    - backend: Kiểu ảnh đầu ra, gồm pil hoặc opencv

    Đầu ra:
    - Danh sách ảnh PIL RGB hoặc mảng NumPy BGR theo backend

    Ngoại lệ:
    - TypeError: Kiểu dữ liệu của tham số không hợp lệ
    - ValueError: backend, định dạng ảnh hoặc nội dung source không hợp lệ
    - FileNotFoundError: source không tồn tại

    Nguồn: TrinhNhuNhat_13072026.
    """
    if not isinstance(source, (str, Path)):
        raise TypeError("source must be str or Path")
    if not isinstance(recursive, bool):
        raise TypeError("recursive must be bool")
    if not isinstance(backend, str):
        raise TypeError("backend must be str")

    backend = backend.lower()
    if backend not in ("pil", "opencv"):
        raise ValueError("backend must be 'pil' or 'opencv'")

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"source does not exist: {source}")

    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"source is not a supported image file: {source}")
        image_paths = [source]
    elif source.is_dir():
        iterator = source.rglob("*") if recursive else source.glob("*")
        image_paths = sorted(
            (
                path
                for path in iterator
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda path: str(path).lower(),
        )
        if not image_paths:
            raise ValueError(f"No supported image files found in source: {source}")
    else:
        raise ValueError(f"source must be a file or directory: {source}")

    images: List[Union[Image.Image, np.ndarray]] = []
    for image_path in image_paths:
        if backend == "pil":
            with Image.open(image_path) as source_image:
                images.append(source_image.convert("RGB"))
        else:
            image = cv.imread(str(image_path), cv.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            images.append(image)

    return images
