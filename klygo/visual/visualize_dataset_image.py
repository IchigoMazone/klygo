from pathlib import Path
from typing import List, Tuple, Union

import cv2 as cv
from PIL import Image

from .draw_bboxes import draw_bboxes
from klygo.utils._read_yolo_boxes import _read_yolo_boxes
from .show_image import show_image


def visualize_dataset_image(
    image_path: Union[str, Path],
    label_path: Union[str, Path],
    classes: List[str],
    backend: str = "matplotlib",
    figsize: Tuple[int, int] = (10, 10),
    box_color: Tuple[int, int, int] = (255, 0, 0),
    text_color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    font: int = cv.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.6,
) -> None:
    """
    Tác dụng:
    - Đọc và hiển thị ảnh dataset cùng các nhãn YOLO

    Đầu vào:
    - image_path: Đường dẫn file ảnh
    - label_path: Đường dẫn file nhãn YOLO
    - classes: Danh sách tên class
    - backend: Công cụ hiển thị ảnh
    - box_color: Màu bounding box theo định dạng BGR
    - text_color: Màu chữ theo định dạng BGR
    - thickness: Độ dày của bounding box và chữ
    - font: Font chữ của OpenCV
    - font_scale: Kích thước chữ OpenCV
    - figsize: Kích thước figure của Matplotlib

    Đầu ra:
    - Không trả về dữ liệu

    Ngoại lệ:
    - ValueError: Đường dẫn trỏ đến thư mục thay vì file
    - FileNotFoundError: File ảnh không tồn tại

    Nguồn: TrinhNhuNhat_12072026.
    """
    image_path = Path(image_path)
    label_path = Path(label_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"image_path must be a file, got directory: {image_path}")
    if label_path.exists() and not label_path.is_file():
        raise ValueError(f"label_path must be a file, got directory: {label_path}")

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    yolo_boxes = _read_yolo_boxes(label_path, image.size, classes)
    bboxes = [box for box, _ in yolo_boxes]
    labels = [label for _, label in yolo_boxes]

    annotated = draw_bboxes(
        image,
        bboxes,
        labels,
        box_color=box_color,
        text_color=text_color,
        thickness=thickness,
        font=font,
        font_scale=font_scale,
    )
    show_image(
        annotated,
        title=f"Dataset Sample: {image_path.name}",
        backend=backend,
        figsize=figsize,
    )
