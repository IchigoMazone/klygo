from typing import Any, Dict, Tuple, Union

import cv2 as cv
import numpy as np
from PIL import Image

from .draw_bboxes import draw_bboxes
from .show_image import show_image


def visualize_prediction(
    image: Union[Image.Image, np.ndarray],
    prediction: Dict[str, Any],
    backend: str = "matplotlib",
    box_color: Tuple[int, int, int] = (255, 0, 0),
    text_color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    font: int = cv.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.6,
    figsize: Tuple[int, int] = (10, 10),
) -> None:
    """
    Tác dụng:
    - Vẽ và hiển thị kết quả dự đoán trên ảnh

    Đầu vào:
    - image: Ảnh PIL hoặc mảng NumPy
    - prediction: Kết quả gồm boxes, labels và scores
    - backend: Công cụ hiển thị ảnh
    - box_color: Màu bounding box theo định dạng BGR
    - text_color: Màu chữ theo định dạng BGR
    - thickness: Độ dày của bounding box và chữ
    - font: Font chữ của OpenCV
    - font_scale: Kích thước chữ OpenCV
    - figsize: Kích thước figure của Matplotlib

    Đầu ra:
    - Không trả về dữ liệu

    Nguồn: TrinhNhuNhat_12072026.
    """
    boxes = prediction.get("boxes", [])
    scores = prediction.get("scores", [])
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    if hasattr(scores, "tolist"):
        scores = scores.tolist()

    annotated = draw_bboxes(
        image,
        boxes,
        prediction.get("labels", []),
        scores,
        box_color=box_color,
        text_color=text_color,
        thickness=thickness,
        font=font,
        font_scale=font_scale,
    )
    show_image(
        annotated,
        title="Prediction Result",
        backend=backend,
        figsize=figsize,
    )
