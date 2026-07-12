from typing import List, Tuple, Union

import cv2 as cv
import numpy as np
from PIL import Image


def draw_bboxes(
    image: Union[Image.Image, np.ndarray],
    bboxes: List[List[float] | Tuple[float, float, float, float]],
    labels: List[str] | None = None,
    scores: List[float] | None = None,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    text_color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    font: int = cv.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.6,
) -> Union[Image.Image, np.ndarray]:
    """
    Tác dụng:
    - Vẽ bounding box, nhãn và điểm tin cậy lên ảnh

    Đầu vào:
    - image: Ảnh PIL hoặc mảng NumPy
    - bboxes: Danh sách bounding box theo định dạng x1, y1, x2, y2
    - labels: Danh sách nhãn của các bounding box
    - scores: Danh sách điểm tin cậy của các bounding box
    - box_color: Màu bounding box theo định dạng BGR
    - text_color: Màu chữ theo định dạng BGR
    - thickness: Độ dày nét vẽ
    - font: Font chữ của OpenCV
    - font_scale: Kích thước chữ OpenCV

    Đầu ra:
    - Ảnh đã được vẽ bounding box

    Nguồn: TrinhNhuNhat_12072026.
    """
    is_pil = isinstance(image, Image.Image)
    img_np = (
        cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        if is_pil
        else image.copy()
    )

    for idx, box in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, box)
        cv.rectangle(img_np, (x1, y1), (x2, y2), box_color, thickness)

        text_parts = []
        if labels and idx < len(labels):
            text_parts.append(str(labels[idx]))
        if scores and idx < len(scores):
            text_parts.append(f"{scores[idx]:.2f}")

        if text_parts:
            cv.putText(
                img_np,
                " ".join(text_parts),
                (x1, y1 - 5 if y1 - 5 > 15 else y1 + 18),
                font,
                font_scale,
                text_color,
                thickness,
            )

    if is_pil:
        return Image.fromarray(cv.cvtColor(img_np, cv.COLOR_BGR2RGB))
    return img_np
