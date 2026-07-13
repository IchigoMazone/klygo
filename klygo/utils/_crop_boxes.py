from typing import List, Tuple

from PIL import Image

from ._read_yolo_boxes import YoloBox


Crop = Tuple[Image.Image, str]


def _crop_boxes(image: Image.Image, boxes: List[YoloBox]) -> List[Crop]:
    """
    Tác dụng:
    - Cắt các bounding box khỏi ảnh

    Đầu vào:
    - image: Ảnh PIL đầu vào
    - boxes: Danh sách bounding box và tên class

    Đầu ra:
    - Danh sách ảnh đã cắt và tên class tương ứng

    Nguồn: TrinhNhuNhat_13072026.
    """
    return [(image.crop(box), label) for box, label in boxes]
