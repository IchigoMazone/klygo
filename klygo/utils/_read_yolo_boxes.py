from pathlib import Path
from typing import List, Tuple, Union


YoloBox = Tuple[Tuple[int, int, int, int], str]


def _read_yolo_boxes(
    label_path: Union[str, Path],
    image_size: Tuple[int, int],
    classes: List[str],
    padding: int = 0,
) -> List[YoloBox]:
    """
    Tác dụng:
    - Đọc nhãn YOLO và chuyển bounding box sang tọa độ pixel

    Đầu vào:
    - label_path: Đường dẫn file nhãn YOLO
    - image_size: Kích thước ảnh theo thứ tự width, height
    - classes: Danh sách tên class
    - padding: Số pixel mở rộng xung quanh bounding box

    Đầu ra:
    - Danh sách bounding box và tên class tương ứng

    Ngoại lệ:
    - ValueError: padding âm hoặc label_path là thư mục
    Nguồn: TrinhNhuNhat_13072026.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return []
    if not label_path.is_file():
        raise ValueError(f"label_path must be a file, got directory: {label_path}")
    if padding < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")

    width, height = image_size
    boxes: List[YoloBox] = []
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

            label = (
                str(classes[class_id])
                if 0 <= class_id < len(classes)
                else f"class_{class_id}"
            )
            boxes.append(((x1, y1, x2, y2), label))

    return boxes
