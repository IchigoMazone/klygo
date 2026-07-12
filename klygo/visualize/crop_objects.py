import io
from pathlib import Path
from typing import Dict, List, Union
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image


def crop_objects(
    image_path: Union[str, Path],
    label_path: Union[str, Path],
    classes: List[str],
    output_path: Union[str, Path],
    overwrite: bool = False,
) -> None:
    """
    Tác dụng:
    - Cắt các đối tượng theo nhãn YOLO và lưu vào file ZIP

    Đầu vào:
    - image_path: Đường dẫn file ảnh
    - label_path: Đường dẫn file nhãn YOLO
    - classes: Danh sách tên class
    - output_path: Đường dẫn file ZIP đầu ra
    - overwrite: Trạng thái cho phép ghi đè

    Đầu ra:
    - Không trả về dữ liệu

    Ngoại lệ:
    - ValueError: Đường dẫn đầu vào không phải file
    - FileNotFoundError: File ảnh không tồn tại
    - FileExistsError: File đầu ra đã tồn tại và không cho phép ghi đè

    Nguồn: TrinhNhuNhat_12072026.
    """
    image_path = Path(image_path)
    label_path = Path(label_path)
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"ZIP file already exists: {output_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"image_path must be a file, got directory: {image_path}")
    if label_path.exists() and not label_path.is_file():
        raise ValueError(f"label_path must be a file, got directory: {label_path}")

    image = Image.open(image_path)
    width, height = image.size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crops: Dict[str, List[Image.Image]] = {}

    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                box_w = float(parts[3]) * width
                box_h = float(parts[4]) * height
                x1 = max(0, int(x_center - box_w / 2))
                y1 = max(0, int(y_center - box_h / 2))
                x2 = min(width, int(x_center + box_w / 2))
                y2 = min(height, int(y_center + box_h / 2))
                if x2 <= x1 or y2 <= y1:
                    continue
                class_name = (
                    classes[class_id]
                    if class_id < len(classes)
                    else f"class_{class_id}"
                )
                crops.setdefault(class_name, []).append(
                    image.crop((x1, y1, x2, y2))
                )

    with ZipFile(output_path, mode="w", compression=ZIP_DEFLATED) as zip_file:
        for class_name, images in crops.items():
            for idx, cropped_image in enumerate(images):
                buffer = io.BytesIO()
                cropped_image.save(buffer, format="JPEG")
                zip_file.writestr(
                    f"{class_name}_crop_{idx}.jpg",
                    buffer.getvalue(),
                )
