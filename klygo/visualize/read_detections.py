import io
import json
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Sequence, Union
from zipfile import ZipFile

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


def _detection_record(
    image: Image.Image,
    item: Mapping[str, Any],
    index: int,
) -> Dict[str, Any]:
    boxes_value = item.get("boxes", [])
    scores_value = item.get("scores", [])
    labels_value = item.get("labels", [])

    if hasattr(boxes_value, "tolist"):
        boxes_value = boxes_value.tolist()
    if hasattr(scores_value, "tolist"):
        scores_value = scores_value.tolist()
    if not isinstance(boxes_value, (list, tuple)):
        raise TypeError(f"source[{index}]['boxes'] must be a sequence")
    if not isinstance(scores_value, (list, tuple)):
        raise TypeError(f"source[{index}]['scores'] must be a sequence")
    if not isinstance(labels_value, (list, tuple)):
        raise TypeError(f"source[{index}]['labels'] must be a sequence")

    boxes: List[List[float]] = []
    for box in boxes_value:
        values = box.tolist() if hasattr(box, "tolist") else list(box)
        if len(values) != 4:
            raise ValueError("each detection box must contain four values")
        boxes.append([float(value) for value in values])

    scores = [float(score) for score in scores_value]
    labels = [str(label) for label in labels_value]
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError(
            f"source[{index}] boxes, scores and labels must have equal lengths"
        )

    return {
        "image": image.convert("RGB"),
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }


def _metadata_items(data: Any) -> List[Mapping[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("metadata.json must contain a list")
    if not all(isinstance(item, Mapping) for item in data):
        raise ValueError("metadata.json items must be objects")
    return data


def _safe_relative_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("detection metadata path must be a non-empty str")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe detection metadata path: {value}")
    return path


def read_detections(
    source: Union[str, Path, Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Tác dụng:
    - Đọc kết quả detect từ thư mục, file ZIP hoặc kết quả trực tiếp của Model.detect

    Đầu vào:
    - source: Thư mục, file ZIP hoặc danh sách kết quả của Model.detect với metadata

    Đầu ra:
    - Danh sách kết quả gồm image, boxes, scores và labels của từng ảnh

    Ngoại lệ:
    - TypeError: Kiểu dữ liệu source hoặc metadata không hợp lệ
    - ValueError: Nội dung metadata, bbox hoặc định dạng source không hợp lệ
    - FileNotFoundError: source hoặc ảnh được khai báo trong metadata không tồn tại

    Nguồn: TrinhNhuNhat_13072026.
    """
    records: List[Dict[str, Any]] = []
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        for index, item in enumerate(source):
            if not isinstance(item, Mapping):
                raise TypeError(
                    f"source[{index}] must be a mapping, "
                    f"got {type(item).__name__}"
                )
            image = item.get("image")
            if not isinstance(image, Image.Image):
                raise TypeError(
                    f"source[{index}]['image'] must be PIL.Image.Image"
                )
            records.append(_detection_record(image, item, index))
        return records

    if not isinstance(source, (str, Path)):
        raise TypeError(
            "source must be a ZIP file, directory or Model.detect result, "
            f"got {type(source).__name__}"
        )

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"source does not exist: {source_path}")

    if source_path.is_dir():
        metadata_path = source_path / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as file:
                items = _metadata_items(json.load(file))
            for index, item in enumerate(items):
                relative_path = _safe_relative_path(item.get("path"))
                image_path = source_path.joinpath(*relative_path.parts)
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"detection image does not exist: {image_path}"
                    )
                with Image.open(image_path) as source_image:
                    image = source_image.convert("RGB")
                records.append(_detection_record(image, item, index))
            return records

        image_paths = sorted(
            path
            for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_paths:
            raise ValueError(f"No detection images found in source: {source_path}")
        for index, image_path in enumerate(image_paths):
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            records.append(_detection_record(image, {}, index))
        return records

    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with ZipFile(source_path, mode="r") as zip_file:
            names = zip_file.namelist()
            metadata_names = sorted(
                name
                for name in names
                if PurePosixPath(name).name == "metadata.json"
            )
            if metadata_names:
                metadata_name = metadata_names[0]
                parent = PurePosixPath(metadata_name).parent
                items = _metadata_items(json.loads(zip_file.read(metadata_name)))
                for index, item in enumerate(items):
                    relative_path = _safe_relative_path(item.get("path"))
                    image_name = (parent / relative_path).as_posix()
                    if image_name not in names:
                        raise FileNotFoundError(
                            f"detection image does not exist in ZIP: {image_name}"
                        )
                    image = Image.open(io.BytesIO(zip_file.read(image_name)))
                    image.load()
                    records.append(_detection_record(image, item, index))
                return records

            image_names = sorted(
                name
                for name in names
                if PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES
            )
            if not image_names:
                raise ValueError(f"No detection images found in source: {source_path}")
            for index, image_name in enumerate(image_names):
                image = Image.open(io.BytesIO(zip_file.read(image_name)))
                image.load()
                records.append(_detection_record(image, {}, index))
            return records

    raise ValueError(f"source must be a ZIP file or directory: {source_path}")
