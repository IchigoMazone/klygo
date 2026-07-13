import io
import json
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union
from zipfile import ZipFile

from PIL import Image

from klygo.utils._make_image_grid import _make_image_grid


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _get_crop_label(path: PurePath) -> Union[str, None]:
    if len(path.parts) > 1:
        return path.parts[-2]
    if "_crop_" in path.stem:
        return path.stem.split("_crop_", maxsplit=1)[0]
    return None


def read_crops(
    source: Union[str, Path, Sequence[Mapping[str, Any]]],
    grid: Union[Tuple[int, ...], None] = (5, 5),
    metadata: bool = False,
    label: Union[str, None] = None,
) -> Union[
    Dict[str, List[Image.Image]],
    List[Dict[str, Any]],
    Image.Image,
]:
    """
    Tác dụng:
    - Đọc các ảnh đối tượng đã cắt từ file ZIP, thư mục hoặc kết quả của Model.crop

    Đầu vào:
    - source: File ZIP, thư mục chứa ảnh đã cắt hoặc danh sách kết quả của Model.crop
    - grid: Lưới ảnh theo số cột hoặc số cột và số hàng
    - metadata: Trạng thái trả danh sách gồm image, score và label
    - label: Tên class cần đọc; None để đọc tất cả class

    Đầu ra:
    - Danh sách image, score, label khi metadata là True
    - Dictionary ảnh theo class khi grid là None, ngược lại trả ảnh PIL dạng lưới

    Ngoại lệ:
    - TypeError: source, label hoặc dữ liệu trong kết quả Model.crop không hợp lệ
    - ValueError: source không phải file ZIP, thư mục hoặc kết quả Model.crop hợp lệ
    - FileNotFoundError: source không tồn tại

    Nguồn: TrinhNhuNhat_13072026.
    """
    if label is not None and (not isinstance(label, str) or not label):
        raise TypeError("label must be a non-empty str or None")
    selected_label = label

    result: Dict[str, List[Image.Image]] = {}
    records: List[Dict[str, Any]] = []
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        for index, item in enumerate(source):
            if not isinstance(item, Mapping):
                raise TypeError(
                    f"source[{index}] must be a mapping, "
                    f"got {type(item).__name__}"
                )
            image = item.get("image")
            label = item.get("label")
            if not isinstance(image, Image.Image):
                raise TypeError(
                    f"source[{index}]['image'] must be PIL.Image.Image, "
                    f"got {type(image).__name__}"
                )
            if not isinstance(label, str) or not label:
                raise TypeError(
                    f"source[{index}]['label'] must be a non-empty str"
                )
            result.setdefault(label, []).append(image)
            records.append(
                {
                    "image": image,
                    "score": item.get("score"),
                    "label": label,
                }
            )
    elif isinstance(source, (str, Path)):
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"source does not exist: {source}")
    else:
        raise TypeError(
            "source must be a ZIP file, directory or Model.crop result, "
            f"got {type(source).__name__}"
        )

    if isinstance(source, Path) and source.is_file():
        if source.suffix.lower() != ".zip":
            raise ValueError(f"source file must be ZIP: {source}")
        with ZipFile(source, mode="r") as zip_file:
            metadata_names = [
                name
                for name in zip_file.namelist()
                if PurePosixPath(name).name == "metadata.json"
            ]
            metadata_items: Dict[str, Mapping[str, Any]] = {}
            metadata_parent = PurePosixPath()
            if metadata_names:
                metadata_name = sorted(metadata_names)[0]
                metadata_parent = PurePosixPath(metadata_name).parent
                loaded = json.loads(zip_file.read(metadata_name))
                if not isinstance(loaded, list):
                    raise ValueError("metadata.json must contain a list")
                metadata_items = {
                    str(item.get("path")): item
                    for item in loaded
                    if isinstance(item, Mapping) and item.get("path")
                }
            for name in zip_file.namelist():
                path = PurePosixPath(name)
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                relative_path = (
                    path.relative_to(metadata_parent)
                    if metadata_names and path.is_relative_to(metadata_parent)
                    else path
                )
                item = metadata_items.get(relative_path.as_posix())
                label = item.get("label") if item is not None else None
                label = label or _get_crop_label(path)
                if label is None:
                    continue

                image = Image.open(io.BytesIO(zip_file.read(name)))
                image.load()
                result.setdefault(label, []).append(image)
                records.append(
                    {
                        "image": image,
                        "score": item.get("score") if item is not None else None,
                        "label": label,
                    }
                )
    elif isinstance(source, Path) and source.is_dir():
        metadata_items: Dict[str, Mapping[str, Any]] = {}
        metadata_path = source / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
            if not isinstance(loaded, list):
                raise ValueError("metadata.json must contain a list")
            metadata_items = {
                str(item.get("path")): item
                for item in loaded
                if isinstance(item, Mapping) and item.get("path")
            }
        for image_path in sorted(source.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative_path = image_path.relative_to(source)
            item = metadata_items.get(relative_path.as_posix())
            label = item.get("label") if item is not None else None
            label = label or _get_crop_label(relative_path)
            if label is None:
                continue

            with Image.open(image_path) as source_image:
                image = source_image.copy()
            result.setdefault(label, []).append(image)
            records.append(
                {
                    "image": image,
                    "score": item.get("score") if item is not None else None,
                    "label": label,
                }
            )
    elif isinstance(source, Path):
        raise ValueError(f"source must be a ZIP file or directory: {source}")

    if selected_label is not None:
        result = {
            class_name: images
            for class_name, images in result.items()
            if class_name == selected_label
        }
        records = [
            item for item in records if item["label"] == selected_label
        ]

    if metadata:
        return records
    if grid is not None:
        images = [image for class_images in result.values() for image in class_images]
        return _make_image_grid(images, grid)
    return result
