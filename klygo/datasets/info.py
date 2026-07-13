import random
import shutil
from pathlib import Path
from typing import Any

from klygo.archive import extract
from klygo.utils.dataset import (
    _find_dataset_root,
    _read_class_names,
    _scan_dataset_files,
)


def get_dataset_info(source: str | Path) -> dict[str, Any]:
    """
    Tác dụng:
    - Lấy thông tin thống kê của dataset

    Đầu vào:
    - source: File hoặc thư mục đầu vào

    Đầu ra:
    - Kết quả xử lý của hàm

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    is_zip = source.is_file()
    temp_extract_dir = None

    if is_zip:
        temp_extract_dir = source.parent / f".temp_info_extract_{random.randint(1000, 9999)}"
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        extract(
            archive_path=source,
            output_dir=temp_extract_dir,
            overwrite=True,
            verbose=False,
        )
        src_base = _find_dataset_root(temp_extract_dir)
    else:
        src_base = source

    try:
        classes = _read_class_names(src_base, temp_extract_dir)

        images_dir = src_base / "images"
        labels_dir = src_base / "labels"

        if not images_dir.exists():
            raise FileNotFoundError("images directory not found in dataset")

        pairs = _scan_dataset_files(images_dir, labels_dir)

        total_images = len(pairs)
        total_labels = sum(1 for p in pairs if p[1] is not None)
        unlabeled_images = total_images - total_labels

        # Analyze split structure
        split_counts = {}
        for img_path, lbl_path, rel_img, rel_lbl in pairs:
            parts = rel_img.parts
            if len(parts) > 1 and parts[0] in ["train", "val", "test"]:
                split_name = parts[0]
            else:
                split_name = "raw"
            split_counts[split_name] = split_counts.get(split_name, 0) + 1

        info = {
            "classes": classes,
            "nc": len(classes),
            "total_images": total_images,
            "total_labels": total_labels,
            "unlabeled_images": unlabeled_images,
            "splits": split_counts
        }
    finally:
        if temp_extract_dir and temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)

    return info
