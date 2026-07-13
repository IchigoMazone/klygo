import shutil
import tempfile
from pathlib import Path
from typing import Tuple, Union

from klygo.archive import extract
from klygo.datasets import get_dataset_info
from klygo.utils.dataset import _find_dataset_root, _scan_dataset_files


def plot_dataset_stats(
    source: Union[str, Path],
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Tác dụng:
    - Vẽ biểu đồ số lượng đối tượng theo class của dataset

    Đầu vào:
    - source: File ZIP hoặc thư mục dataset
    - figsize: Kích thước figure của Matplotlib

    Đầu ra:
    - Không trả về dữ liệu

    Nguồn: TrinhNhuNhat_12072026.
    """
    info = get_dataset_info(source)
    classes = info.get("classes", [])
    source = Path(source)
    temp_dir = None

    if source.is_file():
        temp_dir = Path(tempfile.mkdtemp())
        extract(source, temp_dir, overwrite=True, verbose=False)
        source_dir = _find_dataset_root(temp_dir)
    else:
        source_dir = source

    class_counts = {idx: 0 for idx in range(len(classes))}
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"

    if images_dir.exists() and labels_dir.exists():
        for _, label_path, _, _ in _scan_dataset_files(images_dir, labels_dir):
            if not label_path or not label_path.exists():
                continue
            with open(label_path, "r", encoding="utf-8") as file:
                for line in file:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        class_id = int(parts[0])
                    except ValueError:
                        continue
                    if class_id in class_counts:
                        class_counts[class_id] += 1

    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir)

    import matplotlib.pyplot as plt

    class_names = [classes[idx] for idx in class_counts]
    plt.figure(figsize=figsize)
    plt.bar(class_names, list(class_counts.values()), color="royalblue")
    plt.xlabel("Classes")
    plt.ylabel("Object Count")
    plt.title("Class Distribution in Dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
