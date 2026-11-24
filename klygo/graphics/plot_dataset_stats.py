from os import PathLike
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

from klygo.archive import extract
from klygo.datasets import info as get_dataset_info
from klygo import files, validators
from klygo.datasets._structure import find_dataset_root, scan_dataset_files

from ._render import _finalize_figure

__all__ = ["dataset_stats"]


def dataset_stats(
    source: Union[str, Path],
    figsize: Tuple[int, int] = (10, 6),
    *,
    format: str = "yolo",
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[Union[str, PathLike[str]]] = None,
    dpi: int = 300,
    return_figure: bool = False,
):
    """Plot YOLO object counts for each class in a dataset.

    Parameters
    ----------
    source : str or pathlib.Path
        Dataset directory or supported archive.
    figsize : tuple of int
        Matplotlib figure size in inches.
    format : {'yolo'}, default='yolo'
        Dataset annotation format. Currently only ``'yolo'`` is supported.

    Returns
    -------
    None
        Displays a bar chart.

    Notes
    -----
    Counts class IDs found in label files under ``images`` and ``labels``.
    An archive is extracted to a temporary directory before scanning.
    """
    validators.validate_type(format, str, "format")
    if format.lower() != "yolo":
        raise ValueError("format must be 'yolo'")

    info = get_dataset_info(source)
    classes = info.get("classes", [])
    class_ids = info.get("class_ids", list(range(len(classes))))
    source = files.path(source)
    class_counts = {class_id: 0 for class_id in class_ids}
    with tempfile.TemporaryDirectory() as temporary:
        if files.is_file(source):
            extracted = files.path(temporary)
            extract(source, extracted, overwrite=True, verbose=False)
            source_dir = find_dataset_root(extracted)
        else:
            source_dir = source

        images_dir = files.join(source_dir, "images")
        labels_dir = files.join(source_dir, "labels")
        if files.is_dir(images_dir) and files.is_dir(labels_dir):
            for _, label_path, _, _ in scan_dataset_files(images_dir, labels_dir):
                if not label_path or not files.is_file(label_path):
                    continue
                for line in files.load(label_path, as_lines=True, verbose=False):
                    parts = str(line).strip().split()
                    if not parts:
                        continue
                    try:
                        class_id = int(parts[0])
                    except ValueError:
                        continue
                    if class_id in class_counts:
                        class_counts[class_id] += 1

    import matplotlib.pyplot as plt

    class_names = list(classes)
    counts = [class_counts.get(cid, 0) for cid in class_ids]
    fig = plt.figure(figsize=figsize)
    plt.bar(class_names, counts, color="#2E86C1", edgecolor="#1B4F72", alpha=0.9, width=0.7)
    plt.xlabel("Classes", labelpad=8)
    plt.ylabel("Object Count")
    plt.title("Class Distribution in Dataset" if title is None else title, pad=14)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    max_c = max(counts, default=1)
    plt.ylim(0, max_c * 1.15)

    n_cls = len(class_names)
    rot = 75 if n_cls > 35 else (45 if n_cls > 12 else 0)
    ha = "right" if rot > 0 else "center"
    fontsize = max(6, min(9, int(260 / n_cls))) if n_cls > 25 else 9
    plt.xticks(range(n_cls), class_names, rotation=rot, ha=ha, fontsize=fontsize)
    plt.tight_layout()
    return _finalize_figure(
        fig,
        show=show,
        save_path=save_path,
        dpi=dpi,
        return_figure=return_figure,
    )
