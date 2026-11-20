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
    fig = plt.figure(figsize=figsize)
    plt.bar(class_names, list(class_counts.values()), color="royalblue")
    plt.xlabel("Classes")
    plt.ylabel("Object Count")
    plt.title("Class Distribution in Dataset" if title is None else title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return _finalize_figure(
        fig,
        show=show,
        save_path=save_path,
        dpi=dpi,
        return_figure=return_figure,
    )
