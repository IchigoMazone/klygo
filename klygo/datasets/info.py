"""Inspect YOLO dataset structure and annotation coverage."""

from os import PathLike
from pathlib import Path
from typing import Any

from klygo import files

from ._structure import read_class_map, scan_dataset_files, validate_dataset_format
from ._workspace import open_dataset


def info(
    source: str | PathLike[str],
    format: str = "yolo",
) -> dict[str, Any]:
    """Return structural statistics for a YOLO dataset.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Dataset directory or supported archive.
    format : str, default='yolo'
        Dataset format to inspect. Currently only ``'yolo'`` is supported.

    Returns
    -------
    dict
        Ordered class names and their numeric IDs, class count, image/label
        totals, unlabeled image count, and counts grouped by ``train``,
        ``val``, ``test``, or ``raw``.

    Raises
    ------
    TypeError
        If ``format`` is not a string.
    ValueError
        If ``format`` is not supported or ``data.yaml`` does not define class names.
    FileNotFoundError
        If the source or its ``images`` directory is missing.

    Examples
    --------
    >>> from klygo import datasets
    >>> stats = datasets.info("dataset")
    >>> stats["total_images"] >= stats["total_labels"]
    True
    """
    validate_dataset_format(format)
    with open_dataset(source, verbose=False) as root:
        class_map = read_class_map(root)
        pairs = scan_dataset_files(files.join(root, "images"), files.join(root, "labels"))
        split_counts: dict[str, int] = {}
        for _, _, relative_image, _ in pairs:
            split = relative_image.parts[0] if len(relative_image.parts) > 1 else "raw"
            if split not in {"train", "val", "test"}:
                split = "raw"
            split_counts[split] = split_counts.get(split, 0) + 1

        total_labels = sum(label is not None for _, label, _, _ in pairs)
        return {
            "classes": list(class_map.values()),
            "class_ids": list(class_map),
            "nc": max(class_map, default=-1) + 1,
            "total_images": len(pairs),
            "total_labels": total_labels,
            "unlabeled_images": len(pairs) - total_labels,
            "splits": split_counts,
        }


__all__ = ["info"]
