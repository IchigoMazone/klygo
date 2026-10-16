"""Inspect YOLO dataset structure and annotation coverage."""

from os import PathLike
from pathlib import Path
from typing import Any

from klygo import files

from ._structure import read_class_map, scan_dataset_files
from ._workspace import open_dataset


def get_dataset_info(source: str | PathLike[str]) -> dict[str, Any]:
    """Return structural statistics for a YOLO dataset.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Dataset directory or supported archive.

    Returns
    -------
    dict
        Ordered class names and their numeric IDs, class count, image/label
        totals, unlabeled image count, and counts grouped by ``train``,
        ``val``, ``test``, or ``raw``.

    Raises
    ------
    FileNotFoundError
        If the source or its ``images`` directory is missing.
    ValueError
        If ``data.yaml`` does not define class names.

    Examples
    --------
    >>> from klygo import datasets
    >>> info = datasets.get_dataset_info("dataset")
    >>> info["total_images"] >= info["total_labels"]
    True
    """
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


__all__ = ["get_dataset_info"]
