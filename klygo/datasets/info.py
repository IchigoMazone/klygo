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
    verbose: bool = False,
) -> dict[str, Any]:
    """Return structural statistics for a YOLO dataset.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Dataset directory or supported archive.
    format : str, default='yolo'
        Dataset format to inspect. Currently only ``'yolo'`` is supported.
    verbose : bool, default=False
        Print the dataset statistics as a Rich table while returning the dictionary.

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
        result = {
            "classes": list(class_map.values()),
            "class_ids": list(class_map),
            "nc": max(class_map, default=-1) + 1,
            "total_images": len(pairs),
            "total_labels": total_labels,
            "unlabeled_images": len(pairs) - total_labels,
            "splits": split_counts,
        }

        if verbose:
            from rich.console import Console
            from rich.table import Table

            source_name = files.name(source)
            table = Table(title=f"Dataset Information: {source_name}")
            table.add_column("Property", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")
            table.add_row("Classes", ", ".join(result["classes"]) if result["classes"] else "(None)")
            table.add_row("Class Count", str(result["nc"]))
            table.add_row("Total Images", str(result["total_images"]))
            table.add_row("Total Labels", str(result["total_labels"]))
            table.add_row("Unlabeled Images", str(result["unlabeled_images"]))
            for split_name, count in result["splits"].items():
                table.add_row(f"Split ({split_name})", str(count))
            Console().print(table)

        return result


get_dataset_info = info

__all__ = ["info", "get_dataset_info"]
