"""Flatten a partitioned YOLO dataset."""

from os import PathLike
from pathlib import Path

from klygo import config, files

from ._structure import copy_entry, flattened_name, read_class_map, scan_dataset_files, validate_dataset_format
from ._workspace import dataset_workspace, open_dataset, set_output_metadata_path, write_output


def unpartition(
    source: str | PathLike[str],
    target: str | PathLike[str],
    format: str = "yolo",
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Flatten train, validation, and test folders into one dataset.

    Relative path components are joined with underscores to prevent files
    sharing a basename in different splits from overwriting one another.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Partitioned dataset directory or archive.
    target : str or os.PathLike[str]
        Destination directory or ``.zip`` archive.
    format : str, default='yolo'
        Dataset format to unpartition. Currently only ``'yolo'`` is supported.
    overwrite : bool, default=False
        Replace an existing target.
    verbose : bool, default=True
        Display archive progress.

    Returns
    -------
    pathlib.Path
        Created flat dataset path.

    Raises
    ------
    TypeError
        If ``format`` is not a string.
    ValueError
        If ``format`` is not supported.
    FileNotFoundError
        If the source dataset or its images directory is missing.
    FileExistsError
        If the target exists and ``overwrite`` is false.

    Examples
    --------
    >>> from klygo import datasets
    >>> output = datasets.unpartition("partitioned", "flat")
    """
    validate_dataset_format(format)
    destination = files.path(target)
    if files.exists(destination) and not overwrite:
        raise FileExistsError(f"target already exists: {destination}")

    with open_dataset(source, verbose=False) as root, dataset_workspace("unpartition") as workspace:
        pairs = scan_dataset_files(files.join(root, "images"), files.join(root, "labels"))
        for image_path, label_path, relative_image, _ in pairs:
            image_name = flattened_name(relative_image)
            label_name = files.with_extension(image_name, ".txt")
            copy_entry(image_path, files.join(workspace, "images", image_name))
            if label_path is not None:
                copy_entry(label_path, files.join(workspace, "labels", label_name))

        class_map = read_class_map(root)
        dense = set(class_map) == set(range(len(class_map)))
        metadata = {
            "train": "images",
            "nc": max(class_map, default=-1) + 1,
            "names": list(class_map.values()) if dense else class_map,
        }
        set_output_metadata_path(metadata, destination)
        config.save(files.join(workspace, "data.yaml"), metadata, overwrite=True, verbose=False)
        return write_output(workspace, destination, overwrite=overwrite, verbose=verbose)


__all__ = ["unpartition"]
