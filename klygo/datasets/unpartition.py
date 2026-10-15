"""Flatten a partitioned YOLO dataset."""

from pathlib import Path

from klygo import config, files

from ._structure import copy_entry, flattened_name, read_class_names, scan_dataset_files
from ._workspace import dataset_workspace, open_dataset, write_output


def unpartition(
    source: str | Path,
    target: str | Path,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Flatten train, validation, and test folders into one dataset.

    Relative path components are joined with underscores to prevent files
    sharing a basename in different splits from overwriting one another.

    Parameters
    ----------
    source : str or pathlib.Path
        Partitioned dataset directory or archive.
    target : str or pathlib.Path
        Destination directory or ``.zip`` archive.
    overwrite : bool, default=False
        Replace an existing target.
    verbose : bool, default=True
        Display archive progress.

    Returns
    -------
    pathlib.Path
        Created flat dataset path.

    Examples
    --------
    >>> from klygo import datasets
    >>> output = datasets.unpartition("partitioned", "flat")
    """
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

        names = read_class_names(root)
        metadata = {
            "path": files.resolve(destination).as_posix(),
            "train": "images",
            "nc": len(names),
            "names": names,
        }
        config.save(files.join(workspace, "data.yaml"), metadata, overwrite=True, verbose=False)
        return write_output(workspace, destination, overwrite=overwrite, verbose=verbose)


__all__ = ["unpartition"]
