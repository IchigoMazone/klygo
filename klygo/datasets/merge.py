"""Merge YOLO datasets while reconciling class IDs."""

from contextlib import ExitStack
from os import PathLike
from pathlib import Path

from klygo import config, files
from klygo.validators.datasets import Merge

from ._structure import (
    copy_entry,
    flattened_name,
    read_class_map,
    remap_label_file,
    scan_dataset_files,
    validate_dataset_format,
)
from ._workspace import dataset_workspace, open_dataset, set_output_metadata_path, write_output


def merge(
    sources: list[str | PathLike[str]],
    output_path: str | PathLike[str],
    format: str = "yolo",
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Merge multiple YOLO datasets into one directory or ZIP archive.

    Class names are combined in first-seen order. Label IDs from each source
    are rewritten to the corresponding global IDs, and source prefixes prevent
    filename collisions.

    Parameters
    ----------
    sources : list[str or os.PathLike[str]]
        Two or more dataset directories or supported archives.
    output_path : str or os.PathLike[str]
        Destination directory, or a ZIP archive when the path ends in .zip.
    format : str, default='yolo'
        Dataset format to merge. Currently only ``'yolo'`` is supported.
    overwrite : bool, default=False
        Replace an existing output dataset.
    verbose : bool, default=True
        Display archive progress.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``format`` is not a string.
    ValueError
        If the format or metadata is invalid, or source and output paths overlap.
    FileNotFoundError
        If a source does not exist.
    FileExistsError
        If the destination exists and overwrite is disabled.

    Examples
    --------
    >>> from klygo import datasets
    >>> datasets.merge(["cars", "buses.zip"], "transport.zip")
    """
    validate_dataset_format(format)
    params = Merge(sources, output_path, overwrite, verbose)
    if len(params.sources) < 2:
        raise ValueError("sources must contain at least two datasets")

    with ExitStack() as stack, dataset_workspace("merge") as workspace:
        roots = [
            stack.enter_context(open_dataset(source, verbose=False))
            for source in params.sources
        ]
        local_maps = [read_class_map(root) for root in roots]
        global_names: list[str] = []
        for class_map in local_maps:
            for name in class_map.values():
                if name not in global_names:
                    global_names.append(name)

        for source_index, (root, names) in enumerate(zip(roots, local_maps)):
            class_map = {
                local_id: global_names.index(name) for local_id, name in names.items()
            }
            pairs = scan_dataset_files(files.join(root, "images"), files.join(root, "labels"))
            for image_path, label_path, relative_image, relative_label in pairs:
                image_name = f"src{source_index}_{flattened_name(relative_image)}"
                label_name = files.with_extension(image_name, ".txt")
                copy_entry(image_path, files.join(workspace, "images", image_name))
                if label_path is not None:
                    remap_label_file(
                        label_path,
                        files.join(workspace, "labels", label_name),
                        class_map,
                    )

        metadata = {
            "train": "images",
            "nc": len(global_names),
            "names": global_names,
        }
        set_output_metadata_path(metadata, params.output_path)
        config.save(files.join(workspace, "data.yaml"), metadata, overwrite=True, verbose=False)
        write_output(
            workspace,
            params.output_path,
            overwrite=params.overwrite,
            verbose=params.verbose,
        )


__all__ = ["merge"]
