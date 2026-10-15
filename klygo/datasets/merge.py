"""Merge YOLO datasets while reconciling class IDs."""

from contextlib import ExitStack
from pathlib import Path

from klygo import config, files
from klygo.validators.datasets import Merge

from ._structure import copy_entry, flattened_name, read_class_names, remap_label_file, scan_dataset_files
from ._workspace import dataset_workspace, open_dataset, write_output


def merge(
    sources: list[str | Path],
    output_path: str | Path,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Merge multiple YOLO datasets into one ZIP archive.

    Class names are combined in first-seen order. Label IDs from each source
    are rewritten to the corresponding global IDs, and source prefixes prevent
    filename collisions.

    Parameters
    ----------
    sources : list[str or pathlib.Path]
        Two or more dataset directories or supported archives.
    output_path : str or pathlib.Path
        Destination ``.zip`` archive.
    overwrite : bool, default=False
        Replace an existing archive.
    verbose : bool, default=True
        Display archive progress.

    Returns
    -------
    pathlib.Path
        Created archive path.

    Raises
    ------
    FileNotFoundError
        If a source does not exist.
    FileExistsError
        If the destination exists and overwrite is disabled.
    ValueError
        If the output is not ZIP or a source has invalid metadata.

    Examples
    --------
    >>> from klygo import datasets
    >>> merged = datasets.merge(["cars", "buses.zip"], "transport.zip")
    """
    params = Merge(sources, output_path, overwrite, verbose)
    if len(params.sources) < 2:
        raise ValueError("sources must contain at least two datasets")

    with ExitStack() as stack, dataset_workspace("merge") as workspace:
        roots = [
            stack.enter_context(open_dataset(source, verbose=False))
            for source in params.sources
        ]
        local_names = [read_class_names(root) for root in roots]
        global_names: list[str] = []
        for names in local_names:
            for name in names:
                if name not in global_names:
                    global_names.append(name)

        for source_index, (root, names) in enumerate(zip(roots, local_names)):
            class_map = {
                local_id: global_names.index(name)
                for local_id, name in enumerate(names)
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
            "path": "/content/data",
            "train": "images",
            "nc": len(global_names),
            "names": global_names,
        }
        config.save(files.join(workspace, "data.yaml"), metadata, overwrite=True, verbose=False)
        return write_output(
            workspace,
            params.output_path,
            overwrite=params.overwrite,
            verbose=params.verbose,
        )


__all__ = ["merge"]
