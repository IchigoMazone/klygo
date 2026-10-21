"""Rename classes or change their numeric IDs."""

from os import PathLike
from pathlib import Path

from klygo import config, files
from klygo.validators.datasets import RemapClasses

from ._structure import copy_entry, read_class_map, remap_label_file, scan_dataset_files, validate_dataset_format
from ._workspace import dataset_workspace, open_dataset, set_output_metadata_path, write_output


def remap_classes(
    source: str | PathLike[str],
    target: str | PathLike[str],
    class_map: dict[int | str, int | str],
    format: str = "yolo",
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Rename classes and/or remap YOLO class IDs.

    String keys select a class by name; integer keys select by current ID.
    String values rename a class and integer values change its output ID. One
    mapping operation performs either action for each key.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Dataset directory or supported archive.
    target : str or os.PathLike[str]
        Destination directory or ``.zip`` archive.
    class_map : dict[int or str, int or str]
        Class-name and class-ID transformations.
    format : str, default='yolo'
        Dataset format to remap. Currently only ``'yolo'`` is supported.
    overwrite : bool, default=False
        Replace an existing target.
    verbose : bool, default=True
        Display archive progress.

    Returns
    -------
    pathlib.Path
        Created dataset path.

    Raises
    ------
    TypeError
        If ``format`` is not a string, or mapping keys or values are not integers or strings.
    ValueError
        If ``format`` is not supported, a requested class does not exist, or an ID is out of range.

    Examples
    --------
    >>> from klygo import datasets
    >>> output = datasets.remap_classes(
    ...     "dataset", "renamed", {"car": "vehicle"}
    ... )
    """
    validate_dataset_format(format)
    params = RemapClasses(source, target, class_map, overwrite, verbose)
    destination = files.path(params.target)

    with open_dataset(params.source, verbose=False) as root, dataset_workspace("remap") as workspace:
        names = read_class_map(root)
        id_map = {index: index for index in names}
        renamed = dict(names)

        for key, value in params.class_map.items():
            if isinstance(key, str):
                if key not in names.values():
                    raise ValueError(f"Class name {key!r} not found in source dataset.")
                old_id = next(index for index, name in names.items() if name == key)
            elif isinstance(key, int) and not isinstance(key, bool):
                if key not in names:
                    raise ValueError(f"Class ID {key} out of range of source dataset classes.")
                old_id = key
            else:
                raise TypeError("class_map keys must be int or str")

            if isinstance(value, str):
                renamed[old_id] = value
            elif isinstance(value, int) and not isinstance(value, bool):
                if value < 0:
                    raise ValueError("destination class IDs must be non-negative")
                id_map[old_id] = value
            else:
                raise TypeError("class_map values must be int or str")

        output_names: dict[int, str] = {}
        for old_id, new_id in id_map.items():
            if new_id in output_names and output_names[new_id] != renamed[old_id]:
                raise ValueError(f"multiple class names map to destination ID {new_id}")
            output_names[new_id] = renamed[old_id]

        for image_path, label_path, relative_image, relative_label in scan_dataset_files(
            files.join(root, "images"), files.join(root, "labels")
        ):
            copy_entry(image_path, files.join(workspace, "images", relative_image))
            if label_path is not None:
                remap_label_file(label_path, files.join(workspace, "labels", relative_label), id_map)

        contiguous = set(output_names) == set(range(len(output_names)))
        yaml_names = [output_names[index] for index in range(len(output_names))] if contiguous else output_names
        source_config = config.load(files.join(root, "data.yaml"), verbose=False).to_dict()
        source_config.update(
            {
                "nc": max(output_names, default=-1) + 1,
                "names": yaml_names,
            }
        )
        set_output_metadata_path(source_config, destination)
        config.save(files.join(workspace, "data.yaml"), source_config, overwrite=True, verbose=False)
        return write_output(workspace, destination, overwrite=params.overwrite, verbose=params.verbose)


__all__ = ["remap_classes"]
