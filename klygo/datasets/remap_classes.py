"""Rename classes or change their numeric IDs."""

from pathlib import Path

from klygo import config, files
from klygo.validators.datasets import RemapClasses

from ._structure import copy_entry, read_class_names, remap_label_file, scan_dataset_files
from ._workspace import dataset_workspace, open_dataset, write_output


def remap_classes(
    source: str | Path,
    target: str | Path,
    class_map: dict[int | str, int | str],
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Rename classes and/or remap YOLO class IDs.

    String keys select a class by name; integer keys select by current ID.
    String values rename a class and integer values change its output ID. One
    mapping operation performs either action for each key.

    Parameters
    ----------
    source : str or pathlib.Path
        Dataset directory or supported archive.
    target : str or pathlib.Path
        Destination directory or ``.zip`` archive.
    class_map : dict[int or str, int or str]
        Class-name and class-ID transformations.
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
    ValueError
        If a requested class does not exist or an ID is out of range.
    TypeError
        If mapping keys or values are not integers or strings.

    Examples
    --------
    >>> from klygo import datasets
    >>> output = datasets.remap_classes(
    ...     "dataset", "renamed", {"car": "vehicle"}
    ... )
    """
    params = RemapClasses(source, target, class_map, overwrite, verbose)
    destination = files.path(params.target)

    with open_dataset(params.source, verbose=False) as root, dataset_workspace("remap") as workspace:
        names = read_class_names(root)
        id_map = {index: index for index in range(len(names))}
        renamed = {index: name for index, name in enumerate(names)}

        for key, value in params.class_map.items():
            if isinstance(key, str):
                if key not in names:
                    raise ValueError(f"Class name {key!r} not found in source dataset.")
                old_id = names.index(key)
            elif isinstance(key, int) and not isinstance(key, bool):
                if key < 0 or key >= len(names):
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

        for image_path, label_path, relative_image, relative_label in scan_dataset_files(
            files.join(root, "images"), files.join(root, "labels")
        ):
            copy_entry(image_path, files.join(workspace, "images", relative_image))
            if label_path is not None:
                remap_label_file(label_path, files.join(workspace, "labels", relative_label), id_map)

        output_names = {id_map[old_id]: renamed[old_id] for old_id in range(len(names))}
        contiguous = set(output_names) == set(range(len(output_names)))
        yaml_names = [output_names[index] for index in range(len(output_names))] if contiguous else output_names
        source_config = config.load(files.join(root, "data.yaml"), verbose=False).to_dict()
        source_config.update(
            {
                "path": "/content/data" if files.extension(destination).lower() == ".zip" else files.resolve(destination).as_posix(),
                "nc": max(output_names, default=-1) + 1,
                "names": yaml_names,
            }
        )
        config.save(files.join(workspace, "data.yaml"), source_config, overwrite=True, verbose=False)
        return write_output(workspace, destination, overwrite=params.overwrite, verbose=params.verbose)


__all__ = ["remap_classes"]
