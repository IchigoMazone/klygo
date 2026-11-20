"""Split one dataset into class-based or ratio-based directories or ZIP archives."""

from __future__ import annotations

import random
import re
from os import PathLike
from pathlib import Path

from klygo import config, files
from klygo.validators.datasets import Split

from ._partitioning import _stratified_split
from ._structure import (
    copy_entry,
    flattened_name,
    read_class_map,
    remap_label_file,
    scan_dataset_files,
    validate_dataset_format,
)
from ._workspace import dataset_workspace, open_dataset, write_output


def _default_name(group: list[str], index: int) -> str:
    if len(group) != 1:
        return f"split_group_{index}"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", group[0]).strip("._") or f"class_{index}"
    return f"split_{safe_name}"


def _write_split_metadata(workspace: Path, names: list[str] | dict[int, str]) -> None:
    class_count = max(names, default=-1) + 1 if isinstance(names, dict) else len(names)
    config.save(
        files.join(workspace, "data.yaml"),
        {
            "train": "images",
            "nc": class_count,
            "names": names,
        },
        overwrite=True,
        verbose=False,
    )


def split(
    source: str | PathLike[str],
    output_dir: str | PathLike[str],
    format: str = "yolo",
    by_class: bool = False,
    class_groups: list[list[str]] | None = None,
    ratios: tuple[float, ...] | None = None,
    overwrite: bool = False,
    verbose: bool = True,
    shuffle: bool = True,
    seed: int = 42,
    stratify: bool = False,
) -> None:
    """Split a YOLO dataset into independent directories or ZIP archives.

    Exactly one strategy is selected: ``by_class=True`` creates one dataset
    per class, ``class_groups`` creates one dataset per named group, and
    ``ratios`` creates parts using optional shuffling, seed, and stratification.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Dataset directory or supported archive.
    output_dir : str or os.PathLike[str]
        Output directory or a path pattern with one '*' in its final name.
        A plain directory uses default names (split_part_0, split_<class>, or
        split_group_0). A '*' is replaced by the zero-based part index;
        parts/* creates parts/0, and parts/batch_*.zip creates ZIP archives.
        Without a .zip suffix, generated outputs are directories.
    format : str, default='yolo'
        Dataset format to split. Currently only ``'yolo'`` is supported.
    by_class : bool, default=False
        Create one dataset for every class.
    class_groups : list[list[str]] or None, default=None
        Explicit class-name groups.
    ratios : tuple of float or None, default=None
        Two or more ratios whose sum is one.
    overwrite : bool, default=False
        Replace existing output datasets.
    verbose : bool, default=True
        Display archive progress.
    shuffle : bool, default=True
        Whether to shuffle samples before ratio splitting.
    seed : int, default=42
        Random seed used when shuffling.
    stratify : bool, default=False
        Whether to preserve per-class label proportions across ratio splits
        using iterative multi-label stratification.

    Returns
    -------
    None
        Writes datasets to the selected output destinations.

    Raises
    ------
    TypeError
        If ``format`` is not a string.
    ValueError
        If ``format`` is not supported, or the split strategy, class groups, or ratios are invalid.
    FileNotFoundError
        If the source dataset or its images directory is missing.
    FileExistsError
        If an output dataset exists and ``overwrite`` is false.

    Examples
    --------
    >>> from klygo import datasets
    >>> datasets.split("dataset", "parts", ratios=(0.8, 0.2))
    """
    validate_dataset_format(format)
    params = Split(
        source,
        output_dir,
        by_class,
        class_groups,
        ratios,
        overwrite,
        verbose,
        shuffle=shuffle,
        seed=seed,
        stratify=stratify,
    )
    files.mkdir(params.output_dir)

    with open_dataset(params.source, verbose=False) as root:
        class_map = read_class_map(root)
        names = list(class_map.values())
        pairs = scan_dataset_files(files.join(root, "images"), files.join(root, "labels"))

        if params.class_groups is not None or params.by_class:
            groups = params.class_groups or [[name] for name in names]
            unknown = [name for group in groups for name in group if name not in names]
            if unknown:
                raise ValueError(f"Unknown classes in class_groups: {tuple(unknown)}")

            output_names: list[str] = []
            used_names: set[str] = set()
            for index, group in enumerate(groups):
                name = (
                    params.output_pattern.replace("*", str(index))
                    if params.output_pattern else _default_name(group, index)
                )
                if name.casefold() in used_names:
                    name = f"{name}_{index}"
                output_names.append(name)
                used_names.add(name.casefold())

            for index, (group, output_name) in enumerate(zip(groups, output_names)):
                destination = files.join(params.output_dir, output_name)
                if files.exists(destination) and not params.overwrite:
                    raise FileExistsError(f"split output already exists: {destination}")
                selected_ids = {old_id for old_id, name in class_map.items() if name in group}
                id_map = {old_id: group.index(class_map[old_id]) for old_id in selected_ids}

                with dataset_workspace("split-class") as workspace:
                    copied = 0
                    for image_path, label_path, relative_image, _ in pairs:
                        if label_path is None:
                            continue
                        image_name = flattened_name(relative_image)
                        output_label = files.join(workspace, "labels", files.with_extension(image_name, ".txt"))
                        kept = remap_label_file(
                            label_path,
                            output_label,
                            id_map,
                            allowed_ids=selected_ids,
                        )
                        if kept == 0:
                            files.remove(output_label, missing_ok=True)
                            continue
                        copy_entry(image_path, files.join(workspace, "images", image_name))
                        copied += 1

                    if copied:
                        _write_split_metadata(workspace, group)
                        write_output(
                            workspace,
                            destination,
                            overwrite=params.overwrite,
                            verbose=params.verbose,
                        )
        else:
            shuffled = list(pairs)
            if params.shuffle:
                random.Random(params.seed).shuffle(shuffled)
            part_ratios = list(params.ratios or ())
            if params.stratify:
                targets = {
                    str(i): (
                        len(shuffled) - sum(int(len(shuffled) * r) for r in part_ratios[:-1])
                        if i == len(part_ratios) - 1
                        else int(len(shuffled) * r)
                    )
                    for i, r in enumerate(part_ratios)
                }
                ratio_map = {str(i): r for i, r in enumerate(part_ratios)}
                stratified_groups = _stratified_split(shuffled, targets, ratio_map)
            cursor = 0
            for index, ratio in enumerate(part_ratios):
                if params.stratify:
                    selected = stratified_groups[str(index)]
                else:
                    end = len(shuffled) if index == len(part_ratios) - 1 else cursor + int(len(shuffled) * ratio)
                    selected = shuffled[cursor:end]
                    cursor = end
                output_name = (
                    params.output_pattern.replace("*", str(index))
                    if params.output_pattern else f"split_part_{index}"
                )
                destination = files.join(params.output_dir, output_name)
                if files.exists(destination) and not params.overwrite:
                    raise FileExistsError(f"split output already exists: {destination}")

                with dataset_workspace("split-ratio") as workspace:
                    for image_path, label_path, relative_image, _ in selected:
                        image_name = flattened_name(relative_image)
                        copy_entry(image_path, files.join(workspace, "images", image_name))
                        if label_path is not None:
                            copy_entry(label_path, files.join(workspace, "labels", files.with_extension(image_name, ".txt")))
                    _write_split_metadata(workspace, class_map)
                    write_output(
                        workspace,
                        destination,
                        overwrite=params.overwrite,
                        verbose=params.verbose,
                    )


__all__ = ["split"]
