"""Shared implementation for partition and repartition."""

from __future__ import annotations

import random
from pathlib import Path

from klygo import config, files

from ._structure import copy_entry, flattened_name, scan_dataset_files
from ._workspace import dataset_workspace, open_dataset, write_output


def normalized_ratios(ratios: tuple[float, ...]) -> tuple[float, float, float]:
    """Expand one to three ratios into train, validation, and test values."""
    if len(ratios) == 1:
        train = ratios[0]
        validation = test = (1.0 - train) / 2.0
    elif len(ratios) == 2:
        train, validation = ratios
        test = 1.0 - train - validation
    else:
        train, validation, test = ratios
    return round(train, 8), round(validation, 8), round(test, 8)


def execute_partition(
    source: str | Path,
    target: str | Path,
    ratios: tuple[float, ...],
    *,
    overwrite: bool,
    verbose: bool,
    seed: int = 42,
) -> Path:
    """Create a deterministic train/validation/test dataset."""
    destination = files.path(target)
    train_ratio, validation_ratio, test_ratio = normalized_ratios(ratios)

    with open_dataset(source, verbose=verbose) as root, dataset_workspace("partition") as workspace:
        images_dir = files.join(root, "images")
        labels_dir = files.join(root, "labels")
        pairs = scan_dataset_files(images_dir, labels_dir)
        random.Random(seed).shuffle(pairs)

        train_count = int(len(pairs) * train_ratio)
        validation_count = int(len(pairs) * validation_ratio)
        groups = {
            "train": pairs[:train_count],
            "val": pairs[train_count:train_count + validation_count],
            "test": pairs[train_count + validation_count:] if test_ratio > 0 else [],
        }
        if test_ratio == 0:
            groups["val"] = pairs[train_count:]

        for split_name, split_pairs in groups.items():
            for image_path, label_path, relative_image, relative_label in split_pairs:
                image_name = flattened_name(relative_image)
                label_name = files.with_extension(image_name, ".txt")
                copy_entry(image_path, files.join(workspace, "images", split_name, image_name))
                if label_path is not None:
                    copy_entry(label_path, files.join(workspace, "labels", split_name, label_name))

        yaml_path = files.join(root, "data.yaml")
        metadata = config.load(yaml_path, verbose=False).to_dict() if files.is_file(yaml_path) else {}
        metadata.update(
            {
                "path": files.resolve(destination).as_posix(),
                "train": "images/train",
                "val": "images/val",
            }
        )
        if groups["test"]:
            metadata["test"] = "images/test"
        else:
            metadata.pop("test", None)
        config.save(files.join(workspace, "data.yaml"), metadata, overwrite=True, verbose=False)
        return write_output(workspace, destination, overwrite=overwrite, verbose=verbose)


__all__ = ["execute_partition", "normalized_ratios"]
