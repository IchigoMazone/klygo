"""Shared implementation for partition and repartition."""

from __future__ import annotations

from collections import Counter, defaultdict
import random
from pathlib import Path
from typing import Any

from klygo import config, files

from ._structure import copy_entry, flattened_name, scan_dataset_files
from ._workspace import dataset_workspace, open_dataset, set_output_metadata_path, write_output


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


def _read_label_classes(label_path: Any) -> frozenset[str]:
    """Return unique YOLO class tokens present in a label file."""
    if label_path is None or not files.is_file(label_path):
        return frozenset()
    text = files.load(label_path, verbose=False)
    classes: set[str] = set()
    for line in str(text).splitlines():
        parts = line.strip().split()
        if parts:
            classes.add(parts[0])
    return frozenset(classes)


def _stratified_split(
    pairs: list[tuple[Any, Any, Any, Any]],
    targets: dict[str, int],
    ratios: dict[str, float],
) -> dict[str, list[tuple[Any, Any, Any, Any]]]:
    """Distribute image/label pairs across splits while preserving class proportions."""
    groups: dict[str, list[tuple[Any, Any, Any, Any]]] = {name: [] for name in targets}
    remaining = dict(targets)
    split_order = [name for name, count in targets.items() if count > 0]
    if not pairs or not split_order:
        return groups

    item_classes = [_read_label_classes(pair[1]) for pair in pairs]
    class_to_items: dict[str, list[int]] = defaultdict(list)
    for idx, cls_set in enumerate(item_classes):
        for cls in cls_set:
            class_to_items[cls].append(idx)

    desired: dict[str, dict[str, float]] = {
        split: {cls: len(indices) * ratios[split] for cls, indices in class_to_items.items()}
        for split in split_order
    }
    unassigned = set(range(len(pairs)))

    # Assign images containing the rarest remaining class first (iterative stratification)
    while unassigned and class_to_items:
        active_classes = [
            (sum(1 for i in indices if i in unassigned), cls)
            for cls, indices in class_to_items.items()
        ]
        active_classes = [(count, cls) for count, cls in active_classes if count > 0]
        if not active_classes:
            break
        _, rarest = min(active_classes, key=lambda item: (item[0], item[1]))

        for idx in class_to_items[rarest]:
            if idx not in unassigned:
                continue
            candidates = [split for split in split_order if remaining[split] > 0]
            if not candidates:
                break
            chosen = max(
                candidates,
                key=lambda split: (
                    desired[split][rarest],
                    remaining[split],
                    -split_order.index(split),
                ),
            )
            groups[chosen].append(pairs[idx])
            remaining[chosen] -= 1
            unassigned.remove(idx)
            for cls in item_classes[idx]:
                desired[chosen][cls] -= 1.0

    # Distribute remaining background (label-free) images proportionally by remaining capacity
    for idx in sorted(unassigned):
        candidates = [split for split in split_order if remaining[split] > 0]
        if not candidates:
            break
        chosen = max(
            candidates,
            key=lambda split: (remaining[split], -split_order.index(split)),
        )
        groups[chosen].append(pairs[idx])
        remaining[chosen] -= 1

    return groups


def execute_partition(
    source: str | Path,
    target: str | Path,
    ratios: tuple[float, ...],
    *,
    overwrite: bool,
    verbose: bool,
    shuffle: bool = True,
    seed: int = 42,
    stratify: bool = False,
) -> Path:
    """Create a deterministic train/validation/test dataset."""
    destination = files.path(target)
    train_ratio, validation_ratio, test_ratio = normalized_ratios(ratios)

    with open_dataset(source, verbose=verbose) as root, dataset_workspace("partition") as workspace:
        images_dir = files.join(root, "images")
        labels_dir = files.join(root, "labels")
        pairs = scan_dataset_files(images_dir, labels_dir)
        if shuffle:
            random.Random(seed).shuffle(pairs)

        train_count = int(len(pairs) * train_ratio)
        validation_count = int(len(pairs) * validation_ratio)
        if stratify:
            if test_ratio > 0:
                test_count = len(pairs) - train_count - validation_count
            else:
                validation_count = len(pairs) - train_count
                test_count = 0
            groups = _stratified_split(
                pairs,
                {"train": train_count, "val": validation_count, "test": test_count},
                {"train": train_ratio, "val": validation_ratio, "test": test_ratio},
            )
        else:
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
                "train": "images/train",
                "val": "images/val",
            }
        )
        set_output_metadata_path(metadata, destination)
        if groups["test"]:
            metadata["test"] = "images/test"
        else:
            metadata.pop("test", None)
        config.save(files.join(workspace, "data.yaml"), metadata, overwrite=True, verbose=False)
        return write_output(workspace, destination, overwrite=overwrite, verbose=verbose)


__all__ = ["execute_partition", "normalized_ratios", "_stratified_split"]
