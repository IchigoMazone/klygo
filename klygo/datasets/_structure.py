"""Internal dataset structure, label, and class-name helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.parse import quote

from klygo import config, files


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
DatasetPair = tuple[Path, Path | None, Path, Path]


def find_dataset_root(path: str | Path) -> Path:
    """Return the nearest directory containing ``data.yaml`` and ``images``."""
    root = files.path(path)
    if files.is_file(root):
        root = files.parent(root)
    if files.exists(files.join(root, "data.yaml")) and files.is_dir(files.join(root, "images")):
        return root

    candidates = files.find(root, "data.yaml", recursive=True)
    for candidate in sorted(candidates, key=lambda value: len(value.parts)):
        parent = files.parent(candidate)
        if files.is_dir(files.join(parent, "images")):
            return parent
    return root


def read_class_map(dataset_root: str | Path) -> dict[int, str]:
    """Read class names while preserving numeric IDs from ``data.yaml``."""
    root = files.path(dataset_root)
    yaml_path = files.join(root, "data.yaml")
    if not files.is_file(yaml_path):
        raise ValueError(f"data.yaml not found under {root}")

    data = config.load(yaml_path, verbose=False)
    if "names" not in data:
        raise ValueError(f"names list not found in data.yaml under {root}")
    names = data["names"]
    if isinstance(names, dict):
        try:
            class_map = {int(key): str(value) for key, value in names.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("class IDs in data.yaml must be integers") from error
        if len(class_map) != len(names) or any(index < 0 for index in class_map):
            raise ValueError("class IDs in data.yaml must be unique non-negative integers")
    elif isinstance(names, (list, tuple)):
        class_map = {index: str(value) for index, value in enumerate(names)}
    else:
        class_map = {0: str(names)}
    if len(set(class_map.values())) != len(class_map):
        raise ValueError("class names in data.yaml must be unique")
    return dict(sorted(class_map.items()))


def read_class_names(dataset_root: str | Path) -> list[str]:
    """Read class names in numeric ID order."""
    return list(read_class_map(dataset_root).values())


def scan_dataset_files(
    images_dir: str | Path,
    labels_dir: str | Path,
) -> list[DatasetPair]:
    """Return image/label pairs while preserving paths relative to their roots."""
    images_root = files.path(images_dir)
    labels_root = files.path(labels_dir)
    if not files.is_dir(images_root):
        raise FileNotFoundError(f"images directory not found: {images_root}")

    image_paths = [
        path
        for path in files.find(images_root, "*", recursive=True)
        if files.is_file(path) and files.extension(path).lower() in IMAGE_EXTENSIONS
    ]
    pairs: list[DatasetPair] = []
    for image_path in sorted(set(image_paths), key=lambda value: files.as_posix(value)):
        relative_image = files.relative(image_path, images_root)
        relative_label = files.with_extension(relative_image, ".txt")
        label_path = files.join(labels_root, relative_label)
        if not files.is_file(label_path):
            fallback = files.join(labels_root, files.with_extension(files.name(image_path), ".txt"))
            if files.is_file(fallback):
                label_path = fallback
                relative_label = files.path(files.name(fallback))
            else:
                label_path = None
        pairs.append((image_path, label_path, relative_image, relative_label))
    return pairs


def copy_entry(source: str | Path, target: str | Path) -> Path | None:
    """Copy an existing entry, ignoring an absent optional source."""
    if not files.exists(source):
        return None
    if files.resolve(source) == files.resolve(target):
        return files.path(target)
    return files.copy(source, target, overwrite=True)


def move_entry(source: str | Path, target: str | Path) -> Path | None:
    """Move an existing entry, ignoring an absent optional source."""
    if not files.exists(source):
        return None
    if files.resolve(source) == files.resolve(target):
        return files.path(target)
    return files.move(source, target, overwrite=True)


def remap_label_file(
    source: str | Path,
    target: str | Path,
    class_map: dict[int, int],
    *,
    allowed_ids: Iterable[int] | None = None,
) -> int:
    """Rewrite YOLO class IDs and return the number of retained annotations."""
    allowed = set(allowed_ids) if allowed_ids is not None else None
    output: list[str] = []
    retained = 0
    for line in files.load(source, as_lines=True, verbose=False):
        parts = str(line).strip().split()
        if not parts:
            output.append("")
            continue
        try:
            old_id = int(parts[0])
        except ValueError:
            output.append(str(line).rstrip("\n"))
            continue
        if allowed is not None and old_id not in allowed:
            continue
        parts[0] = str(class_map.get(old_id, old_id))
        output.append(" ".join(parts))
        retained += 1
    files.save(target, output, overwrite=True, verbose=False)
    return retained


def flattened_name(relative_path: Path) -> str:
    """Encode a relative path as one filename without losing separators."""
    return quote(files.as_posix(relative_path), safe="")


__all__ = [
    "DatasetPair",
    "IMAGE_EXTENSIONS",
    "copy_entry",
    "find_dataset_root",
    "flattened_name",
    "move_entry",
    "read_class_map",
    "read_class_names",
    "remap_label_file",
    "scan_dataset_files",
]
