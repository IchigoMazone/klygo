"""Convert raw image and label trees into a flat YOLO dataset."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
import warnings

from klygo import config, files, validators

from ._structure import IMAGE_EXTENSIONS, copy_entry, validate_dataset_format
from ._workspace import dataset_workspace, write_output


_CONFLICT_MODES = {"error", "rename", "skip", "overwrite"}


@dataclass(frozen=True)
class _Sample:
    image: Path
    label: Path | None
    stem: str


def _patterns(value, name: str) -> list[str]:
    if isinstance(value, str):
        result = [value]
    elif isinstance(value, list):
        result = value
    else:
        raise TypeError(f"{name} must be a string or list of strings")
    if not result or any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{name} must contain non-empty path patterns")
    return result


def _matches(root: Path, pattern: str, *, directories: bool) -> list[Path]:
    relative = files.path(pattern)
    if relative.is_absolute():
        raise ValueError("tree paths must be relative to source")
    direct = files.join(root, relative)
    if files.exists(direct):
        candidates = [direct]
    else:
        candidates = files.list_entries(root, pattern=pattern, recursive=False)
    result = []
    for candidate in candidates:
        resolved = files.resolve(candidate)
        if not files.is_within(resolved, root):
            raise ValueError(f"tree path escapes source: {pattern}")
        if directories and files.is_dir(resolved):
            result.append(resolved)
        elif not directories and files.is_file(resolved):
            result.append(resolved)
    return sorted(set(result), key=files.as_posix)


def _class_map(value) -> dict[int, str]:
    if isinstance(value, (list, tuple)):
        result = {index: str(name).strip() for index, name in enumerate(value)}
    elif isinstance(value, dict):
        try:
            result = {int(index): str(name).strip() for index, name in value.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("class IDs must be integers") from error
    else:
        raise TypeError("classes or YAML names must be a mapping or sequence")
    if set(result) != set(range(len(result))):
        raise ValueError("class IDs must be contiguous from 0")
    if any(not name for name in result.values()) or len(set(result.values())) != len(result):
        raise ValueError("class names must be non-empty and unique")
    return result


def _metadata(root: Path, yaml_pattern: str | None, classes) -> dict:
    if yaml_pattern is not None and classes is not None:
        raise ValueError("classes cannot be used together with a YAML source")
    if yaml_pattern is not None:
        matches = _matches(root, yaml_pattern, directories=False)
        if len(matches) != 1 or files.extension(matches[0]).lower() not in {".yaml", ".yml"}:
            raise ValueError("the third tree entry must resolve to exactly one YAML file")
        source_metadata = config.load(matches[0], verbose=False).to_dict()
        if "names" not in source_metadata:
            raise ValueError("source YAML must define names")
        names = _class_map(source_metadata["names"])
        if "nc" in source_metadata and int(source_metadata["nc"]) != len(names):
            raise ValueError("source YAML nc must match names")
    elif classes is not None:
        if not isinstance(classes, dict):
            raise TypeError("classes must be a mapping of contiguous integer IDs to names")
        names = _class_map(classes)
    else:
        names = {}
        warnings.warn("No class metadata was provided; the dataset is not ready for training", UserWarning, stacklevel=3)
    return {"path": ".", "train": "images", "nc": len(names), "names": names}


def _discover_group(image_root: Path, label_root: Path | None) -> tuple[list[_Sample], int]:
    images = [
        path for path in files.find(image_root, "*", recursive=True)
        if files.extension(path).lower() in IMAGE_EXTENSIONS
    ]
    labels = files.find(label_root, "*.txt", recursive=True) if label_root is not None else []
    label_map: dict[str, Path] = {}
    for label in labels:
        key = files.as_posix(files.with_extension(files.relative(label, label_root), "")).lower()
        if key in label_map:
            raise ValueError(f"duplicate label stem in one source group: {key}")
        label_map[key] = label
    used = set()
    samples = []
    for image in images:
        key = files.as_posix(files.with_extension(files.relative(image, image_root), "")).lower()
        label = label_map.get(key)
        if label is not None:
            used.add(key)
        samples.append(_Sample(image, label, files.stem(image)))
    return samples, len(set(label_map) - used)


def _plan(samples: list[_Sample], mode: str) -> list[_Sample]:
    planned: dict[str, _Sample] = {}
    for sample in samples:
        base = sample.stem
        key = base.lower()
        if key not in planned:
            planned[key] = sample
            continue
        if mode == "error":
            raise ValueError(f"sample name conflict: {base}")
        if mode == "skip":
            continue
        if mode == "overwrite":
            planned[key] = sample
            continue
        suffix = 1
        while f"{base}_{suffix}".lower() in planned:
            suffix += 1
        renamed = f"{base}_{suffix}"
        planned[renamed.lower()] = _Sample(sample.image, sample.label, renamed)
    return list(planned.values())


def _validate_labels(samples: list[_Sample], names: dict[int, str]) -> None:
    if not names:
        return
    for sample in samples:
        if sample.label is None:
            continue
        for line_number, line in enumerate(files.load(sample.label, as_lines=True, verbose=False), 1):
            parts = str(line).strip().split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
            except ValueError as error:
                raise ValueError(f"{sample.label}:{line_number}: invalid class ID") from error
            if class_id not in names:
                raise ValueError(f"{sample.label}:{line_number}: unknown class ID {class_id}")


def convert(
    source: str | PathLike[str],
    target: str | PathLike[str],
    format: str = "yolo",
    tree: dict[str, str | list[str]] | None = None,
    classes: dict[int, str] | None = None,
    on_conflict: str = "error",
    overwrite: bool = False,
) -> Path:
    """Convert mapped source directories into one flat YOLO dataset.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Root directory containing every mapped input.
    target : str or os.PathLike[str]
        Flat dataset output directory.
    format : str, default='yolo'
        Target dataset format. Currently only ``'yolo'`` is supported.
    tree : dict
        Ordered mapping whose first value selects image directories, second
        value selects label directories, and optional third value selects one YAML file.
    classes : dict[int, str], optional
        Contiguous class names used when no YAML source is supplied.
    on_conflict : {'error', 'rename', 'skip', 'overwrite'}, default='error'
        Whole-sample filename conflict policy.
    overwrite : bool, default=False
        Atomically replace an existing target.

    Returns
    -------
    pathlib.Path
        Created flat dataset directory.

    Raises
    ------
    FileNotFoundError
        If ``source`` does not exist.
    FileExistsError
        If ``target`` exists and overwrite is disabled.
    TypeError
        If an argument or tree value has an unsupported type.
    ValueError
        If paths, metadata, conflicts, or class IDs violate the conversion contract.
    """
    validate_dataset_format(format)
    validators.validate_type(source, (str, PathLike), "source")
    validators.validate_type(target, (str, PathLike), "target")
    validators.validate_type(overwrite, bool, "overwrite")
    if tree is None or not isinstance(tree, dict) or len(tree) not in {2, 3}:
        raise ValueError("tree must be an ordered mapping with exactly two or three entries")
    if not isinstance(on_conflict, str) or on_conflict not in _CONFLICT_MODES:
        raise ValueError(f"on_conflict must be one of {sorted(_CONFLICT_MODES)}")

    root, destination = files.resolve(source), files.resolve(target)
    if not files.is_dir(root):
        raise FileNotFoundError(f"source directory does not exist: {root}")
    if files.exists(destination) and not overwrite:
        raise FileExistsError(f"target already exists: {destination}")
    if files.is_within(destination, root) or files.is_within(root, destination):
        raise ValueError("source and target directories must not overlap")

    values = list(tree.values())
    image_patterns = _patterns(values[0], "image tree entry")
    label_patterns = _patterns(values[1], "label tree entry")
    if len(image_patterns) != len(label_patterns):
        raise ValueError("image and label tree entries must contain the same number of patterns")
    yaml_pattern = values[2] if len(values) == 3 else None
    if yaml_pattern is not None and not isinstance(yaml_pattern, str):
        raise TypeError("the third tree entry must be one YAML path string")
    metadata = _metadata(root, yaml_pattern, classes)

    discovered: list[_Sample] = []
    orphan_labels = 0
    for image_pattern, label_pattern in zip(image_patterns, label_patterns):
        image_roots = _matches(root, image_pattern, directories=True)
        label_roots = _matches(root, label_pattern, directories=True)
        if not image_roots:
            warnings.warn(f"No image directories matched: {image_pattern}", UserWarning, stacklevel=2)
        if not label_roots:
            warnings.warn(f"No label directories matched: {label_pattern}", UserWarning, stacklevel=2)
        if image_roots and label_roots and len(image_roots) != len(label_roots):
            raise ValueError("image and label glob patterns must resolve to the same number of directories")
        if not image_roots:
            orphan_labels += sum(len(files.find(label_root, "*.txt", recursive=True)) for label_root in label_roots)
        for index, image_root in enumerate(image_roots):
            label_root = label_roots[index] if label_roots else None
            group, orphans = _discover_group(image_root, label_root)
            discovered.extend(group)
            orphan_labels += orphans

    planned = _plan(discovered, on_conflict)
    _validate_labels(planned, metadata["names"])
    if not planned:
        warnings.warn("No supported images were found; an empty dataset will be created", UserWarning, stacklevel=2)
    unlabeled = sum(sample.label is None for sample in planned)
    if unlabeled:
        warnings.warn(f"Keeping {unlabeled} unlabeled image(s) as negative samples", UserWarning, stacklevel=2)
    if orphan_labels:
        warnings.warn(f"Ignoring {orphan_labels} label file(s) without matching images", UserWarning, stacklevel=2)

    with dataset_workspace("convert") as workspace:
        files.mkdir(files.join(workspace, "images"))
        files.mkdir(files.join(workspace, "labels"))
        for sample in planned:
            image_name = sample.stem + files.extension(sample.image).lower()
            copy_entry(sample.image, files.join(workspace, "images", image_name))
            if sample.label is not None:
                copy_entry(sample.label, files.join(workspace, "labels", sample.stem + ".txt"))
        config.save(files.join(workspace, "data.yaml"), metadata, overwrite=True, verbose=False)
        return write_output(workspace, destination, overwrite=overwrite, verbose=False)


__all__ = ["convert"]
