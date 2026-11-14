"""Load a flat YOLO detection dataset into normalized outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from os import PathLike
from pathlib import Path
from urllib.parse import unquote

from klygo import files, media, outputs, validators
from klygo.outputs.detect import Box, Detection, Detections
from klygo.utils.progress import ProgressBar

from ._structure import read_class_map, scan_dataset_files, validate_dataset_format
from ._workspace import open_dataset


def _read_boxes(label_path: Path | None, class_map: dict[int, str], image) -> list[Box]:
    """Decode five-column YOLO annotations, preserving their class IDs."""
    if label_path is None:
        return []
    width, height = image.size
    boxes: list[Box] = []
    for line_number, line in enumerate(files.load(label_path, as_lines=True, verbose=False), 1):
        parts = line.split()
        if not parts:
            continue
        location = f"{label_path}:{line_number}"
        if len(parts) != 5:
            raise ValueError(f"{location}: expected class_id cx cy width height")
        try:
            class_id = int(parts[0])
            cx, cy, box_width, box_height = map(float, parts[1:])
        except ValueError as error:
            raise ValueError(f"{location}: invalid YOLO detection annotation") from error
        if class_id not in class_map:
            raise ValueError(f"{location}: unknown class ID {class_id}")
        coordinates = (cx, cy, box_width, box_height)
        if (
            not all(isfinite(value) and 0 <= value <= 1 for value in coordinates)
            or box_width == 0
            or box_height == 0
        ):
            raise ValueError(f"{location}: expected normalized coordinates and positive box dimensions")
        boxes.append(
            Box(
                uid=len(boxes),
                label=class_map[class_id],
                score=1.0,
                box=[
                    (cx - box_width / 2) * width,
                    (cy - box_height / 2) * height,
                    (cx + box_width / 2) * width,
                    (cy + box_height / 2) * height,
                ],
                class_id=class_id,
                metadata={"class_id": class_id, "ground_truth": True},
            )
        )
    return boxes


def _validate_non_empty_name(value: object, param_name: str) -> str:
    """Validate that a directory or YAML filename parameter is a non-empty string."""
    validators.validate_type(value, str, param_name)
    if not value.strip():
        raise ValueError(f"{param_name} must be a non-empty string")
    return value


def _resolve_label_specs(
    labels_dir: str | Sequence[str] | Mapping[str, str],
    yaml_name: str | Sequence[str] | Mapping[str, str],
) -> tuple[bool, list[tuple[str, str, str]]]:
    """Normalize ``labels_dir`` and ``yaml_name`` into ``(multi, [(key, dir_name, yaml_file), ...])``."""
    if isinstance(labels_dir, str):
        return (
            False,
            [(labels_dir, _validate_non_empty_name(labels_dir, "labels_dir"), _validate_non_empty_name(yaml_name, "yaml_name"))],
        )

    if isinstance(labels_dir, Mapping):
        if not labels_dir:
            raise ValueError("labels_dir must not be empty")
        entries = [
            (_validate_non_empty_name(key, "labels_dir key"), _validate_non_empty_name(val, "labels_dir value"))
            for key, val in labels_dir.items()
        ]
    elif isinstance(labels_dir, Sequence) and not isinstance(labels_dir, (bytes, bytearray)):
        if not labels_dir:
            raise ValueError("labels_dir must not be empty")
        names = [_validate_non_empty_name(item, "labels_dir") for item in labels_dir]
        if len(set(names)) != len(names):
            raise ValueError("labels_dir entries must be unique")
        entries = [(name, name) for name in names]
    else:
        raise TypeError("labels_dir must be a string, sequence of strings, or mapping")

    if isinstance(yaml_name, str):
        shared_yaml = _validate_non_empty_name(yaml_name, "yaml_name")
        return True, [(key, dir_name, shared_yaml) for key, dir_name in entries]
    if isinstance(yaml_name, Mapping):
        specs: list[tuple[str, str, str]] = []
        for key, dir_name in entries:
            if key in yaml_name:
                yname = yaml_name[key]
            elif dir_name in yaml_name:
                yname = yaml_name[dir_name]
            else:
                raise ValueError(f"yaml_name mapping is missing entry for {key!r}")
            specs.append((key, dir_name, _validate_non_empty_name(yname, "yaml_name")))
        return True, specs
    if isinstance(yaml_name, Sequence) and not isinstance(yaml_name, (bytes, bytearray)):
        if len(yaml_name) != len(entries):
            raise ValueError("yaml_name sequence length must match labels_dir")
        return True, [
            (key, dir_name, _validate_non_empty_name(yname, "yaml_name"))
            for (key, dir_name), yname in zip(entries, yaml_name)
        ]
    raise TypeError("yaml_name must be a string, sequence of strings, or mapping")


def load(
    source: str | PathLike[str],
    format: str = "yolo",
    verbose: bool = True,
    *,
    images_dir: str = "images",
    labels_dir: str | Sequence[str] | Mapping[str, str] = "labels",
    yaml_name: str | Sequence[str] | Mapping[str, str] = "data.yaml",
) -> Detections | outputs.DetectionGroup:
    """Load all images and annotations from a flat YOLO dataset.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Flat YOLO detection dataset directory or supported archive containing
        ``data.yaml``, ``images``, and optional ``labels``. Use
        :func:`unpartition` first when images are organized into splits.
    format : str, default='yolo'
        Annotation format to decode. Currently only ``'yolo'`` is supported.
        Custom datasets must first be normalized to the flat YOLO contract.
    verbose : bool, default=True
        Display loading progress.
    images_dir : str, default='images'
        Name of the image subdirectory under ``source``.
    labels_dir : str or sequence of str or mapping, default='labels'
        Name of one label subdirectory under ``source``, or multiple label
        subdirectories (such as ``['labels', 'pseudo_labels']`` or
        ``{'gt': 'labels', 'pseudo': 'pseudo_labels'}``) to load together
        into a same-source ``DetectionGroup`` while decoding each image once.
        Mappings preserve member names. Sequences use pred/truth for two sets,
        or set_1, set_2, and so on for other counts, in input order.
    yaml_name : str or sequence of str or mapping, default='data.yaml'
        Name of the YAML metadata file under ``source`` (for example,
        ``'pseudo.yaml'``), or per-set YAML names matching ``labels_dir``.

    Returns
    -------
    klygo.outputs.Detections or klygo.outputs.DetectionGroup
        One ``Detections`` collection when ``labels_dir`` is a string, or a
        named ``DetectionGroup`` supporting shared operations and indexing
        by name or position when multiple label sets are requested. Images are materialized as RGB PIL
        images. Each annotation is a pixel XYXY ``Box`` with score 1.0 and its
        original class ID in ``box.metadata``.

    Raises
    ------
    FileNotFoundError
        If the source or its images directory does not exist.
    ValueError
        If ``format`` is unsupported, the dataset is partitioned, the archive
        is unsupported, class metadata is invalid, no images exist, or a label
        is not a valid five-column YOLO detection annotation.
    OSError
        If an image cannot be decoded or a dataset file cannot be read.

    Notes
    -----
    This function only reads a flat dataset. Call ``unpartition()`` explicitly
    to flatten a real train/validation/test dataset, then pass its result here.
    Missing or empty labels produce empty ``Detection`` objects.
    Detection metadata records ``dataset_source``, ``relative_path`` (relative
    to the original images directory), ``split``, and ``ground_truth``.

    Examples
    --------
    >>> from klygo import datasets
    >>> flat = datasets.unpartition("dataset.zip", "flat")
    >>> results = datasets.load(flat, format="yolo", verbose=False)
    >>> sets = datasets.load(
    ...     flat,
    ...     labels_dir=["labels", "pseudo_labels"],
    ...     yaml_name=["data.yaml", "pseudo.yaml"],
    ...     verbose=False,
    ... )
    """
    validate_dataset_format(format)
    _validate_non_empty_name(images_dir, "images_dir")
    multi, specs = _resolve_label_specs(labels_dir, yaml_name)

    source_path = files.resolve(source)
    with open_dataset(source_path, verbose=verbose) as root:
        images_root = files.join(root, images_dir)
        if not files.is_dir(images_root):
            raise ValueError("dataset contains no supported images")

        class_maps = {key: read_class_map(root, yaml_name=yname) for key, _, yname in specs}
        pair_sets = {key: scan_dataset_files(images_root, files.join(root, dir_name)) for key, dir_name, _ in specs}
        reference_pairs = pair_sets[specs[0][0]]
        if not reference_pairs:
            raise ValueError("dataset contains no supported images")
        if any(len(relative_image.parts) != 1 for _, _, relative_image, _ in reference_pairs):
            raise ValueError("dataset must be flat; call datasets.unpartition() before datasets.load()")

        frames_by_key: dict[str, list[Detection]] = {key: [] for key, _, _ in specs}
        with ProgressBar(total=len(reference_pairs), desc="Loading dataset", unit="image", verbose=verbose) as progress:
            for index, (image_path, _, relative_image, _) in enumerate(reference_pairs):
                image = media.load(image_path, verbose=False)[0].to_pil()
                original_path = files.path(unquote(files.as_posix(relative_image)))
                split = original_path.parts[0] if len(original_path.parts) > 1 else "raw"
                rel_posix = files.as_posix(original_path)
                resolved_split = split if split in {"train", "val", "test"} else "raw"
                for key, _, _ in specs:
                    label_path = pair_sets[key][index][1]
                    class_map = class_maps[key]
                    boxes = _read_boxes(label_path, class_map, image)
                    frames_by_key[key].append(
                        Detection(
                            boxes,
                            source_id=rel_posix,
                            frame_index=0,
                            source_path=str(image_path),
                            image_size=image.size,
                            source=image,
                            names=class_map,
                            metadata={
                                "dataset_source": str(source_path),
                                "relative_path": rel_posix,
                                "split": resolved_split,
                                "ground_truth": True,
                            },
                        )
                    )
                progress.update(1)

    collections = {
        key: Detections(frames_by_key[key], metadata={"source_type": "images"}, names=class_maps[key])
        for key, _, _ in specs
    }
    if not multi:
        return collections[specs[0][0]]
    return outputs.group(collections) if isinstance(labels_dir, Mapping) else outputs.group(*collections.values())


__all__ = ["load"]
