"""Load a flat YOLO detection dataset into normalized outputs."""

from __future__ import annotations

from math import isfinite
from os import PathLike
from pathlib import Path
from urllib.parse import unquote

from klygo import files, media
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


def load(
    source: str | PathLike[str],
    format: str = "yolo",
    verbose: bool = True,
) -> Detections:
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

    Returns
    -------
    klygo.outputs.detect.Detections
        One collection containing every image in the flat dataset. Images are
        materialized as RGB PIL images. Each annotation is a pixel XYXY ``Box`` with
        score 1.0 and its original class ID in ``box.metadata``.

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
    >>> results[0].show()
    >>> results[0].show(mode="uid")
    """
    validate_dataset_format(format)

    source_path = files.resolve(source)
    frames: list[Detection] = []
    with open_dataset(source_path, verbose=verbose) as root:
        class_map = read_class_map(root)
        images_dir = files.join(root, "images")
        if not files.is_dir(images_dir):
            raise ValueError("dataset contains no supported images")
        pairs = scan_dataset_files(images_dir, files.join(root, "labels"))
        if not pairs:
            raise ValueError("dataset contains no supported images")
        if any(len(relative_image.parts) != 1 for _, _, relative_image, _ in pairs):
            raise ValueError("dataset must be flat; call datasets.unpartition() before datasets.load()")
        with ProgressBar(total=len(pairs), desc="Loading dataset", unit="image", verbose=verbose) as progress:
            for index, (image_path, label_path, relative_image, _) in enumerate(pairs):
                image = media.load(image_path, verbose=False)[0].to_pil()
                boxes = _read_boxes(label_path, class_map, image)
                original_path = files.path(unquote(files.as_posix(relative_image)))
                split = original_path.parts[0] if len(original_path.parts) > 1 else "raw"
                frames.append(
                    Detection(
                        boxes,
                        source_id=files.as_posix(original_path),
                        frame_index=0,
                        source_path=str(image_path),
                        image_size=image.size,
                        source=image,
                        names=class_map,
                        metadata={"dataset_source": str(source_path), "relative_path": files.as_posix(original_path),
                                  "split": split if split in {"train", "val", "test"} else "raw", "ground_truth": True},
                    )
                )
                progress.update(1)
    return Detections(frames, metadata={"source_type": "images"}, names=class_map)


__all__ = ["load"]
