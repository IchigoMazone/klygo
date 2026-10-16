"""Build detection or classification datasets from model predictions."""

from __future__ import annotations

import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import PIL.Image

from klygo import config, files, media

from ._workspace import publish_output


def _resolve_source(source: Any) -> list[PIL.Image.Image]:
    if isinstance(source, (str, os.PathLike)):
        return list(media.load(source, recursive=True, verbose=False))
    if isinstance(source, (list, tuple)):
        images: list[PIL.Image.Image] = []
        for item in source:
            if isinstance(item, (str, os.PathLike)):
                images.extend(media.load(item, recursive=True, verbose=False))
            elif isinstance(item, PIL.Image.Image):
                images.append(item.convert("RGB"))
            else:
                images.append(media.to_pil(item).convert("RGB"))
        return images
    raise TypeError("source must be a path or a sequence of paths/images")


def _classification_names(labels: list[str]) -> dict[str, str]:
    """Build unique, filesystem-safe folders for the requested classes."""
    names: dict[str, str] = {}
    used: set[str] = set()
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"COM{index}" for index in range(1, 10))
    reserved.update(f"LPT{index}" for index in range(1, 10))
    for index, label in enumerate(labels):
        candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._") or f"class_{index}"
        if candidate.split(".", 1)[0].upper() in reserved:
            candidate = f"class_{index}_{candidate}"
        if candidate.casefold() in used:
            candidate = f"{candidate}_{index}"
        names[label] = candidate
        used.add(candidate.casefold())
    return names


def export(
    model: Any,
    output_path: str | os.PathLike[str],
    format: str,
    source: Any,
    text_prompt: Sequence[str] | None = None,
    classes: Sequence[str] | None = None,
    batch_size: int = 16,
    threshold: float = 0.4,
    overwrite: bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> Path:
    """Create a dataset from a detector's normalized predictions.

    Parameters
    ----------
    model : object
        Callable Klygo detector or object exposing ``predict``.
    output_path : str or os.PathLike[str]
        Destination directory.
    format : str
        Detection/YOLO or classification layout.
    source : path-like or sequence
        Image, video, directory, or sequence accepted by ``klygo.media``.
    text_prompt : sequence[str] or None, default=None
        Ordered text prompts for the detector.
    classes : sequence[str] or None, default=None
        Ordered class names for the exported dataset.
    batch_size : int, default=16
        Prediction batch size.
    threshold : float, default=0.4
        Runtime postprocessing threshold passed to the model.
    overwrite : bool, default=False
        Replace an existing output directory.
    verbose : bool, default=True
        Enable model progress output.
    **kwargs : object
        Additional permitted runtime overrides passed to the model.

    Returns
    -------
    pathlib.Path
        Created dataset directory.

    Raises
    ------
    ValueError
        If classes are absent, the format is unknown, or no images resolve.
    FileExistsError
        If the output exists and overwrite is disabled.

    Examples
    --------
    >>> from klygo import datasets
    >>> output = datasets.export(
    ...     model, "generated", "yolo", "images", classes=["car"]
    ... )
    """
    labels = list(text_prompt or classes or ())
    if not labels or not all(isinstance(label, str) and label.strip() for label in labels):
        raise ValueError("text_prompt or classes must contain at least one class name")
    if len(set(labels)) != len(labels):
        raise ValueError("class names must be unique")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    format_name = format.strip().lower()
    if format_name not in {"classification", "classify", "cls", "detection", "detect", "det", "yolo"}:
        raise ValueError(f"unsupported dataset format: {format!r}")
    destination = files.path(output_path)
    if files.exists(destination) and not overwrite:
        raise FileExistsError(f"output_path already exists: {destination}")
    images = _resolve_source(source)
    if not images:
        raise ValueError("source did not resolve to any images")
    predictor = model if callable(model) else model.predict
    predictions = predictor(
        images,
        prompt=labels,
        batch=batch_size,
        threshold=threshold,
        verbose=verbose,
        **kwargs,
    )
    label_to_id = {label: index for index, label in enumerate(labels)}

    files.mkdir(files.parent(destination))
    with TemporaryDirectory(prefix=f".{files.name(destination)}-", dir=files.parent(destination)) as directory:
        temporary_root = files.path(directory)
        staging = files.join(temporary_root, "dataset")
        files.mkdir(staging)

        if format_name in {"classification", "classify", "cls"}:
            counters: dict[str, int] = {}
            folder_names = _classification_names(labels)
            for _, detection in zip(images, predictions, strict=True):
                for crop in detection.crop():
                    if crop.label not in label_to_id or crop.image is None:
                        continue
                    class_name = folder_names[crop.label]
                    index = counters.get(class_name, 0)
                    media.save(
                        files.join(staging, class_name, f"{class_name}_{index:05d}.jpg"),
                        crop.image,
                        overwrite=True,
                        verbose=False,
                    )
                    counters[class_name] = index + 1
        else:
            images_dir = files.join(staging, "images")
            labels_dir = files.join(staging, "labels")
            files.mkdir(images_dir)
            files.mkdir(labels_dir)
            for index, (image, detection) in enumerate(zip(images, predictions, strict=True), start=1):
                stem = f"img_{index:05d}"
                media.save(files.join(images_dir, f"{stem}.jpg"), image, overwrite=True, verbose=False)
                width, height = image.size
                annotations: list[str] = []
                for box in detection:
                    class_id = label_to_id.get(box.label)
                    if class_id is None:
                        continue
                    center_x = ((box.xmin + box.xmax) / 2.0) / width
                    center_y = ((box.ymin + box.ymax) / 2.0) / height
                    box_width = (box.xmax - box.xmin) / width
                    box_height = (box.ymax - box.ymin) / height
                    annotations.append(
                        f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
                    )
                files.save(
                    files.join(labels_dir, f"{stem}.txt"),
                    annotations,
                    overwrite=True,
                    verbose=False,
                )

            config.save(
                files.join(staging, "data.yaml"),
                {
                    "path": files.as_posix(files.resolve(destination)),
                    "train": "images",
                    "val": "images",
                    "nc": len(labels),
                    "names": labels,
                },
                overwrite=True,
                verbose=False,
            )

        publish_output(staging, destination, temporary_root)
    return destination


__all__ = ["export"]
