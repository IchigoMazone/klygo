"""Build detection or classification datasets from model predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import PIL.Image

from klygo import config, files, media


def _resolve_source(source: Any) -> list[PIL.Image.Image]:
    if isinstance(source, (str, Path)):
        return list(media.load(source, recursive=True, verbose=False))
    if isinstance(source, (list, tuple)):
        images: list[PIL.Image.Image] = []
        for item in source:
            if isinstance(item, (str, Path)):
                images.extend(media.load(item, recursive=True, verbose=False))
            elif isinstance(item, PIL.Image.Image):
                images.append(item.convert("RGB"))
            else:
                images.append(media.to_pil(item).convert("RGB"))
        return images
    raise TypeError("source must be a path or a sequence of paths/images")


def export(
    model: Any,
    output_path: str | Path,
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
    output_path : str or pathlib.Path
        Destination directory.
    format : str
        Detection/YOLO or classification layout.
    source : path-like or sequence
        Image, video, directory, or sequence accepted by ``klygo.media``.
    text_prompt, classes : sequence[str] or None
        Ordered labels. ``classes`` is a compatibility alias.
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
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    destination = files.path(output_path)
    if files.exists(destination):
        if not overwrite:
            raise FileExistsError(f"output_path already exists: {destination}")
        files.remove(destination)
    files.mkdir(destination)

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
    format_name = format.strip().lower()
    label_to_id = {label: index for index, label in enumerate(labels)}

    if format_name in {"classification", "classify", "cls"}:
        counters: dict[str, int] = {}
        for detection in predictions:
            for crop in detection.crop():
                if crop.label not in label_to_id or crop.image is None:
                    continue
                class_name = str(crop.label).strip().replace(" ", "_")
                index = counters.get(class_name, 0)
                media.save(
                    files.join(destination, class_name, f"{class_name}_{index:05d}.jpg"),
                    crop.image,
                    overwrite=True,
                    verbose=False,
                )
                counters[class_name] = index + 1
        return destination

    if format_name not in {"detection", "detect", "det", "yolo"}:
        files.remove(destination)
        raise ValueError(f"unsupported dataset format: {format!r}")

    images_dir = files.join(destination, "images")
    labels_dir = files.join(destination, "labels")
    files.mkdir(images_dir)
    files.mkdir(labels_dir)
    for index, (image, detection) in enumerate(zip(images, predictions), start=1):
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
        files.join(destination, "data.yaml"),
        {
            "path": files.resolve(destination).as_posix(),
            "train": "images",
            "val": "images",
            "nc": len(labels),
            "names": labels,
        },
        overwrite=True,
        verbose=False,
    )
    return destination


__all__ = ["export"]
