"""Dataset statistical analysis, multi-criteria scoring, and comparative reporting.

This submodule provides functions to measure dataset class imbalance, calculate
per-image visual and informational quality scores, and quantify deltas between
original and curated dataset versions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Optional

import numpy as np

from klygo.outputs import Detections
from klygo import validators
from klygo.utils.progress import _iter_progress

from ._common import _Report, _normalize, _quality, _require


@dataclass
class MiningStats(_Report):
    """Aggregate statistics summarizing images, annotations, classes, and visual quality.

    Attributes
    ----------
    images : int
        Total number of images in the dataset.
    boxes : int
        Total number of bounding boxes across all images.
    classes : int
        Number of unique class labels defined in the dataset registry.
    empty_images : int
        Number of background images with no bounding box annotations.
    multilabel_images : int
        Number of images containing more than one distinct class.
    images_per_class : dict[str, int]
        Mapping of class name to the count of images containing that class.
    boxes_per_class : dict[str, int]
        Mapping of class name to total bounding box count.
    box_area : dict[str, float]
        Normalized box area statistics (`min`, `median`, `mean`, `max`) relative
        to image dimensions.
    cooccurrence : dict[str, dict[str, int]]
        Pairwise co-occurrence frequency matrix between distinct classes.
    imbalance_ratio : float
        Ratio of maximum class count to minimum class count among present classes.
    quality : dict[str, dict[str, float]]
        Aggregated image quality metrics (`brightness`, `contrast`, `sharpness`,
        `entropy`, `noise`, `exposure`), each providing `min`, `mean`, and `max`.

    Methods
    ----------
    show(console=None)
        Display a compact, formatted Rich table in the terminal.
    to_dict()
        Return a JSON-serializable dictionary representation.
    save(path)
        Save statistics to disk as `.json`, `.md`, or `.csv`.
    """

    images: int
    boxes: int
    classes: int
    empty_images: int
    multilabel_images: int
    images_per_class: dict[str, int]
    boxes_per_class: dict[str, int]
    box_area: dict[str, float]
    cooccurrence: dict[str, dict[str, int]]
    imbalance_ratio: float
    quality: dict[str, dict[str, float]] = field(default_factory=dict)
    present_classes: int = 0
    missing_classes: list[str] = field(default_factory=list)
    box_imbalance_ratio: float = 0.0
    failed_quality_images: int = 0

@dataclass
class MiningScores(_Report):
    """Composite and component-level score distributions for every image.

    Attributes
    ----------
    values : list[float]
        Normalized composite scores in [0.0, 1.0] for every frame in the dataset.
    details : list[dict[str, float]]
        Per-frame breakdown of individual components (`quality`, `rarity`,
        `coverage`, `boxes`).
    weights : dict[str, float]
        Active weighting configuration used to compute the composite scores.
    identities : list[list[Any]]
        Source identifiers `[source_id, frame_index]` matching each score entry.

    Methods
    ----------
    show(console=None)
        Print a ranked Rich score table in the terminal.
    to_dict()
        Convert scores and components to a dictionary.
    save(path)
        Save the scores to a `.json`, `.md`, or `.csv` table.
    """

    values: list[float]
    details: list[dict[str, float]]
    weights: dict[str, float]
    identities: list[list[Any]]

    def _rows(self) -> list[dict[str, Any]]:
        return [
            {"index": index, "score": value, **self.details[index]}
            for index, value in enumerate(self.values)
        ]


@dataclass
class MiningComparison(_Report):
    """Comparative analysis report between two dataset snapshots.

    Attributes
    ----------
    before : MiningStats
        Full statistics of the original reference dataset.
    after : MiningStats
        Full statistics of the curated or filtered dataset.
    changes : dict[str, float]
        Numeric differences (`after - before`) for key metrics (`images`,
        `boxes`, `classes`, `imbalance_ratio`).

    Methods
    ----------
    show(console=None)
        Display before, after, and delta values side-by-side.
    to_dict()
        Return a JSON-serializable representation of the comparison.
    save(path)
        Save the comparison report as `.json`, `.md`, or `.csv`.
    """

    before: MiningStats
    after: MiningStats
    changes: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "changes": self.changes,
        }


def _declared_classes(data: Detections) -> set[str]:
    """Return declared class labels from `data.names` and optional `metadata['names']`."""
    declared = set(data.names.values())
    meta_names = data.metadata.get("names")
    if isinstance(meta_names, Mapping):
        declared.update(str(v) for v in meta_names.values())
    elif isinstance(meta_names, (list, tuple, set)):
        declared.update(str(v) for v in meta_names)
    return declared


def stats(
    data: Detections,
    *,
    quality: bool = True,
    cooccurrence: bool = True,
    verbose: bool = False,
) -> MiningStats:
    """Summarize an object-detection dataset at the image and annotation level.

    Aggregates image counts, box distributions, class frequencies, relative
    box area percentiles, label co-occurrence, dataset imbalance ratio, and
    optional image quality scores.

    Parameters
    ----------
    data : Detections
        Materialized detection collection to analyze.
    quality : bool, default=True
        Whether to compute image quality metrics (sharpness, contrast, entropy,
        noise, exposure). Requires reading image pixels.
    cooccurrence : bool, default=True
        Whether to compute the pairwise class co-occurrence frequency matrix.

    verbose : bool, default=False
        Display throttled progress for processing stages; disabled by default.

    Returns
    -------
    MiningStats
        Structured report dataclass containing all computed statistics.

    Raises
    ------
    TypeError
        If `data` is not an instance of `klygo.outputs.Detections`.
    ValueError
        If `data` is a streaming generator rather than a materialized collection.

    Notes
    -----
    This function operates in a read-only manner and leaves the input dataset
    completely unmodified.

    Examples
    --------
    >>> from klygo import datasets, mining
    >>> data = datasets.load("artifacts/datasets/flat_data", format="yolo", verbose=False)
    >>> summary = mining.stats(data)
    >>> utils.tabulate(summary.to_dict())
    >>> print(summary.imbalance_ratio)
    """
    validators.validate_type(verbose, bool, "verbose")
    _require(data)

    image_counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()
    pairs: dict[str, Counter[str]] = defaultdict(Counter)
    areas: list[float] = []
    empty = 0
    multilabel = 0

    # 1. Aggregate annotations, class frequencies, and relative box areas
    for frame in _iter_progress(data, verbose, "Annotation statistics"):
        labels = list(frame.labels)
        unique = sorted(set(labels))

        if not labels:
            empty += 1
        if len(unique) > 1:
            multilabel += 1

        image_counts.update(unique)
        box_counts.update(labels)

        width, height = frame.image_size
        areas.extend((box.width * box.height) / (width * height) for box in frame)

        if cooccurrence:
            for index, first in enumerate(unique):
                for second in unique[index + 1:]:
                    pairs[first][second] += 1
                    pairs[second][first] += 1

    positives = [value for value in image_counts.values() if value]
    box_positives = [value for value in box_counts.values() if value]
    present_classes = len(image_counts)
    declared_classes = _declared_classes(data)
    missing_classes = sorted(declared_classes - set(image_counts))

    # 2. Compute visual image quality distribution
    raw_quality = _quality(data, verbose=verbose, ignore_errors=True) if quality else []
    quality_rows = [row for row in raw_quality if row is not None]
    failed_quality = len(raw_quality) - len(quality_rows)
    quality_summary = {
        key: {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for key in (quality_rows[0] if quality_rows else {})
        for values in [[row[key] for row in quality_rows]]
    }

    # 3. Calculate box area percentiles
    if areas:
        p25, p50, p75, p90 = (float(v) for v in np.percentile(areas, [25, 50, 75, 90]))
    else:
        p25 = p50 = p75 = p90 = 0.0
    area = {
        "mean": float(np.mean(areas)) if areas else 0.0,
        "median": float(np.median(areas)) if areas else 0.0,
        "min": min(areas, default=0.0),
        "max": max(areas, default=0.0),
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p90": p90,
    }

    # 4. Calculate dataset imbalance ratios (image-level and box-level)
    imbalance = (max(positives) / min(positives)) if positives else 0.0
    box_imbalance = (max(box_positives) / min(box_positives)) if box_positives else 0.0

    return MiningStats(
        images=len(data),
        boxes=sum(box_counts.values()),
        classes=len(declared_classes),
        empty_images=empty,
        multilabel_images=multilabel,
        images_per_class=dict(sorted(image_counts.items())),
        boxes_per_class=dict(sorted(box_counts.items())),
        box_area=area,
        cooccurrence={key: dict(sorted(value.items())) for key, value in sorted(pairs.items())},
        imbalance_ratio=imbalance,
        quality=quality_summary,
        present_classes=present_classes,
        missing_classes=missing_classes,
        box_imbalance_ratio=box_imbalance,
        failed_quality_images=failed_quality,
    )


_DEFAULT_WEIGHTS = {"quality": 0.4, "rarity": 0.3, "coverage": 0.2, "boxes": 0.1}
_ALLOWED_WEIGHT_KEYS = frozenset(_DEFAULT_WEIGHTS)


def _validate_weights(weights: Optional[Mapping[str, float]]) -> dict[str, float]:
    """Validate and normalize component weights for :func:`score`."""
    if weights is None:
        return dict(_DEFAULT_WEIGHTS)
    validators.validate_type(weights, Mapping, "weights")
    if not weights:
        raise ValueError("weights must not be empty")

    active: dict[str, float] = {}
    for key, raw_value in weights.items():
        if key not in _ALLOWED_WEIGHT_KEYS:
            raise ValueError("weights must contain only quality, rarity, coverage, or boxes")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError(f"weight '{key}' must be a real number, not {type(raw_value).__name__}")
        value = float(raw_value)
        if math.isnan(value) or math.isinf(value) or value < 0:
            raise ValueError(f"weight '{key}' must be a finite non-negative number")
        active[key] = value

    if sum(active.values()) <= 0:
        raise ValueError("weights must sum to a positive number")
    return active


def score(
    data: Detections,
    *,
    weights: Optional[Mapping[str, float]] = None,
    verbose: bool = False,
) -> MiningScores:
    """Calculate normalized information and quality scores for every image.

    Combines four distinct information signals into a composite score in [0.0, 1.0]:
    - `quality`: visual fidelity (brightness, sharpness, contrast, entropy, low noise, exposure).
    - `rarity`: presence of infrequent class labels across the dataset.
    - `coverage`: fraction of classes present in the dataset represented in the image.
    - `boxes`: normalized annotation density and object count.

    Parameters
    ----------
    data : Detections
        Materialized detection collection.
    weights : Mapping[str, float], optional
        Relative weighting for each component. Supported keys are `'quality'`,
        `'rarity'`, `'coverage'`, and `'boxes'`. Passing `{}` raises `ValueError`.
        Default: `{'quality': 0.4, 'rarity': 0.3, 'coverage': 0.2, 'boxes': 0.1}`.

    verbose : bool, default=False
        Display throttled progress for processing stages; disabled by default.

    Returns
    -------
    MiningScores
        Structured scores container holding normalized composite scores and
        per-component details for every image.

    Raises
    ------
    TypeError
        If `data` is not an instance of `Detections`, or `weights` / values have invalid types.
    ValueError
        If `weights` is empty, contains invalid keys, NaN/inf, negative values, or sums to zero.

    Examples
    --------
    >>> from klygo import datasets, mining
    >>> data = datasets.load("artifacts/datasets/flat_data", format="yolo", verbose=False)
    >>> scores = mining.score(data, weights={"quality": 0.6, "rarity": 0.4})
    >>> utils.tabulate(scores.to_dict())
    >>> print("Best score:", max(scores.values))
    """
    validators.validate_type(verbose, bool, "verbose")
    _require(data)

    active = _validate_weights(weights)

    if not len(data):
        return MiningScores(values=[], details=[], weights=active, identities=[])

    # 1. Compute global class frequencies for rarity and coverage calculation
    image_counts = Counter(label for frame in data for label in set(frame.labels))
    present_classes = len(image_counts)

    # 2. Extract and normalize visual quality components onto [0.0, 1.0]
    use_quality = active.get("quality", 0.0) > 0
    quality_rows = _quality(data, verbose=verbose) if use_quality else [{} for _ in data]
    bright = _normalize([row.get("brightness", 0.0) for row in quality_rows])
    sharp = _normalize([row.get("sharpness", 0.0) for row in quality_rows])
    contrast = _normalize([row.get("contrast", 0.0) for row in quality_rows])
    entropy = _normalize([row.get("entropy", 0.0) for row in quality_rows])
    noise = [0.0] if len(data) == 1 else _normalize([row.get("noise", 0.0) for row in quality_rows])
    exposure = _normalize([row.get("exposure", 0.0) for row in quality_rows])
    box_values = _normalize([float(len(frame)) for frame in data])

    # 3. Assemble per-frame component breakdowns
    details: list[dict[str, float]] = []
    for index, frame in enumerate(_iter_progress(data, verbose, "Scoring images")):
        labels = set(frame.labels)
        rarity = float(np.mean([1.0 / image_counts[label] for label in labels])) if labels else 0.0
        quality_value = (
            (
                bright[index]
                + sharp[index]
                + contrast[index]
                + entropy[index]
                + (1.0 - noise[index])
                + exposure[index]
            )
            / 6.0
            if use_quality
            else 0.0
        )

        details.append({
            "quality": quality_value,
            "rarity": rarity,
            "coverage": (len(labels) / present_classes) if present_classes else 0.0,
            "boxes": box_values[index],
        })

    # Normalize rarity across all images
    rarity_norm = _normalize([row["rarity"] for row in details])
    for index, value in enumerate(rarity_norm):
        details[index]["rarity"] = value

    # 4. Compute weighted composite score per frame
    total = sum(active.values())
    values = [
        sum(active.get(key, 0.0) * row[key] for key in _ALLOWED_WEIGHT_KEYS) / total
        for row in details
    ]

    return MiningScores(
        values=values,
        details=details,
        weights=active,
        identities=[[frame.source_id, frame.frame_index] for frame in data],
    )


def compare(
    before: Detections,
    after: Detections,
    *,
    quality: bool = False,
    verbose: bool = False,
) -> MiningComparison:
    """Compare image, box, class, and imbalance deltas between two dataset states.

    Parameters
    ----------
    before : Detections
        Original reference dataset before filtering or curation.
    after : Detections
        Modified dataset resulting from selection, deduplication, or balancing.
    quality : bool, default=False
        Whether to re-compute visual image quality metrics for both datasets.

    verbose : bool, default=False
        Display throttled progress for processing stages; disabled by default.

    Returns
    -------
    MiningComparison
        Comparative report containing statistics for both datasets and their
        numeric differences (`after - before`).

    Raises
    ------
    TypeError
        If either `before` or `after` is not a `Detections` collection.

    Examples
    --------
    >>> from klygo import datasets, mining
    >>> raw = datasets.load("artifacts/datasets/raw", format="yolo", verbose=False)
    >>> curated = mining.select(raw, method="weighted", limit=100)
    >>> comparison = mining.compare(raw, curated)
    >>> utils.tabulate(comparison.to_dict())
    >>> print("Removed images:", comparison.changes["images"])
    """
    validators.validate_type(verbose, bool, "verbose")
    first = stats(before, quality=quality, verbose=verbose)
    second = stats(after, quality=quality, verbose=verbose)

    changes = {
        "images": float(second.images - first.images),
        "boxes": float(second.boxes - first.boxes),
        "classes": float(second.classes - first.classes),
        "imbalance_ratio": second.imbalance_ratio - first.imbalance_ratio,
    }

    return MiningComparison(before=first, after=second, changes=changes)


__all__ = [
    "MiningStats",
    "MiningScores",
    "MiningComparison",
    "stats",
    "score",
    "compare",
]
