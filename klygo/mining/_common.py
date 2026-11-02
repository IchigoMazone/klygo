"""Internal utilities, shared report base, and common feature extractors for mining.

This module provides private helper functions and data structures shared
across the public mining submodules (`analysis`, `curation`, `duplicates`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from klygo import files, media, preprocessing, validators
from klygo.outputs import Detections


def _require(data: Detections) -> None:
    """Validate that input data is a materialized Detections collection.

    Parameters
    ----------
    data : Detections
        Detection dataset to validate.

    Raises
    ------
    TypeError
        If `data` is not an instance of `Detections`.
    ValueError
        If `data` is a lazy stream rather than a materialized collection.
    """
    validators.validate_type(data, Detections, "data")
    if data.is_stream:
        raise ValueError("mining requires materialized Detections")


class _InMemoryImage:
    """Small adapter exposing the LazyImage.to_pil() interface for memory sources."""

    def __init__(self, source: Any) -> None:
        self.source = source

    def to_pil(self, cache: bool = False):
        from PIL import Image

        if isinstance(self.source, Image.Image):
            return self.source.convert("RGB")
        array = np.asarray(self.source)
        if array.ndim == 2:
            return Image.fromarray(array).convert("RGB")
        if array.ndim == 3 and array.shape[2] in (3, 4):
            return Image.fromarray(array).convert("RGB")
        raise RuntimeError("in-memory image source must be a PIL image or HxW/HxWxC array")


def _image(frame: Any) -> Any:
    """Load image pixels from a path or the in-memory source retained by Detection.

    Raises RuntimeError only when neither a source path nor an image source exists.
    """
    if frame.source_path is not None:
        return media.load(frame.source_path, verbose=False)[0]

    source = getattr(frame, "_source", None)
    if source is None:
        raise RuntimeError("image pixels are unavailable: source_path and source are missing")
    if hasattr(source, "to_pil"):
        return source
    return _InMemoryImage(source)


def _subset(data: Detections, indices: Sequence[int], operation: str) -> Detections:
    """Construct an independent Detections subset preserving collection metadata.

    Parameters
    ----------
    data : Detections
        Original detection collection.
    indices : Sequence[int]
        Positional indices of frames to retain.
    operation : str
        Descriptive mining operation name recorded in result metadata.

    Returns
    -------
    Detections
        Independent collection containing only the selected frames.
    """
    metadata = dict(data.metadata)
    metadata["mining"] = {
        "operation": operation,
        "selected": len(indices),
        "total": len(data),
    }

    result = Detections(
        [data[index] for index in indices],
        metadata=metadata,
        names=data.names,
        packed=data.is_packed,
    )

    # Maintain array backend and device consistency for empty subsets
    if not indices:
        result.astype(data.dtype)
        if data.device.startswith("cuda"):
            result.cuda(int(data.device.split(":", 1)[1]))

    return result


def _normalize(values: Sequence[float]) -> list[float]:
    """Perform min-max normalization mapping numeric values to [0.0, 1.0].

    Parameters
    ----------
    values : Sequence[float]
        Sequence of numbers to scale.

    Returns
    -------
    list[float]
        Normalized values in [0.0, 1.0].
    """
    if not values:
        return []

    low, high = min(values), max(values)
    if high <= low:
        return [1.0 if high > 0 else 0.0 for _ in values]

    return [(float(value) - low) / (high - low) for value in values]


def _quality(data: Detections) -> list[dict[str, float]]:
    """Compute six classical image-quality metrics for every frame in a dataset.

    Calculates brightness, contrast, sharpness, entropy, noise, and exposure
    using lazy transforms from `klygo.preprocessing`.

    Parameters
    ----------
    data : Detections
        Materialized detection collection.

    Returns
    -------
    list[dict[str, float]]
        Per-frame dictionary of computed metric scores.
    """
    metrics = {
        "brightness": preprocessing.brightness_score(),
        "contrast": preprocessing.contrast_score(),
        "sharpness": preprocessing.sharpness_score(),
        "entropy": preprocessing.entropy_score(),
        "noise": preprocessing.noise_score(),
        "exposure": preprocessing.exposure_score(),
    }

    rows: list[dict[str, float]] = []
    for frame in data:
        image = _image(frame).to_pil(cache=False)
        rows.append({name: float(metric(image)) for name, metric in metrics.items()})

    return rows


def _save_report(report: "_Report", path: Any) -> Any:
    """Serialize a report to disk in JSON, Markdown, or CSV format.

    Parameters
    ----------
    report : _Report
        Report dataclass instance to save.
    path : str or os.PathLike[str]
        Target file path ending with `.json`, `.md`, or `.csv`.

    Returns
    -------
    pathlib.Path
        Resolved destination path.

    Raises
    ------
    ValueError
        If the file extension is not `.json`, `.md`, or `.csv`.
    """
    suffix = files.extension(path).lower()
    if suffix == ".json":
        value: Any = report.to_dict()
    elif suffix == ".md":
        value = report._markdown()
    elif suffix == ".csv":
        value = report._rows()
    else:
        raise ValueError("report path must end with .json, .md, or .csv")

    return files.save(path, value, overwrite=True, verbose=False)


class _Report:
    """Abstract base class providing shared serialization and rich terminal display.

    All report dataclasses (`MiningStats`, `MiningScores`, `DuplicateReport`,
    `ClusterResult`, `MiningComparison`) inherit from this class.
    """

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation of the report.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all fields formatted for serialization.
        """
        return asdict(self)  # type: ignore[arg-type]

    def save(self, path: Any) -> Any:
        """Save the report to disk as JSON, Markdown, or CSV.

        Parameters
        ----------
        path : str or os.PathLike[str]
            Target destination path. The file extension (`.json`, `.md`, `.csv`)
            determines the serialization format.

        Returns
        -------
        pathlib.Path
            Path to the saved file.
        """
        return _save_report(self, path)

    def _rows(self) -> list[dict[str, Any]]:
        """Return tabular rows for CSV serialization."""
        return [{"metric": key, "value": value} for key, value in self.to_dict().items()]

    def _markdown(self) -> str:
        """Generate a GitHub Flavored Markdown table representing the report."""
        rows = self._rows()
        columns = list(rows[0]) if rows else ["metric", "value"]

        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        lines.extend(
            "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
            for row in rows
        )
        return "\n".join(lines) + "\n"

    def show(self, console: Optional[Console] = None) -> None:
        """Render a formatted, colorful Rich table to the console.

        Parameters
        ----------
        console : rich.console.Console, optional
            Console instance used for rendering. Defaults to a new Console().
        """
        table = Table(title=self.__class__.__name__)
        rows = self._rows()
        columns = list(rows[0]) if rows else ["metric", "value"]

        for column in columns:
            table.add_column(column.replace("_", " ").title())

        for row in rows:
            table.add_row(*(str(row.get(column, "")) for column in columns))

        (console or Console()).print(table)


class _UnionFind:
    """Disjoint Set Union (Union-Find) with path compression for grouping items."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        """Find the root representative of the set containing `value`."""
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        """Merge the sets containing `first` and `second`."""
        root_a, root_b = self.find(first), self.find(second)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)


def _phash(frame: Any) -> tuple[str, np.ndarray]:
    """Calculate 63-bit perceptual hash (pHash) for an image frame.

    Converts image to 32x32 grayscale, computes 2D Discrete Cosine Transform (DCT),
    and thresholds the low-frequency 8x8 DCT coefficients against their median.

    Parameters
    ----------
    frame : Detection
        Detection frame containing source pixels.

    Returns
    -------
    tuple[str, np.ndarray]
        Binary hash string and boolean NumPy array of 63 bits.
    """
    array = np.asarray(
        _image(frame).to_pil().convert("L").resize((32, 32)),
        dtype=np.float32,
    )
    low = cv2.dct(array)[:8, :8].ravel()[1:]
    bits = low > np.median(low)
    return "".join("1" if value else "0" for value in bits), bits


def _feature_matrix(data: Detections, features: tuple[str, ...]) -> np.ndarray:
    """Extract and standardize a multi-modal feature matrix across all images.

    Supports combining HSV color histograms, HOG gradient descriptors, image
    quality metrics, bounding box geometries, and class indicator flags.

    Parameters
    ----------
    data : Detections
        Materialized detection collection.
    features : tuple of str
        Features to extract, chosen from {'histogram', 'hog', 'quality', 'boxes', 'classes'}.

    Returns
    -------
    np.ndarray
        Standardized (z-score normalized) 2D array of shape `(N, D)`.
    """
    allowed = {"histogram", "hog", "quality", "boxes", "classes"}
    if not features or set(features) - allowed:
        raise ValueError("features must use histogram, hog, quality, boxes, or classes")

    quality_metrics = (
        [
            preprocessing.brightness_score(),
            preprocessing.contrast_score(),
            preprocessing.sharpness_score(),
            preprocessing.entropy_score(),
            preprocessing.noise_score(),
            preprocessing.exposure_score(),
        ]
        if "quality" in features
        else []
    )

    labels = [data.names[key] for key in sorted(data.names)]
    rows: list[np.ndarray] = []
    block_widths: list[int] = []

    for frame in data:
        pil_image = _image(frame).to_pil(cache=False).convert("RGB")
        image = np.asarray(pil_image)
        parts: list[np.ndarray] = []

        # 1. HSV Color Histogram (64 bins)
        if "histogram" in features:
            hsv = cv2.cvtColor(cv2.resize(image, (64, 64)), cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).ravel()
            parts.append(cv2.normalize(hist, None).ravel())

        # 2. Histogram of Oriented Gradients (HOG): 8x8 cells * 9 bins = 576 features
        if "hog" in features:
            gray = cv2.cvtColor(cv2.resize(image, (64, 64)), cv2.COLOR_RGB2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            angle %= 180.0

            cells = []
            for top in range(0, 64, 8):
                for left in range(0, 64, 8):
                    cells.append(
                        np.histogram(
                            angle[top:top + 8, left:left + 8],
                            bins=9,
                            range=(0, 180),
                            weights=magnitude[top:top + 8, left:left + 8],
                        )[0]
                    )
            parts.append(np.concatenate(cells).astype(np.float32))

        # 3. Quality Metrics (6 features)
        if "quality" in features:
            parts.append(
                np.asarray([metric(pil_image) for metric in quality_metrics], dtype=np.float32)
            )

        # 4. Box Statistics (box count, mean area, max area) (3 features)
        if "boxes" in features:
            width, height = frame.image_size
            areas = [box.width * box.height / (width * height) for box in frame]
            parts.append(
                np.asarray(
                    [
                        len(frame),
                        np.mean(areas) if areas else 0.0,
                        max(areas, default=0.0),
                    ],
                    dtype=np.float32,
                )
            )

        # 5. Class Presence Flags (C features)
        if "classes" in features:
            present = set(frame.labels)
            parts.append(np.asarray([label in present for label in labels], dtype=np.float32))

        block_widths = [len(part) for part in parts]
        rows.append(np.concatenate(parts).astype(np.float32))

    matrix = np.vstack(rows)

    # Standardize each feature dimension, then normalize each modality block
    # by sqrt(width) so a 576-D HOG descriptor does not dominate 3-D box data
    # merely because it contributes many more dimensions.
    deviation = matrix.std(axis=0)
    deviation[deviation == 0] = 1.0
    matrix = (matrix - matrix.mean(axis=0)) / deviation
    offset = 0
    for width in block_widths:
        matrix[:, offset:offset + width] /= np.sqrt(float(width))
        offset += width
    return matrix


def _class_filter(data: Detections, classes: Sequence[str], match: str) -> list[int]:
    """Filter frame indices matching class criteria.

    Parameters
    ----------
    data : Detections
        Source collection.
    classes : Sequence[str]
        Target class names to filter by.
    match : {'any', 'all', 'only', 'exclude'}
        Matching mode.

    Returns
    -------
    list[int]
        Selected frame indices.
    """
    requested = set(classes)
    known = set(data.names.values())
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown classes: {sorted(unknown)}")

    if match not in {"any", "all", "only", "exclude"}:
        raise ValueError("match must be 'any', 'all', 'only', or 'exclude'")

    selected = []
    for index, frame in enumerate(data):
        present = set(frame.labels)
        accepted = bool(present & requested) if match == "any" else requested <= present
        if match == "only":
            accepted = bool(present) and present <= requested
        elif match == "exclude":
            accepted = not bool(present & requested)
        if accepted:
            selected.append(index)

    return selected


def _quota(
    data: Detections,
    quotas: Mapping[str, int],
    limit: Optional[int],
    *,
    count_by: str = "images",
    seed: Optional[int] = None,
) -> list[int]:
    """Greedily select frames to fulfill specified per-class target quotas.

    Parameters
    ----------
    data : Detections
        Source dataset.
    quotas : Mapping[str, int]
        Desired target count per class name.
    limit : int, optional
        Maximum total images to select.
    count_by : {'images', 'boxes'}, default='images'
        Whether quota targets count total images containing the class or total boxes.
    seed : int or None, default=None
        Optional seed used only to resolve equal-gain candidate ties. None preserves
        stable input-order tie-breaking.

    Returns
    -------
    list[int]
        Indices of chosen frames in deterministic order.
    """
    known = set(data.names.values())
    unknown = set(quotas) - known
    if unknown:
        raise ValueError(f"unknown classes: {sorted(unknown)}")

    if any(value < 0 for value in quotas.values()):
        raise ValueError("class quotas must be non-negative")

    counts: Counter[str] = Counter()
    remaining = set(range(len(data)))
    chosen: list[int] = []
    contributions = {
        index: (
            Counter(data[index].labels)
            if count_by == "boxes"
            else Counter(set(data[index].labels))
        )
        for index in range(len(data))
    }
    if seed is None:
        tie_rank = {index: index for index in range(len(data))}
    else:
        tie_rank = {
            int(index): rank
            for rank, index in enumerate(np.random.default_rng(seed).permutation(len(data)))
        }

    while remaining and (limit is None or len(chosen) < limit):
        deficits = {label: max(0, target - counts[label]) for label, target in quotas.items()}
        if not any(deficits.values()):
            break

        # Greedily pick the candidate maximizing deficit fulfillment.
        # Seeded tie ranks make undersampling reproducible without changing non-ties.
        candidate = max(
            remaining,
            key=lambda index: (
                sum(
                    min(deficits.get(label, 0), amount)
                    for label, amount in contributions[index].items()
                ),
                -tie_rank[index],
            ),
        )

        values = contributions[candidate]
        gain = sum(min(deficits.get(label, 0), amount) for label, amount in values.items())
        if gain <= 0:
            break

        chosen.append(candidate)
        counts.update(values)
        remaining.remove(candidate)

    return sorted(chosen)
