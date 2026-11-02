"""Dataset curation: deterministic clustering, diverse selection, and class balancing.

This submodule provides functions to group images into visual clusters via K-Means,
sample representative subsets using multiple strategies, and mitigate class
imbalance across object-detection datasets.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from klygo.outputs import Detections

from ._common import _Report, _class_filter, _feature_matrix, _quota, _require, _subset
from .analysis import MiningScores, score, stats


@dataclass
class ClusterResult(_Report):
    """Cluster assignments, visual groups, and central representatives.

    Attributes
    ----------
    labels : list[int]
        Cluster identifier assigned to each image in the dataset.
    groups : dict[int, list[int]]
        Mapping of cluster index to the list of image indices belonging to it.
    representatives : dict[int, int]
        Mapping of cluster index to the image index closest to the cluster center.
    features : tuple[str, ...]
        Active feature modalities used during feature extraction.
    clusters : int
        Number of non-empty clusters actually produced (which may be below the requested target).
    seed : int
        RNG seed used for deterministic K-Means initialization.
    feature_vectors : np.ndarray
        Standardized 2D NumPy array of extracted multi-modal feature vectors.

    Methods
    ----------
    show(console=None)
        Display a summary of clusters, sizes, and representatives in the terminal.
    to_dict()
        Return a JSON-serializable dictionary including cached feature vectors.
    save(path)
        Save the cluster assignments to disk as `.json`, `.md`, or `.csv`.
    """

    labels: list[int]
    groups: dict[int, list[int]]
    representatives: dict[int, int]
    features: tuple[str, ...]
    clusters: int
    seed: int
    feature_vectors: np.ndarray = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary including cached features."""
        return {
            "labels": self.labels,
            "groups": self.groups,
            "representatives": self.representatives,
            "features": list(self.features),
            "clusters": self.clusters,
            "seed": self.seed,
            "feature_vectors": self.feature_vectors.tolist(),
        }

    def _rows(self) -> list[dict[str, Any]]:
        return [
            {
                "cluster": key,
                "size": len(value),
                "representative": self.representatives[key],
            }
            for key, value in sorted(self.groups.items())
        ]


def cluster(
    data: Detections,
    *,
    features: Sequence[str] = ("histogram", "hog", "quality", "boxes", "classes"),
    clusters: int = 100,
    method: str = "kmeans",
    seed: int = 42,
) -> ClusterResult:
    """Cluster images into visual and semantic groups using classical feature vectors.

    Extracts multi-modal feature descriptors (color histograms, HOG gradients,
    image quality, box geometries, and class indicators), standardizes them,
    and applies deterministic OpenCV K-Means++ clustering.

    Parameters
    ----------
    data : Detections
        Materialized detection collection.
    features : Sequence[str], default=('histogram', 'hog', 'quality', 'boxes', 'classes')
        Feature modalities to extract. Allowed choices:
        - `'histogram'`: 64-bin HSV color histogram capturing overall lighting and palette.
        - `'hog'`: 576-dimensional gradient descriptor (8x8 cells, 9 bins per cell).
        - `'quality'`: 6 classical quality metrics (sharpness, contrast, noise, etc.).
        - `'boxes'`: box count, mean relative area, and maximum relative area.
        - `'classes'`: binary indicator flags for label presence.
    clusters : int, default=100
        Target number of clusters. Must be between 1 and `len(data)`.
    method : {'kmeans'}, default='kmeans'
        Clustering algorithm. Currently only `'kmeans'` is supported.
    seed : int, default=42
        Random number generator seed for deterministic cluster initialization.

    Returns
    -------
    ClusterResult
        Report containing cluster assignments, grouped image indices, and
        the closest representative image for each cluster.

    Raises
    ------
    TypeError
        If `data` is not an instance of `Detections`.
    ValueError
        If `method` is unsupported, `features` is empty or invalid, or `clusters`
        is not within `[1, len(data)]`.

    Notes
    -----
    The closest representative image for each cluster is computed via Euclidean
    distance in the standardized feature space, providing optimal candidates
    for diversity-driven dataset curation.

    Examples
    --------
    >>> from klygo import datasets, mining
    >>> data = datasets.load("artifacts/datasets/flat_data", format="yolo", verbose=False)
    >>> result = mining.cluster(data, clusters=10, seed=42)
    >>> result.show()
    >>> print("Cluster 0 representative index:", result.representatives[0])
    """
    _require(data)

    if method != "kmeans":
        raise ValueError("method must be 'kmeans'")
    if not 1 <= clusters <= len(data):
        raise ValueError("clusters must be between 1 and the number of images")

    names = tuple(features)
    matrix = _feature_matrix(data, names)

    # Deterministic K-Means clustering with OpenCV
    cv2.setRNGSeed(int(seed))
    _, labels, centers = cv2.kmeans(
        matrix.astype(np.float32),
        clusters,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4),
        1,
        cv2.KMEANS_PP_CENTERS,
    )

    values = labels.ravel().astype(int).tolist()
    groups: dict[int, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        groups[value].append(index)

    # Identify the representative image closest to each cluster center
    representatives = {
        key: min(
            indices,
            key=lambda index: (
                float(np.linalg.norm(matrix[index] - centers[key])),
                index,
            ),
        )
        for key, indices in groups.items()
    }

    return ClusterResult(
        labels=values,
        groups=dict(sorted(groups.items())),
        representatives=dict(sorted(representatives.items())),
        features=names,
        clusters=len(groups),
        seed=seed,
        feature_vectors=matrix,
    )


def select(
    data: Detections,
    *,
    method: str = "weighted",
    limit: Optional[int] = None,
    classes: Optional[Sequence[str]] = None,
    match: str = "any",
    threshold: float = 0.5,
    min_boxes: int = 0,
    max_boxes: Optional[int] = None,
    min_area_ratio: float = 0.0,
    per_class: int | Mapping[str, int] = 1,
    scores: Optional[MiningScores] = None,
    cluster_result: Optional[ClusterResult] = None,
    per_cluster: int = 1,
    seed: int = 42,
) -> Detections:
    """Select whole images from a dataset using specialized curation strategies.

    Supports 9 distinct selection strategies:
    - `'random'`: reproducible uniform random sampling without replacement using `seed`.
    - `'weighted'`: rank images by multi-criteria composite scores from :func:`score`.
    - `'quality'`: filter images exceeding a minimum quality score `threshold`.
    - `'rare_first'`: prioritize images containing the least frequent class labels.
    - `'diversity'`: sample visual cluster representatives across K-Means groups.
    - `'boxes'`: filter images by annotation box count and relative area constraints.
    - `'quota'`: greedy selection to fulfill explicit target quotas per class.
    - `'coverage'`: alias of the greedy per-class quota selection used by `'quota'`.
    - `'class'`: filter images by specified class names with inclusion/exclusion rules.

    Parameters
    ----------
    data : Detections
        Materialized detection collection.
    method : {'weighted', 'random', 'quality', 'rare_first', 'diversity', 'boxes', 'quota', 'coverage', 'class'}, default='weighted'
        Selection strategy to execute.
    limit : int, optional
        Maximum number of images to return. If None, returns all matching frames.
    classes : Sequence[str], optional
        Target class names required when `method='class'`.
    match : {'any', 'all', 'only', 'exclude'}, default='any'
        Matching condition for class filtering:
        - `'any'`: image must contain at least one requested class.
        - `'all'`: image must contain all requested classes.
        - `'only'`: image must contain only requested classes (no other classes).
        - `'exclude'`: image must not contain any of the requested classes.
    threshold : float, default=0.5
        Minimum composite quality score required when `method='quality'`.
    min_boxes : int, default=0
        Minimum annotation box count required when `method='boxes'`.
    max_boxes : int, optional
        Maximum annotation box count allowed when `method='boxes'`.
    min_area_ratio : float, default=0.0
        Minimum relative area required for the largest box in an image when `method='boxes'`.
    per_class : int or Mapping[str, int], default=1
        Target quota per class when `method='quota'` or `method='coverage'`.
    scores : MiningScores, optional
        Pre-computed scores instance to avoid re-evaluating metrics.
    cluster_result : ClusterResult, optional
        Pre-computed clustering instance used when `method='diversity'`.
    per_cluster : int, default=1
        Number of representative samples to draw per cluster in diversity selection.
    seed : int, default=42
        RNG seed for reproducible random or diversity sampling.

    Returns
    -------
    Detections
        New, independent detection collection containing only the selected frames.

    Raises
    ------
    TypeError
        If `data` is not an instance of `Detections`.
    ValueError
        If `method` is unknown, `limit` is negative, or required arguments are missing.

    Examples
    --------
    >>> from klygo import datasets, mining
    >>> data = datasets.load("artifacts/datasets/flat_data", format="yolo", verbose=False)
    >>> # 1. Random sampling
    >>> rand_subset = mining.select(data, method="random", limit=50, seed=42)
    >>> # 2. Multi-criteria weighted selection
    >>> top_subset = mining.select(data, method="weighted", limit=50)
    >>> # 3. Class exclusion filtering
    >>> no_cars = mining.select(data, method="class", classes=["car"], match="exclude")
    """
    _require(data)

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")

    method = method.lower()

    # 1. Class-based filtering
    if method == "class":
        if not classes:
            raise ValueError("classes must contain at least one class for class selection")
        indices = _class_filter(data, classes, match)

    # 2. Random uniform sampling
    elif method == "random":
        count = min(len(data), len(data) if limit is None else limit)
        indices = sorted(
            np.random.default_rng(seed).choice(len(data), count, replace=False).tolist()
        )

    # 3. Score-based ranking: quality, weighted, or rare_first
    elif method in {"quality", "weighted", "rare_first"}:
        report = scores or score(data, weights={"quality": 1.0} if method == "quality" else None)
        if len(report.values) != len(data):
            raise ValueError("scores length must match data length")

        key = "rarity" if method == "rare_first" else None
        ranked = sorted(
            range(len(data)),
            key=lambda index: (
                -(report.details[index][key] if key else report.values[index]),
                index,
            ),
        )
        if method == "quality":
            ranked = [index for index in ranked if report.values[index] >= threshold]
        indices = ranked[:limit]

    # 4. Box geometry and count constraints
    elif method == "boxes":
        indices = []
        for index, frame in enumerate(data):
            width, height = frame.image_size
            areas = [box.width * box.height / (width * height) for box in frame]
            if (
                len(frame) >= min_boxes
                and (max_boxes is None or len(frame) <= max_boxes)
                and (min_area_ratio <= 0 or max(areas, default=0) >= min_area_ratio)
            ):
                indices.append(index)

    # 5. Class quota and coverage optimization
    elif method in {"quota", "coverage"}:
        quotas = (
            {label: int(per_class) for label in data.names.values()}
            if isinstance(per_class, int)
            else {str(k): int(v) for k, v in per_class.items()}
        )
        indices = _quota(data, quotas, limit)

    # 6. Cluster-based visual diversity sampling
    elif method == "diversity":
        if per_cluster <= 0:
            raise ValueError("per_cluster must be positive")
        if limit == 0:
            return _subset(data, [], f"select:{method}")

        result = cluster_result or cluster(
            data,
            clusters=min(len(data), limit or min(100, len(data))),
            seed=seed,
        )
        if len(result.labels) != len(data):
            raise ValueError("cluster_result length must match data length")

        rng = np.random.default_rng(seed)
        cluster_keys = sorted(result.groups)
        rng.shuffle(cluster_keys)

        # Round-robin across clusters: take one item from every cluster before
        # taking second/third items, so a small global limit retains broad coverage.
        per_group = {
            key: [result.representatives[key]]
            + [
                index for index in result.groups[key]
                if index != result.representatives[key]
            ][:per_cluster - 1]
            for key in cluster_keys
        }
        chosen = []
        for position in range(per_cluster):
            for key in cluster_keys:
                candidates = per_group[key]
                if position < len(candidates):
                    chosen.append(candidates[position])
                    if limit is not None and len(chosen) >= limit:
                        break
            if limit is not None and len(chosen) >= limit:
                break
        indices = sorted(chosen)

    else:
        raise ValueError(
            "method must be class, quality, boxes, random, quota, rare_first, coverage, diversity, or weighted"
        )

    # Apply global limit if not already applied
    if limit is not None and method not in {
        "random", "weighted", "quality", "rare_first", "quota", "coverage", "diversity"
    }:
        indices = indices[:limit]

    result = _subset(data, sorted(indices), f"select:{method}")

    # Record unmet quotas in metadata for auditability
    if method in {"quota", "coverage"}:
        quotas = (
            {label: int(per_class) for label in data.names.values()}
            if isinstance(per_class, int)
            else {str(k): int(v) for k, v in per_class.items()}
        )
        actual = Counter(label for frame in result for label in set(frame.labels))
        result.metadata["mining"]["unmet_quotas"] = {
            label: target - actual[label]
            for label, target in quotas.items()
            if actual[label] < target
        }

    return result


def balance(
    data: Detections,
    *,
    target: str | int = "median",
    strategy: str = "select",
    count_by: str = "images",
    seed: int = 42,
    keep_background: bool = True,
) -> Detections:
    """Mitigate dataset class imbalance without duplicating frames.

    Resolves target quota numbers per class and uses greedy quota selection
    to retain a balanced subset across all present classes.

    Parameters
    ----------
    data : Detections
        Materialized detection collection.
    target : {'min', 'median', 'mean'} or int, default='median'
        Target representation quota for each class:
        - `'min'`: bring all classes down to the frequency of the least common class.
        - `'median'`: target the median frequency across all classes.
        - `'mean'`: target the arithmetic mean class frequency.
        - int: explicit maximum quota limit per class.
    strategy : {'select', 'undersample'}, default='select'
        Tie-breaking policy. 'select' preserves input order for equal-gain choices;
        'undersample' uses reproducible seeded tie-breaking.
    count_by : {'images', 'boxes'}, default='images'
        Whether quota targets are evaluated by total images containing the class
        or total bounding box count.
    seed : int, default=42
        RNG seed used for tie-breaking when strategy='undersample'.
    keep_background : bool, default=True
        Preserve images with no boxes. Background images do not contribute to class quotas.

    Returns
    -------
    Detections
        New, independent detection collection with reduced class imbalance.

    Raises
    ------
    TypeError
        If `data` is not an instance of `Detections`.
    ValueError
        If `strategy`, `count_by`, or `target` is invalid.

    Notes
    -----
    Because object detection annotations frequently contain multiple classes
    in a single image, exact equality across all classes is mathematically
    unachievable without image synthesis. :func:`balance` minimizes disparity
    while strictly preserving original image and annotation integrity.

    Examples
    --------
    >>> from klygo import datasets, mining
    >>> data = datasets.load("artifacts/datasets/flat_data", format="yolo", verbose=False)
    >>> balanced = mining.balance(data, target="median", count_by="images")
    >>> print(f"Balanced collection contains {len(balanced)} frames.")
    """
    _require(data)

    if strategy not in {"select", "undersample"}:
        raise ValueError("strategy must be 'select' or 'undersample'")
    if count_by not in {"images", "boxes"}:
        raise ValueError("count_by must be 'images' or 'boxes'")
    if not isinstance(keep_background, bool):
        raise TypeError("keep_background must be a boolean")

    summary = stats(data, quality=False, cooccurrence=False)
    counts = summary.images_per_class if count_by == "images" else summary.boxes_per_class
    values = list(counts.values())

    if isinstance(target, str):
        if target not in {"min", "median", "mean"}:
            raise ValueError("target must be an integer, 'min', 'median', or 'mean'")
        resolved = (
            int(
                min(values)
                if target == "min"
                else median(values)
                if target == "median"
                else np.mean(values)
            )
            if values
            else 0
        )
    else:
        resolved = int(target)

    if resolved < 0:
        raise ValueError("target must be non-negative")

    quotas = {label: min(resolved, value) for label, value in counts.items()}
    indices = _quota(
        data,
        quotas,
        None,
        count_by=count_by,
        seed=seed if strategy == "undersample" else None,
    )
    if keep_background:
        indices.extend(index for index, frame in enumerate(data) if len(frame) == 0)
    indices = sorted(set(indices))
    result = _subset(data, indices, f"balance:{strategy}")

    # Record actual versus target counts in metadata
    actual = Counter()
    for frame in result:
        actual.update(frame.labels if count_by == "boxes" else set(frame.labels))

    result.metadata["mining"]["unmet_quotas"] = {
        label: target_value - actual[label]
        for label, target_value in quotas.items()
        if actual[label] < target_value
    }

    return result


__all__ = [
    "ClusterResult",
    "cluster",
    "select",
    "balance",
]
