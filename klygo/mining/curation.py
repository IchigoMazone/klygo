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
from klygo import validators
from klygo.utils.progress import _iter_progress

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
        Total number of clusters generated.
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
    identities: list[list[Any]] = field(default_factory=list)

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
            "identities": self.identities,
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
    verbose: bool = False,
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
        - `'hog'`: 72-bin gradient orientation histogram capturing contours and textures.
        - `'quality'`: 6 classical quality metrics (sharpness, contrast, noise, etc.).
        - `'boxes'`: box count and relative area percentiles.
        - `'classes'`: binary indicator flags for label presence.
    clusters : int, default=100
        Target number of clusters. Must be between 1 and `len(data)`.
    method : {'kmeans'}, default='kmeans'
        Clustering algorithm. Currently only `'kmeans'` is supported.
    seed : int, default=42
        Random number generator seed for deterministic cluster initialization.

    verbose : bool, default=False
        Display throttled progress for processing stages; disabled by default.

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
    >>> utils.tabulate(result.to_dict())
    >>> print("Cluster 0 representative index:", result.representatives[0])
    """
    validators.validate_type(verbose, bool, "verbose")
    _require(data)

    if method != "kmeans":
        raise ValueError("method must be 'kmeans'")
    if not 1 <= clusters <= len(data):
        raise ValueError("clusters must be between 1 and the number of images")

    names = tuple(features)
    matrix = _feature_matrix(data, names, verbose=verbose)

    # Deterministic K-Means clustering with OpenCV
    cv2.setRNGSeed(int(seed))
    for _ in _iter_progress(range(1), verbose, "K-Means clustering", unit="stage"):
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
        clusters=clusters,
        seed=seed,
        feature_vectors=matrix,
        identities=[[frame.source_id, frame.frame_index] for frame in data],
    )


_CONFIDENCE_AGGREGATORS = {
    "mean": lambda values: float(sum(values) / len(values)),
    "min": lambda values: float(min(values)),
    "max": lambda values: float(max(values)),
    "median": lambda values: float(median(values)),
}


def select(
    data: Detections,
    *,
    method: str = "weighted",
    limit: Optional[int] = None,
    classes: Optional[Sequence[str]] = None,
    match: str = "any",
    threshold: float = 0.5,
    aggregate: str = "mean",
    order: str = "desc",
    min_boxes: int = 0,
    max_boxes: Optional[int] = None,
    min_area_ratio: float = 0.0,
    per_class: int | Mapping[str, int] = 1,
    scores: Optional[MiningScores] = None,
    cluster_result: Optional[ClusterResult] = None,
    per_cluster: int = 1,
    seed: int = 42,
    verbose: bool = False,
) -> Detections:
    """Select whole images from a dataset using specialized curation strategies.

    Supports 10 distinct selection strategies:
    - `'random'`: reproducible uniform random sampling without replacement using `seed`.
    - `'weighted'`: rank images by multi-criteria composite scores from :func:`score`.
    - `'quality'`: filter images exceeding a minimum quality score `threshold`.
    - `'confidence'`: filter and rank images by aggregated bounding-box confidence scores.
    - `'rare_first'`: prioritize images containing the least frequent class labels.
    - `'diversity'`: sample visual cluster representatives across K-Means groups.
    - `'boxes'`: filter images by annotation box count and relative area constraints.
    - `'quota'`: greedy selection to fulfill explicit target quotas per class.
    - `'coverage'`: greedy selection to maximize distinct class coverage.
    - `'class'`: filter images by specified class names with inclusion/exclusion rules.

    Parameters
    ----------
    data : Detections
        Materialized detection collection.
    method : {'weighted', 'random', 'quality', 'confidence', 'rare_first', 'diversity', 'boxes', 'quota', 'coverage', 'class'}, default='weighted'
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
        Minimum composite quality score required when `method='quality'`, or
        minimum aggregated box confidence score required when `method='confidence'`.
    aggregate : {'mean', 'min', 'max', 'median'}, default='mean'
        Aggregation function applied to per-frame box confidence scores when
        `method='confidence'`.
    order : {'desc', 'asc'}, default='desc'
        Ranking direction when `method='confidence'` (`'desc'` for highest
        confidence first, `'asc'` for lowest confidence first).
    min_boxes : int, default=0
        Minimum annotation box count required when `method='boxes'`.
    max_boxes : int, optional
        Maximum annotation box count allowed when `method='boxes'`.
    min_area_ratio : float, default=0.0
        Minimum relative box area required when `method='boxes'`.
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

    verbose : bool, default=False
        Display throttled progress for processing stages; disabled by default.

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
    validators.validate_type(verbose, bool, "verbose")
    _require(data)

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")

    method = method.lower()

    # 1. Class-based filtering
    if method == "class":
        if not classes:
            raise ValueError("classes must contain at least one class for class selection")
        indices = _class_filter(data, classes, match, verbose=verbose)

    # 2. Random uniform sampling
    elif method == "random":
        count = min(len(data), len(data) if limit is None else limit)
        indices = sorted(
            np.random.default_rng(seed).choice(len(data), count, replace=False).tolist()
        )

    # 3. Score-based ranking: quality, weighted, or rare_first
    elif method in {"quality", "weighted", "rare_first"}:
        report = scores or score(data, weights={"quality": 1.0} if method == "quality" else None, verbose=verbose)
        if len(report.values) != len(data):
            raise ValueError("scores length must match data length")
        if scores is not None and report.identities != [[frame.source_id, frame.frame_index] for frame in data]:
            raise ValueError("scores identities must match data")

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
        for index, frame in enumerate(_iter_progress(data, verbose, "Selecting images")):
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
        indices = _quota(data, quotas, limit, verbose=verbose)

    # 6. Cluster-based visual diversity sampling
    elif method == "diversity":
        if per_cluster <= 0:
            raise ValueError("per_cluster must be positive")
        if not len(data):
            indices = []
        else:
            result = cluster_result or cluster(
                data,
                clusters=min(len(data), limit or min(100, len(data))),
                verbose=verbose,
            )
            if len(result.labels) != len(data):
                raise ValueError("cluster_result length must match data length")
            if cluster_result is not None and result.identities and result.identities != [[frame.source_id, frame.frame_index] for frame in data]:
                raise ValueError("cluster_result identities must match data")

            buckets: list[list[int]] = []
            for key in sorted(result.groups):
                group = result.groups[key]
                if not group:
                    continue
                representative = result.representatives.get(key, group[0])
                ordered = (
                    [representative] + [index for index in group if index != representative]
                    if representative in group
                    else list(group)
                )
                buckets.append(ordered)

            if limit is None:
                chosen = [index for bucket in buckets for index in bucket[:per_cluster]]
            else:
                target = min(limit, len(data))
                if target == 0 or not buckets:
                    chosen = []
                elif len(buckets) > target:
                    # Evenly sample target distinct clusters and take each cluster's representative
                    selected_clusters = [int(i * len(buckets) / target) for i in range(target)]
                    chosen = [buckets[idx][0] for idx in selected_clusters]
                else:
                    # Round-robin across clusters from round 0 (representatives first, then extras/backfill)
                    chosen = []
                    offset = 0
                    while len(chosen) < target:
                        added = False
                        for bucket in buckets:
                            if offset < len(bucket):
                                chosen.append(bucket[offset])
                                added = True
                                if len(chosen) >= target:
                                    break
                        if not added:
                            break
                        offset += 1
            indices = sorted(chosen)

    # 7. Box confidence aggregation and ranking
    elif method == "confidence":
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0 for confidence selection")
        agg_key = aggregate.lower()
        aggregator = _CONFIDENCE_AGGREGATORS.get(agg_key)
        if aggregator is None:
            raise ValueError("aggregate must be 'mean', 'min', 'max', or 'median'")
        order_key = order.lower()
        if order_key not in {"desc", "asc"}:
            raise ValueError("order must be 'desc' or 'asc'")

        candidates: list[tuple[int, float]] = []
        for index, frame in enumerate(_iter_progress(data, verbose, "Selecting images")):
            frame_scores = frame.scores
            if not frame_scores:
                continue
            value = aggregator(frame_scores)
            if value >= threshold:
                candidates.append((index, value))

        ranked = sorted(
            candidates,
            key=lambda item: (-item[1] if order_key == "desc" else item[1], item[0]),
        )
        indices = [index for index, _ in ranked[:limit]]

    else:
        raise ValueError(
            "method must be class, quality, confidence, boxes, random, quota, rare_first, coverage, diversity, or weighted"
        )

    # Apply global limit if not already applied
    if limit is not None and method not in {
        "random", "weighted", "quality", "confidence", "rare_first", "quota", "coverage", "diversity"
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
    verbose: bool = False,
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
        Balancing policy. Currently both strategies select existing whole images.
    count_by : {'images', 'boxes'}, default='images'
        Whether quota targets are evaluated by total images containing the class
        or total bounding box count.
    seed : int, default=42
        RNG seed for deterministic tie-breaking.

    verbose : bool, default=False
        Display throttled progress for processing stages; disabled by default.

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
    validators.validate_type(verbose, bool, "verbose")
    _require(data)

    if strategy not in {"select", "undersample"}:
        raise ValueError("strategy must be 'select' or 'undersample'")
    if count_by not in {"images", "boxes"}:
        raise ValueError("count_by must be 'images' or 'boxes'")

    summary = stats(data, quality=False, cooccurrence=False, verbose=verbose)
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
    indices = _quota(data, quotas, None, count_by=count_by, seed=seed, verbose=verbose)
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
