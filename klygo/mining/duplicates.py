"""Cryptographic and perceptual duplicate image detection and filtering.

This submodule detects identical duplicate images via MD5 hashing and near-duplicate
images via perceptual Discrete Cosine Transform (DCT) pHash similarity.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import os
from typing import Any

import numpy as np

from klygo import files
from klygo.outputs import Detections
from klygo import validators
from klygo.utils.progress import _iter_progress

from ._common import _Report, _UnionFind, _phash, _require, _subset
from .analysis import score


@dataclass
class DuplicateReport(_Report):
    """Detailed summary describing exact and perceptual duplicate image groups.

    Attributes
    ----------
    groups : list[list[int]]
        List of duplicate groups, where each group is a list of frame indices
        determined to be exact or near-duplicates.
    exact_groups : list[list[int]]
        Subset of groups containing bit-for-bit identical files (matching MD5).
    similarities : dict[str, float]
        Pairwise perceptual similarity scores between image index pairs `"i:j"`.
    hashes : list[str]
        Computed 63-bit perceptual hash binary strings for all frames in the dataset.
    total_images : int
        Total number of images inspected.

    Properties
    ----------
    duplicate_images : int
        Total count of redundant images eligible for removal across all groups
        (`sum(len(group) - 1 for group in groups)`).

    Methods
    ----------
    show(console=None)
        Display the duplicate groups and removal summary in the terminal.
    to_dict()
        Return a JSON-serializable dictionary including the removal count.
    save(path)
        Save the duplicate report to disk as `.json`, `.md`, or `.csv`.
    """

    groups: list[list[int]]
    exact_groups: list[list[int]]
    similarities: dict[str, float]
    hashes: list[str]
    total_images: int

    @property
    def duplicate_images(self) -> int:
        """Return the number of redundant duplicate images eligible for removal."""
        return sum(len(group) - 1 for group in self.groups)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary including the removal count."""
        value = asdict(self)
        value["duplicate_images"] = self.duplicate_images
        return value


def find_duplicates(
    data: Detections,
    *,
    exact: bool = True,
    method: str = "phash",
    threshold: float = 0.95,
    verbose: bool = False,
) -> DuplicateReport:
    """Detect exact and near-duplicate images across a detection dataset.

    Identifies duplicate image clusters using two cooperative passes:
    1. Bit-for-bit file hashing via MD5 (when `exact=True`).
    2. Perceptual hashing (pHash) based on low-frequency 2D DCT coefficients
       with normalized Hamming similarity.

    Parameters
    ----------
    data : Detections
        Materialized detection collection.
    exact : bool, default=True
        Whether to check for identical file checksums before computing pHash.
    method : {'phash'}, default='phash'
        Perceptual hash algorithm to use. Currently only `'phash'` is supported.
    threshold : float, default=0.95
        Minimum normalized perceptual similarity in [0.0, 1.0] required to group
        two images as near-duplicates. Default 0.95 corresponds to <= 3 bit differences.

    verbose : bool, default=False
        Display throttled progress for processing stages; disabled by default.

    Returns
    -------
    DuplicateReport
        Comprehensive report containing duplicate groups, exact groups, pairwise
        similarity scores, and total removable duplicate counts.

    Raises
    ------
    TypeError
        If `data` is not an instance of `Detections`.
    ValueError
        If `method` is unsupported or `threshold` is outside [0.0, 1.0].

    Notes
    -----
    This function performs non-destructive analysis and does not alter the
    dataset. To remove duplicate images, call :func:`deduplicate`.

    Examples
    --------
    >>> from klygo import datasets, mining
    >>> data = datasets.load("artifacts/datasets/flat_data", format="yolo", verbose=False)
    >>> report = mining.find_duplicates(data, threshold=0.95)
    >>> utils.tabulate(report.to_dict())
    >>> print(f"Removable duplicates: {report.duplicate_images}")
    """
    validators.validate_type(verbose, bool, "verbose")
    _require(data)

    if method != "phash":
        raise ValueError("method must be 'phash'")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    union = _UnionFind(len(data))
    exact_map: dict[str, list[int]] = defaultdict(list)

    total = len(data)
    workers = min(8, os.cpu_count() or 4, max(1, total))

    # 1. Exact pass: group identical source files using cryptographic hashes
    if exact:
        if total >= 16 and not verbose:
            def _hash_one(frame: Any) -> str | None:
                return files.hash(frame.source_path) if frame.source_path is not None else None

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for index, digest in enumerate(pool.map(_hash_one, data)):
                    if digest is not None:
                        exact_map[digest].append(index)
        else:
            for index, frame in enumerate(_iter_progress(data, verbose, "Exact image hashes")):
                if frame.source_path is not None:
                    exact_map[files.hash(frame.source_path)].append(index)
        for group in exact_map.values():
            for index in group[1:]:
                union.union(group[0], index)

    # 2. Perceptual pass: compute 63-bit DCT pHash for each frame
    if total >= 16 and not verbose:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            computed = list(pool.map(_phash, data))
    else:
        computed = [_phash(frame) for frame in _iter_progress(data, verbose, "Perceptual image hashes")]
    similarities: dict[str, float] = {}

    # Vectorized pairwise comparison across frames
    if len(computed) >= 2:
        bits = np.stack([item[1] for item in computed]).astype(np.float32)
        n_bits = bits.shape[1]
        matches = bits @ bits.T + (1.0 - bits) @ (1.0 - bits).T
        min_matches = threshold * n_bits - 1e-4
        first_idx, second_idx = np.where(np.triu(matches >= min_matches, k=1))
        for first, second in zip(first_idx.tolist(), second_idx.tolist()):
            similarity = float(1.0 - np.mean(computed[first][1] != computed[second][1]))
            if similarity >= threshold:
                union.union(first, second)
                similarities[f"{first}:{second}"] = similarity

    # 3. Assemble disjoint sets into grouped duplicate clusters
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(data)):
        grouped[union.find(index)].append(index)

    groups = sorted(
        (group for group in grouped.values() if len(group) > 1),
        key=lambda value: value[0],
    )
    exact_groups = sorted(
        (group for group in exact_map.values() if len(group) > 1),
        key=lambda value: value[0],
    )

    return DuplicateReport(
        groups=groups,
        exact_groups=exact_groups,
        similarities=similarities,
        hashes=[value[0] for value in computed],
        total_images=len(data),
    )


def deduplicate(
    data: Detections,
    *,
    keep: str = "best",
    threshold: float = 0.95,
    verbose: bool = False,
) -> Detections:
    """Return an independent detection dataset with duplicate images filtered out.

    Identifies duplicate clusters using :func:`find_duplicates` and retains
    exactly one chosen representative per duplicate group according to the
    `keep` strategy.

    Parameters
    ----------
    data : Detections
        Materialized detection collection.
    keep : {'best', 'first', 'quality', 'annotations'}, default='best'
        Disambiguation policy for choosing which frame to keep in each duplicate group:
        - `'best'`: maximize composite quality and rarity score, breaking ties by box count.
        - `'first'`: keep the frame with lowest positional index in the dataset.
        - `'quality'`: maximize visual image quality score (sharpness, contrast, entropy).
        - `'annotations'`: keep the frame with highest bounding box count.
    threshold : float, default=0.95
        Perceptual hash similarity threshold in [0.0, 1.0].

    verbose : bool, default=False
        Display throttled progress for processing stages; disabled by default.

    Returns
    -------
    Detections
        New, independent detection collection with redundant duplicates removed.

    Raises
    ------
    TypeError
        If `data` is not an instance of `Detections`.
    ValueError
        If `keep` is not one of `{'best', 'first', 'quality', 'annotations'}`.

    Notes
    -----
    The operation is whole-sample atomic: selected images are copied with all
    their original annotations intact. Unselected duplicate images are excluded.

    Examples
    --------
    >>> from klygo import datasets, mining
    >>> data = datasets.load("artifacts/datasets/flat_data", format="yolo", verbose=False)
    >>> unique = mining.deduplicate(data, keep="best", threshold=0.95)
    >>> print(f"Removed {len(data) - len(unique)} duplicate frames.")
    """
    validators.validate_type(verbose, bool, "verbose")
    _require(data)

    if keep not in {"best", "first", "quality", "annotations"}:
        raise ValueError("keep must be 'best', 'first', 'quality', or 'annotations'")

    report = find_duplicates(data, threshold=threshold, verbose=verbose)
    ranking = (
        score(data, weights={"quality": 1.0} if keep == "quality" else None, verbose=verbose)
        if keep in {"best", "quality"}
        else None
    )

    removed: set[int] = set()

    for group in report.groups:
        if keep == "first":
            chosen = group[0]
        elif keep == "annotations":
            chosen = max(group, key=lambda index: (len(data[index]), -index))
        else:
            assert ranking is not None
            chosen = max(
                group,
                key=lambda index: (ranking.values[index], len(data[index]), -index),
            )
        removed.update(set(group) - {chosen})

    return _subset(
        data,
        [index for index in range(len(data)) if index not in removed],
        "deduplicate",
    )


__all__ = [
    "DuplicateReport",
    "find_duplicates",
    "deduplicate",
]
