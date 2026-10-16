"""Inspect, restructure, relabel, and generate datasets.

Colab tutorial placeholder (replace with a Klygo datasets notebook):
    https://colab.research.google.com/notebooks/intro.ipynb

Public API (8 functions):
    Inspection:
         1. get_dataset_info(...)     Report classes, files, and split statistics.

    Partitioning:
         2. partition(...)            Create train, validation, and test folders.
         3. repartition(...)          Rebuild an existing dataset with new ratios.
         4. unpartition(...)          Flatten partitioned data into one dataset.

    Combination and splitting:
         5. merge(...)                Combine datasets and reconcile class IDs.
         6. split(...)                Create class-based or ratio-based ZIP archives.

    Relabeling and generation:
         7. remap_classes(...)        Rename classes or remap numeric IDs.
         8. export(...)               Build a dataset from model predictions.

Uses klygo.files, klygo.config, klygo.archive, and klygo.media
for filesystem, metadata, archive, and image operations.
"""

from .export import export
from .get_dataset_info import get_dataset_info
from .merge import merge
from .partition import partition
from .remap_classes import remap_classes
from .repartition import repartition
from .split import split
from .unpartition import unpartition

__all__ = [
    "get_dataset_info",
    "partition",
    "repartition",
    "unpartition",
    "merge",
    "split",
    "remap_classes",
    "export",
]
