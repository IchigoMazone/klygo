"""Load, inspect, restructure, relabel, and split datasets.

Colab tutorial:
    https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/datasets/Document_Datasets_Overview.ipynb

Public API (8 functions):
    Loading and inspection:
         1. load(...)                 Flatten and load all images and annotations.
         2. info(...)                 Report classes, files, and split statistics.

    Partitioning:
         3. partition(...)            Create train, validation, and test folders.
         4. repartition(...)          Rebuild an existing dataset with new ratios.
         5. unpartition(...)          Flatten partitioned data into one dataset.

    Combination and splitting:
         6. merge(...)                Combine datasets and reconcile class IDs.
         7. split(...)                Create class-based or ratio-based ZIP archives.

    Relabeling:
         8. remap_classes(...)        Rename classes or remap numeric IDs.

Uses klygo.files, klygo.config, klygo.archive, and klygo.media
for filesystem, metadata, archive, and image operations.
"""

from .info import get_dataset_info, info
from .load import load
from .merge import merge
from .partition import partition
from .remap_classes import remap_classes
from .repartition import repartition
from .split import split
from .unpartition import unpartition

__all__ = [
    "load",
    "info",
    "partition",
    "repartition",
    "unpartition",
    "merge",
    "split",
    "remap_classes",
]
