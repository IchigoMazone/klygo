"""Dataset inspection, transformation, partitioning, and model export.

Public APIs (8 functions):
    1. get_dataset_info  2. partition  3. repartition  4. unpartition
    5. merge             6. split      7. remap_classes 8. export

Dataset filesystem work is delegated to :mod:`klygo.files`, structured
metadata to :mod:`klygo.config`, archives to :mod:`klygo.archive`, and image
input/output to :mod:`klygo.media`.
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
