"""Create train, validation, and test partitions."""

from os import PathLike
from pathlib import Path

from klygo.validators.datasets import Partition

from ._partitioning import execute_partition
from ._structure import validate_dataset_format


def partition(
    source: str | PathLike[str],
    target: str | PathLike[str],
    ratios: tuple[float, ...],
    format: str = "yolo",
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Partition a YOLO dataset reproducibly.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Dataset directory or supported archive.
    target : str or os.PathLike[str]
        Destination directory or ``.zip`` archive.
    ratios : tuple of float
        One value selects the train ratio and divides the remainder equally.
        Two values select train and validation, using the remainder for test.
        Three values must sum to one.
    format : str, default='yolo'
        Dataset format to partition. Currently only ``'yolo'`` is supported.
    overwrite : bool, default=False
        Replace an existing target.
    verbose : bool, default=True
        Display archive progress.

    Returns
    -------
    pathlib.Path
        Created dataset path.

    Raises
    ------
    TypeError
        If ``format`` is not a string.
    ValueError
        If ``format`` is not supported or ratios do not form a valid partition.
    FileNotFoundError
        If the source or its ``images`` directory is missing.
    FileExistsError
        If the target exists and overwrite is disabled.

    Examples
    --------
    >>> from klygo import datasets
    >>> output = datasets.partition("dataset", "partitioned", (0.8, 0.1, 0.1))
    """
    validate_dataset_format(format)
    params = Partition(source, target, ratios, overwrite, verbose)
    return execute_partition(
        params.source,
        params.target,
        params.ratios,
        overwrite=params.overwrite,
        verbose=params.verbose,
    )


__all__ = ["partition"]
