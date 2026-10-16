"""Create train, validation, and test partitions."""

from os import PathLike
from pathlib import Path

from klygo.validators.datasets import Partition

from ._partitioning import execute_partition


def partition(
    source: str | PathLike[str],
    target: str | PathLike[str],
    ratios: tuple[float, ...],
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
    FileNotFoundError
        If the source or its ``images`` directory is missing.
    FileExistsError
        If the target exists and overwrite is disabled.
    ValueError
        If ratios do not form a valid partition.

    Examples
    --------
    >>> from klygo import datasets
    >>> output = datasets.partition("dataset", "partitioned", (0.8, 0.1, 0.1))
    """
    params = Partition(source, target, ratios, overwrite, verbose)
    return execute_partition(
        params.source,
        params.target,
        params.ratios,
        overwrite=params.overwrite,
        verbose=params.verbose,
    )


__all__ = ["partition"]
