"""Rebuild existing dataset partitions with new ratios."""

from os import PathLike
from pathlib import Path

from klygo.validators.datasets import Repartition

from ._partitioning import execute_partition


def repartition(
    source: str | PathLike[str],
    target: str | PathLike[str],
    ratios: tuple[float, ...],
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Flatten existing splits and create a new deterministic partition.

    Parameters are identical to :func:`klygo.datasets.partition`. Source
    split paths are flattened safely so repeated partitioning does not create
    nested split structures.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Partitioned dataset directory or supported archive.
    target : str or os.PathLike[str]
        Destination directory or ``.zip`` archive.
    ratios : tuple[float, ...]
        Two or more split ratios whose sum is one.
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
        If the source dataset or its images directory is missing.
    FileExistsError
        If the target exists and ``overwrite`` is false.
    ValueError
        If the ratios are invalid.

    Examples
    --------
    >>> from klygo import datasets
    >>> output = datasets.repartition("old", "new", (0.7, 0.2, 0.1))
    """
    params = Repartition(source, target, ratios, overwrite, verbose)
    return execute_partition(
        params.source,
        params.target,
        params.ratios,
        overwrite=params.overwrite,
        verbose=params.verbose,
    )


__all__ = ["repartition"]
