"""Rebuild existing dataset partitions with new ratios."""

from pathlib import Path

from klygo.validators.datasets import Repartition

from ._partitioning import execute_partition


def repartition(
    source: str | Path,
    target: str | Path,
    ratios: tuple[float, ...],
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Flatten existing splits and create a new deterministic partition.

    Parameters are identical to :func:`klygo.datasets.partition`. Source
    split paths are flattened safely so repeated partitioning does not create
    nested split structures.

    Returns
    -------
    pathlib.Path
        Created dataset path.

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
