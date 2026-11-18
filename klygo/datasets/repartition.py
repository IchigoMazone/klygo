"""Rebuild existing dataset partitions with new ratios."""

from os import PathLike
from pathlib import Path

from klygo.validators.datasets import Repartition

from ._partitioning import execute_partition
from ._structure import validate_dataset_format


def repartition(
    source: str | PathLike[str],
    target: str | PathLike[str],
    ratios: tuple[float, ...],
    format: str = "yolo",
    overwrite: bool = False,
    verbose: bool = True,
    shuffle: bool = True,
    seed: int = 42,
) -> None:
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
    format : str, default='yolo'
        Dataset format to partition. Currently only ``'yolo'`` is supported.
    overwrite : bool, default=False
        Replace an existing target.
    verbose : bool, default=True
        Display archive progress.
    shuffle : bool, default=True
        Whether to shuffle samples before assigning to partitions.
    seed : int, default=42
        Random seed used when shuffling.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``format`` is not a string.
    ValueError
        If ``format`` is not supported or the ratios are invalid.
    FileNotFoundError
        If the source dataset or its images directory is missing.
    FileExistsError
        If the target exists and ``overwrite`` is false.

    Examples
    --------
    >>> from klygo import datasets
    >>> datasets.repartition("old", "new", (0.7, 0.2, 0.1), shuffle=True, seed=42)
    """
    validate_dataset_format(format)
    params = Repartition(source, target, ratios, overwrite, verbose, shuffle=shuffle, seed=seed)
    execute_partition(
        params.source,
        params.target,
        params.ratios,
        overwrite=params.overwrite,
        verbose=params.verbose,
        shuffle=params.shuffle,
        seed=params.seed,
    )


__all__ = ["repartition"]
