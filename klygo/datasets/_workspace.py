"""Temporary workspace and archive lifecycle helpers for datasets."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

from klygo import archive, files

from ._structure import find_dataset_root


@contextmanager
def open_dataset(source: str | Path, *, verbose: bool = False) -> Iterator[Path]:
    """Yield a dataset root from a directory or supported archive."""
    source_path = files.path(source)
    if not files.exists(source_path):
        raise FileNotFoundError(f"Source not found: {source_path}")
    if files.is_dir(source_path):
        yield find_dataset_root(source_path)
        return
    if not archive.is_archive(source_path):
        raise ValueError(f"source file is not a supported archive: {source_path}")

    with TemporaryDirectory(prefix="klygo-dataset-extract-") as directory:
        extraction_root = files.path(directory)
        archive.extract(source_path, extraction_root, overwrite=True, verbose=verbose)
        yield find_dataset_root(extraction_root)


@contextmanager
def dataset_workspace(prefix: str) -> Iterator[Path]:
    """Yield an empty dataset directory removed automatically on exit."""
    with TemporaryDirectory(prefix=f"klygo-{prefix}-") as directory:
        workspace = files.join(directory, "dataset")
        files.mkdir(workspace)
        yield workspace


def write_output(
    workspace: str | Path,
    target: str | Path,
    *,
    overwrite: bool,
    verbose: bool,
) -> Path:
    """Write a workspace to a directory or archive destination."""
    source = files.path(workspace)
    destination = files.path(target)
    if files.extension(destination).lower() == ".zip":
        archive.compress(
            source,
            destination,
            include_root=False,
            overwrite=overwrite,
            verbose=verbose,
        )
        return destination
    return files.copy(source, destination, overwrite=overwrite)


__all__ = ["dataset_workspace", "open_dataset", "write_output"]
