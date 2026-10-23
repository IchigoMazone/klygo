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


def set_output_metadata_path(metadata: dict, destination: str | Path) -> None:
    """Use a local path for directories and no machine-specific path for ZIPs."""
    output = files.path(destination)
    if files.extension(output).lower() == ".zip":
        metadata.pop("path", None)
    else:
        metadata["path"] = files.as_posix(files.resolve(output))


def publish_output(staging: Path, destination: Path, temporary_root: Path) -> None:
    """Install a finished output, restoring the previous one on rename failure."""
    backup = files.join(temporary_root, "previous")
    target_dest = files.join(files.resolve(files.parent(destination)), files.name(destination))
    if files.exists(destination):
        try:
            files.rename(destination, files.resolve(backup))
        except (PermissionError, OSError):
            files.move(destination, files.resolve(backup), overwrite=True)
    try:
        try:
            files.rename(staging, target_dest)
        except (PermissionError, OSError):
            files.move(staging, target_dest, overwrite=True)
    except BaseException:
        if files.exists(backup):
            try:
                files.rename(backup, target_dest)
            except (PermissionError, OSError):
                files.move(backup, target_dest, overwrite=True)
        raise


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
    if files.exists(destination) and not overwrite:
        raise FileExistsError(f"target already exists: {destination}")
    files.mkdir(files.parent(destination))
    with TemporaryDirectory(prefix=f".{files.name(destination)}-", dir=files.parent(destination)) as directory:
        temporary_root = files.path(directory)
        if files.extension(destination).lower() == ".zip":
            staging = files.join(temporary_root, "dataset.zip")
            archive.compress(
                source,
                staging,
                include_root=False,
                overwrite=True,
                verbose=verbose,
            )
        else:
            staging = files.join(temporary_root, "dataset")
            files.copy(source, staging)
        publish_output(staging, destination, temporary_root)
    return destination


__all__ = ["dataset_workspace", "open_dataset", "publish_output", "set_output_metadata_path", "write_output"]
