"""Internal helpers shared by the public archive modules."""

from pathlib import Path
from typing import Optional, Tuple, Union

import klygo.files as files
from klygo.archive.backend import ArchiveBackend, get_backend


PathLike = Union[str, Path]


def resolve_backend(
    archive_path: PathLike,
    *,
    format_hint: Optional[str] = None,
) -> Tuple[Path, ArchiveBackend]:
    """Normalize an archive path and select its backend once."""
    path = files.path(archive_path)
    return path, get_backend(path, format_hint=format_hint)


def ensure_writable(path: Path, *, overwrite: bool, parameter: str) -> None:
    """Enforce the common destination overwrite policy."""
    if files.exists(path) and not overwrite:
        raise FileExistsError(
            f"{parameter} already exists: {path}. Use overwrite=True."
        )


__all__ = ["PathLike", "ensure_writable", "resolve_backend"]
