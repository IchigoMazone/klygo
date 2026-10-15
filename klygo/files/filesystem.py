"""Filesystem status, traversal, and mutation operations."""

from __future__ import annotations

import os
import shutil
from fnmatch import fnmatch
from pathlib import Path
from typing import Generator, Iterable, List, Tuple

from klygo.validators import validate_type

from ._typing import PathInput


def exists(path: PathInput) -> bool:
    """Check whether a filesystem path exists.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Path to inspect or operate on.

    Returns
    -------
    bool
        Whether the path exists.

    See Also
    --------
    is_file
    is_dir

    Examples
    --------
    >>> from klygo import files
    >>> files.exists("README.md")
    True
    """
    return Path(path).exists()


def is_file(path: PathInput) -> bool:
    """Check whether a path identifies a regular file.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Path to inspect or operate on.

    Returns
    -------
    bool
        Whether the path is a regular file.

    See Also
    --------
    exists
    is_dir

    Examples
    --------
    >>> from klygo import files
    >>> files.is_file("README.md")
    True
    """
    return Path(path).is_file()


def is_dir(path: PathInput) -> bool:
    """Check whether a path identifies a directory.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Path to inspect or operate on.

    Returns
    -------
    bool
        Whether the path is a directory.

    See Also
    --------
    exists
    is_file

    Examples
    --------
    >>> from klygo import files
    >>> files.is_dir("klygo")
    True
    """
    return Path(path).is_dir()


def list_entries(path: PathInput = ".", pattern: str = "*", recursive: bool = False) -> List[Path]:
    """List matching files and directories below a directory.

    Both files and directories are included. Results are sorted case-insensitively for deterministic behavior.

    Parameters
    ----------
    path : str or os.PathLike[str], default='.'
        Path to inspect or operate on.
    pattern : str, default='*'
        Glob expression used to match entries.
    recursive : bool, default=False
        Include descendants.

    Returns
    -------
    list[pathlib.Path]
        Matching files and directories in deterministic order.

    Raises
    ------
    FileNotFoundError
        If the root does not exist.
    ValueError
        If the root is not a directory.

    See Also
    --------
    find
    walk

    Examples
    --------
    >>> from klygo import files
    >>> files.list_entries("dataset", pattern="*.json")
    """
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    if not candidate.is_dir():
        raise ValueError(f"Path must be a directory: {candidate}")
    iterator = candidate.rglob(pattern) if recursive else candidate.glob(pattern)
    return sorted(iterator, key=lambda item: str(item).lower())


def find(path: PathInput = ".", pattern: str = "*", recursive: bool = True) -> List[Path]:
    """Find files matching a glob pattern below a directory.

    Only regular files are returned; matching directories are excluded. Results are sorted case-insensitively.

    Parameters
    ----------
    path : str or os.PathLike[str], default='.'
        Path to inspect or operate on.
    pattern : str, default='*'
        Glob expression used to match entries.
    recursive : bool, default=True
        Include descendants.

    Returns
    -------
    list[pathlib.Path]
        Matching regular files in deterministic order.

    Raises
    ------
    FileNotFoundError
        If the root does not exist.

    See Also
    --------
    list_entries
    walk

    Examples
    --------
    >>> from klygo import files
    >>> images = files.find("dataset", pattern="*.jpg")
    """
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    iterator = candidate.rglob(pattern) if recursive else candidate.glob(pattern)
    return sorted((item for item in iterator if item.is_file()), key=lambda item: str(item).lower())


def iter_find(
    path: PathInput = ".",
    pattern: str = "*",
    recursive: bool = True,
    *,
    exclude_dirs: Iterable[str] = (),
) -> Generator[Path, None, None]:
    """Yield matching regular files without collecting the whole directory tree.

    Matches basename patterns using the host's case rules. Directory exclusions
    prune traversal, and nested directory symlinks are never followed. File
    symlinks are included when their targets are regular files.

    Parameters
    ----------
    path : str or os.PathLike[str], default='.'
        Existing directory at which to start searching.
    pattern : str, default='*'
        Filename wildcard (``*``, ``?``, ``[seq]``); no path separators.
    recursive : bool, default=True
        Include descendants.
    exclude_dirs : iterable[str], default=()
        Directory basename wildcards to omit, such as ``('.git', 'cache*')``.

    Returns
    -------
    generator[pathlib.Path, None, None]
        Matching files in filesystem order. Results are not sorted or reserved.

    Raises
    ------
    TypeError
        If a pattern or option has an invalid type.
    FileNotFoundError
        If the search root does not exist.
    ValueError
        If the root is not a directory or a pattern is empty or contains separators.
    OSError
        If a directory cannot be scanned; raised during iteration.

    Notes
    -----
    Root and options are validated immediately. Traversal starts on iteration
    and retains directory entries for the current traversal, not all results.
    For root-relative glob patterns or a sorted list, use ``find`` instead.

    See Also
    --------
    find
    walk

    Examples
    --------
    >>> from klygo import files
    >>> for image in files.iter_find("dataset", "*.jpg", exclude_dirs=("cache",)):
    ...     print(image.name)
    """
    validate_type(pattern, str, "pattern")
    validate_type(recursive, bool, "recursive")
    if isinstance(exclude_dirs, (str, bytes)):
        raise TypeError("exclude_dirs must be an iterable of directory patterns")
    excluded = tuple(exclude_dirs)
    for item in (pattern, *excluded):
        validate_type(item, str, "pattern")
        if not item or "/" in item or "\\" in item:
            raise ValueError("Patterns must be non-empty basenames without path separators")
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    if not candidate.is_dir():
        raise ValueError(f"Path must be a directory: {candidate}")

    def on_error(error):
        # os.walk normally suppresses scan failures; expose incomplete searches.
        raise error

    def generate():
        for current, directories, filenames in os.walk(candidate, onerror=on_error, followlinks=False):
            directories[:] = [
                name for name in directories
                if recursive and not any(fnmatch(name, rule) for rule in excluded)
            ]
            for name in filenames:
                if fnmatch(name, pattern):
                    item = Path(current) / name
                    if item.is_file():
                        yield item

    return generate()


def walk(path: PathInput = ".") -> Generator[Tuple[str, List[str], List[str]], None, None]:
    """Walk a directory tree lazily.

    The function returns ``os.walk`` directly, so traversal begins only when the iterator is consumed.

    Parameters
    ----------
    path : str or os.PathLike[str], default='.'
        Path to inspect or operate on.

    Returns
    -------
    iterator
        Tuples of root path, directory names, and file names.

    See Also
    --------
    list_entries
    find

    Examples
    --------
    >>> from klygo import files
    >>> for root, dirs, names in files.walk("dataset"):
    ...     print(root, names)
    """
    value = os.fspath(path)
    validate_type(value, str, "path")
    return os.walk(value)


def mkdir(path: PathInput, parents: bool = True, exist_ok: bool = True) -> Path:
    """Create a directory and return its path.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Path to inspect or operate on.
    parents : bool, default=True
        Create missing parent directories.
    exist_ok : bool, default=True
        Do not fail when the directory already exists.

    Returns
    -------
    pathlib.Path
        The created directory.

    Raises
    ------
    FileExistsError
        If creation conflicts with an existing entry.

    See Also
    --------
    remove

    Examples
    --------
    >>> from klygo import files
    >>> output = files.mkdir("runs/detect")
    """
    candidate = Path(path)
    candidate.mkdir(parents=parents, exist_ok=exist_ok)
    return candidate


def remove(path: PathInput, recursive: bool = True, missing_ok: bool = True) -> None:
    """Remove a file, symbolic link, or directory.

    Directory removal is recursive by default. Missing entries are ignored by
    default. Symbolic links, including broken ones, are unlinked without
    removing their targets.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Path to inspect or operate on.
    recursive : bool, default=True
        Remove non-empty directories recursively.
    missing_ok : bool, default=True
        Do not fail when the target is absent.

    Returns
    -------
    None
        The target is removed as a side effect.

    Raises
    ------
    FileNotFoundError
        If the target is absent and ``missing_ok=False``.

    See Also
    --------
    mkdir

    Examples
    --------
    >>> from klygo import files
    >>> files.remove("runs/temp", missing_ok=True)
    """
    candidate = Path(path)
    if not (candidate.exists() or candidate.is_symlink()):
        if missing_ok:
            return
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    if candidate.is_file() or candidate.is_symlink():
        candidate.unlink()
    elif recursive:
        shutil.rmtree(candidate)
    else:
        candidate.rmdir()


def copy(source: PathInput, target: PathInput, overwrite: bool = True) -> Path:
    """Copy a file or directory.

    File metadata is preserved. Copying a directory replaces the destination tree when overwrite is enabled.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Source URL or filesystem path.
    target : str or os.PathLike[str]
        Destination filesystem path.
    overwrite : bool, default=True
        Allow an existing destination to be replaced.

    Returns
    -------
    pathlib.Path
        Destination of the copied entry.

    Raises
    ------
    FileNotFoundError
        If the source does not exist.
    FileExistsError
        If the target exists and overwrite is disabled.

    See Also
    --------
    move

    Examples
    --------
    >>> from klygo import files
    >>> backup = files.copy("config.yaml", "backup/config.yaml")
    """
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.exists():
        raise FileNotFoundError(f"Source does not exist: {source_path}")
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"Target already exists: {target_path}")
    if source_path.is_file():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    else:
        if target_path.exists():
            remove(target_path)
        shutil.copytree(source_path, target_path)
    return target_path


def move(source: PathInput, target: PathInput, overwrite: bool = True) -> Path:
    """Move a file or directory.

    The destination parent is created automatically. An existing destination can be replaced explicitly.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Source URL or filesystem path.
    target : str or os.PathLike[str]
        Destination filesystem path.
    overwrite : bool, default=True
        Allow an existing destination to be replaced.

    Returns
    -------
    pathlib.Path
        Destination of the moved entry.

    Raises
    ------
    FileNotFoundError
        If the source does not exist.
    FileExistsError
        If the target exists and overwrite is disabled.

    See Also
    --------
    copy
    rename

    Examples
    --------
    >>> from klygo import files
    >>> result = files.move("draft.json", "output/result.json")
    """
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.exists():
        raise FileNotFoundError(f"Source does not exist: {source_path}")
    if target_path.exists():
        if not overwrite:
            raise FileExistsError(f"Target already exists: {target_path}")
        remove(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return Path(shutil.move(str(source_path), str(target_path)))


def rename(path: PathInput, new_name_or_path: PathInput, overwrite: bool = False) -> Path:
    """Rename a filesystem entry or move it to an explicit path.

    A single path component is interpreted relative to the source parent; a multi-part value is treated as an explicit destination.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Path to inspect or operate on.
    new_name_or_path : str or os.PathLike[str]
        New basename or explicit destination path.
    overwrite : bool, default=False
        Allow an existing destination to be replaced.

    Returns
    -------
    pathlib.Path
        New path of the renamed entry.

    Raises
    ------
    FileNotFoundError
        If the source does not exist.
    FileExistsError
        If the target exists and overwrite is disabled.

    See Also
    --------
    move

    Examples
    --------
    >>> from klygo import files
    >>> result = files.rename("draft.txt", "final.txt")
    """
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Path does not exist: {source_path}")
    requested = Path(new_name_or_path)
    target_path = source_path.parent / requested if len(requested.parts) == 1 else requested
    if target_path.exists():
        if not overwrite:
            raise FileExistsError(f"Target already exists: {target_path}")
        remove(target_path)
    source_path.rename(target_path)
    return target_path


__all__ = [
    "exists", "is_file", "is_dir", "list_entries", "find", "iter_find", "walk",
    "mkdir", "copy", "move", "rename", "remove",
]






