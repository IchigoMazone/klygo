"""Filesystem status, traversal, and mutation operations."""

from __future__ import annotations

import os
import shutil
import tempfile
from fnmatch import fnmatch
from pathlib import Path
from typing import Generator, Iterable, List, Tuple

from klygo.validators import validate_type

from ._typing import PathInput
from .paths import is_absolute, name, parent, parents, path as _to_path, resolve


def _entry_exists(candidate: Path) -> bool:
    return exists(candidate) or is_symlink(candidate)


def _validate_transfer(source: Path, target: Path) -> None:
    if not _entry_exists(source):
        raise FileNotFoundError(f"Source does not exist: {source}")
    source_resolved = resolve(source, strict=False)
    target_resolved = resolve(target, strict=False)
    if source_resolved == target_resolved:
        raise ValueError("Source and destination must be different paths")
    if target_resolved in parents(source_resolved) or (
        is_dir(source) and source_resolved in parents(target_resolved)
    ):
        raise ValueError("Source and destination cannot contain each other")


def _publish_staged(staged: Path, target: Path, overwrite: bool) -> None:
    if not overwrite:
        if is_symlink(staged):
            target.symlink_to(os.readlink(staged), target_is_directory=is_dir(staged))
        elif is_file(staged):
            os.link(staged, target)
        else:
            if _entry_exists(target):
                raise FileExistsError(f"Target already exists: {target}")
            if os.name == "nt":
                os.rename(staged, target)
            else:
                target.mkdir()
                try:
                    os.replace(staged, target)
                except BaseException:
                    target.rmdir()
                    raise
        return
    backup = parent(staged) / "previous"
    if _entry_exists(target):
        os.replace(target, backup)
    try:
        os.replace(staged, target)
    except BaseException:
        if _entry_exists(backup):
            try:
                os.replace(backup, target)
            except OSError as rollback_error:
                raise OSError(
                    f"Could not restore the previous destination; backup retained at {backup}"
                ) from rollback_error
        raise


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

    Raises
    ------
    TypeError
        If a path or option has an unsupported type.

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
    return _to_path(path).exists()


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

    Raises
    ------
    TypeError
        If a path or option has an unsupported type.

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
    return _to_path(path).is_file()


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

    Raises
    ------
    TypeError
        If a path or option has an unsupported type.

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
    return _to_path(path).is_dir()


def is_symlink(path: PathInput) -> bool:
    """Return whether a path is a symbolic link, including a broken link.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Path to inspect.

    Returns
    -------
    bool
        True when the directory entry itself is a symbolic link.

    Raises
    ------
    TypeError
        If a path or option has an unsupported type.

    Examples
    --------
    >>> from klygo import files
    >>> files.is_symlink("missing-link")
    False
    """
    return _to_path(path).is_symlink()


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
    candidate = _to_path(path)
    if not exists(candidate):
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    if not is_dir(candidate):
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
    candidate = _to_path(path)
    if not exists(candidate):
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    iterator = candidate.rglob(pattern) if recursive else candidate.glob(pattern)
    return sorted((item for item in iterator if is_file(item)), key=lambda item: str(item).lower())


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
    candidate = _to_path(path)
    if not exists(candidate):
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    if not is_dir(candidate):
        raise ValueError(f"Path must be a directory: {candidate}")

    def on_error(error):
        # os.walk normally suppresses scan failures; expose incomplete searches.
        raise error

    def generate():
        for current, directories, filenames in os.walk(candidate, onerror=on_error, followlinks=False):
            directories[:] = [
                entry_name for entry_name in directories
                if recursive and not any(fnmatch(entry_name, rule) for rule in excluded)
            ]
            for entry_name in filenames:
                if fnmatch(entry_name, pattern):
                    item = _to_path(current, expand_user=False) / entry_name
                    if is_file(item):
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

    Raises
    ------
    TypeError
        If a path or option has an unsupported type.

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
    return os.walk(os.path.expanduser(value))


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
    candidate = _to_path(path)
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
    candidate = _to_path(path)
    if not _entry_exists(candidate):
        if missing_ok:
            return
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    if is_file(candidate) or is_symlink(candidate):
        candidate.unlink()
    elif recursive:
        shutil.rmtree(candidate)
    else:
        candidate.rmdir()


def copy(source: PathInput, target: PathInput, overwrite: bool = False) -> Path:
    """Copy a file or directory.

    File metadata is preserved. The source is copied to a sibling staging path
    before the destination is replaced, so a failed copy leaves it intact.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Source URL or filesystem path.
    target : str or os.PathLike[str]
        Destination filesystem path.
    overwrite : bool, default=False
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
    ValueError
        If source and target are the same or overlap as directories.

    See Also
    --------
    move

    Examples
    --------
    >>> from klygo import files
    >>> backup = files.copy("config.yaml", "backup/config.yaml")
    """
    source_path = _to_path(source)
    target_path = _to_path(target)
    validate_type(overwrite, bool, "overwrite")
    _validate_transfer(source_path, target_path)
    if _entry_exists(target_path) and not overwrite:
        raise FileExistsError(f"Target already exists: {target_path}")
    mkdir(parent(target_path), parents=True, exist_ok=True)
    staging = _to_path(
        tempfile.mkdtemp(prefix=f".{name(target_path)}.", dir=parent(target_path)),
        expand_user=False,
    )
    staged = staging / "payload"
    published = False
    try:
        if is_symlink(source_path):
            staged.symlink_to(resolve(source_path, strict=False), target_is_directory=is_dir(source_path))
        elif is_dir(source_path):
            shutil.copytree(source_path, staged, symlinks=True)
        elif is_file(source_path):
            shutil.copy2(source_path, staged)
        else:
            raise ValueError(f"Source must be a file or directory: {source_path}")
        _publish_staged(staged, target_path, overwrite)
        published = True
    finally:
        if published or not _entry_exists(staging / "previous"):
            shutil.rmtree(staging)
    return target_path


def move(source: PathInput, target: PathInput, overwrite: bool = False) -> Path:
    """Move a file or directory.

    The destination is prepared before the source is removed. A failed copy or
    publication leaves the source intact.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Source URL or filesystem path.
    target : str or os.PathLike[str]
        Destination filesystem path.
    overwrite : bool, default=False
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
    ValueError
        If source and target are the same or overlap as directories.

    See Also
    --------
    copy
    rename

    Examples
    --------
    >>> from klygo import files
    >>> result = files.move("draft.json", "output/result.json")
    """
    source_path = _to_path(source)
    target_path = _to_path(target)
    validate_type(overwrite, bool, "overwrite")
    _validate_transfer(source_path, target_path)
    copy(source_path, target_path, overwrite=overwrite)
    remove(source_path, missing_ok=False)
    return target_path


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
    ValueError
        If source and target are the same or overlap as directories.

    See Also
    --------
    move

    Examples
    --------
    >>> from klygo import files
    >>> result = files.rename("draft.txt", "final.txt")
    """
    source_path = _to_path(path)
    validate_type(overwrite, bool, "overwrite")
    requested = _to_path(new_name_or_path)
    target_path = parent(source_path) / requested if len(requested.parts) == 1 else requested
    _validate_transfer(source_path, target_path)
    if (
        is_symlink(source_path)
        and not is_absolute(os.readlink(source_path))
        and resolve(parent(source_path), strict=False) != resolve(parent(target_path), strict=False)
    ):
        raise ValueError("Renaming a relative symlink into another directory would change its target")
    if _entry_exists(target_path) and not overwrite:
        raise FileExistsError(f"Target already exists: {target_path}")
    mkdir(parent(target_path), parents=True, exist_ok=True)
    if not overwrite:
        if os.name == "nt":
            os.rename(source_path, target_path)
        elif is_symlink(source_path):
            target_path.symlink_to(
                os.readlink(source_path), target_is_directory=is_dir(source_path)
            )
            source_path.unlink()
        elif is_file(source_path):
            os.link(source_path, target_path)
            source_path.unlink()
        else:
            target_path.mkdir()
            try:
                os.replace(source_path, target_path)
            except BaseException:
                target_path.rmdir()
                raise
        return target_path
    staging = _to_path(
        tempfile.mkdtemp(prefix=f".{name(target_path)}.", dir=parent(target_path)),
        expand_user=False,
    )
    backup = staging / "previous"
    published = False
    try:
        if _entry_exists(target_path):
            os.replace(target_path, backup)
        try:
            os.replace(source_path, target_path)
        except BaseException:
            if _entry_exists(backup):
                try:
                    os.replace(backup, target_path)
                except OSError as rollback_error:
                    raise OSError(
                        f"Could not restore the previous destination; backup retained at {backup}"
                    ) from rollback_error
            raise
        published = True
    finally:
        if published or not _entry_exists(backup):
            shutil.rmtree(staging)
    return target_path


__all__ = [
    "exists", "is_file", "is_dir", "is_symlink", "list_entries", "find", "iter_find", "walk",
    "mkdir", "copy", "move", "rename", "remove",
]
