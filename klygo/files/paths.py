"""Pure path helpers used across Klygo."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple, Union

from klygo.validators import validate_type

PathInput = Union[str, Path]

_COMPOUND_SUFFIXES = (".tar.gz", ".tar.xz", ".tar.bz2")


def path(value: PathInput, expand_user: bool = True) -> Path:
    """Convert a string or path object to ``pathlib.Path``.

    Parameters
    ----------
    value : str or pathlib.Path
        Path-like value to transform.
    expand_user : bool, default=True
        Expand a leading ``~`` to the current user's home.

    Returns
    -------
    pathlib.Path
        Normalized path object.

    See Also
    --------
    resolve
    normalize

    Examples
    --------
    >>> from klygo import files
    >>> files.path("~/dataset").name
    'dataset'
    """
    validate_type(value, (str, Path), "value")
    validate_type(expand_user, bool, "expand_user")
    result = Path(value)
    return result.expanduser() if expand_user else result


def join(*parts: PathInput) -> Path:
    """Join one or more path components.

    Parameters
    ----------
    parts : str or pathlib.Path
        One or more path components.

    Returns
    -------
    pathlib.Path
        Joined path.

    Raises
    ------
    ValueError
        If no path components are supplied.

    See Also
    --------
    path
    parent

    Examples
    --------
    >>> from klygo import files
    >>> files.join("dataset", "images", "cat.jpg")
    """
    if not parts:
        raise ValueError("join() requires at least one path component")
    for index, part in enumerate(parts):
        validate_type(part, (str, Path), f"parts[{index}]")
    return Path(parts[0]).joinpath(*parts[1:])


def normalize(value: PathInput) -> Path:
    """Normalize separators and dot components in a path.

    Parameters
    ----------
    value : str or pathlib.Path
        Path-like value to transform.

    Returns
    -------
    pathlib.Path
        Lexically normalized path.

    See Also
    --------
    resolve

    Examples
    --------
    >>> from klygo import files
    >>> files.normalize("dataset/images/../labels")
    """
    return Path(os.path.normpath(str(path(value))))


def resolve(value: PathInput, strict: bool = False) -> Path:
    """Return an absolute normalized path.

    Parameters
    ----------
    value : str or pathlib.Path
        Path-like value to transform.
    strict : bool, default=False
        Require the resolved path to exist.

    Returns
    -------
    pathlib.Path
        Absolute normalized path.

    Raises
    ------
    FileNotFoundError
        If ``strict=True`` and the path does not exist.

    See Also
    --------
    normalize
    relative

    Examples
    --------
    >>> from klygo import files
    >>> files.resolve("README.md", strict=True)
    """
    validate_type(strict, bool, "strict")
    return path(value).resolve(strict=strict)


def relative(value: PathInput, start: PathInput = ".") -> Path:
    """Compute a path relative to a starting directory.

    Parameters
    ----------
    value : str or pathlib.Path
        Path-like value to transform.
    start : str or pathlib.Path, default='.'
        Starting directory used to compute the relative path.

    Returns
    -------
    pathlib.Path
        Path relative to ``start``.

    See Also
    --------
    resolve

    Examples
    --------
    >>> from klygo import files
    >>> files.relative("dataset/images/train", "dataset")
    """
    return Path(os.path.relpath(str(path(value)), start=str(path(start))))


def is_within(value: PathInput, root: PathInput, resolve_paths: bool = True) -> bool:
    """Check whether a path is contained by a root directory.

    Paths are resolved by default, making containment checks robust against ``..`` components and symbolic links.

    Parameters
    ----------
    value : str or pathlib.Path
        Path-like value to transform.
    root : str or pathlib.Path
        Root directory for the containment check.
    resolve_paths : bool, default=True
        Resolve both paths before testing containment.

    Returns
    -------
    bool
        Whether ``value`` is contained by ``root``.

    See Also
    --------
    relative
    common_path

    Examples
    --------
    >>> from klygo import files
    >>> files.is_within("dataset/images/cat.jpg", "dataset")
    True
    """
    validate_type(resolve_paths, bool, "resolve_paths")
    candidate = path(value)
    root_path = path(root)
    if resolve_paths:
        candidate = candidate.resolve(strict=False)
        root_path = root_path.resolve(strict=False)
    else:
        candidate = normalize(candidate)
        root_path = normalize(root_path)
    try:
        candidate.relative_to(root_path)
        return True
    except ValueError:
        return False


def common_path(values: Iterable[PathInput]) -> Path:
    """Return the longest common parent of multiple paths.

    Parameters
    ----------
    values : Iterable[str or pathlib.Path]
        Non-empty collection of paths.

    Returns
    -------
    pathlib.Path
        Longest common parent.

    Raises
    ------
    TypeError
        If a single path is supplied instead of an iterable.
    ValueError
        If the iterable is empty.

    See Also
    --------
    is_within

    Examples
    --------
    >>> from klygo import files
    >>> files.common_path(["data/images", "data/labels"])
    """
    if isinstance(values, (str, Path)):
        raise TypeError("values must be an iterable of paths, not a single path")
    items = tuple(values)
    if not items:
        raise ValueError("common_path() requires at least one path")
    for index, item in enumerate(items):
        validate_type(item, (str, Path), f"values[{index}]")
    return Path(os.path.commonpath([str(path(item)) for item in items]))


def replace_root(value: PathInput, old_root: PathInput, new_root: PathInput) -> Path:
    """Map a path from one directory tree into another.

    The relative location below ``old_root`` is preserved below ``new_root``.

    Parameters
    ----------
    value : str or pathlib.Path
        Path-like value to transform.
    old_root : str or pathlib.Path
        Root removed from the input path.
    new_root : str or pathlib.Path
        Replacement root.

    Returns
    -------
    pathlib.Path
        Mapped path below ``new_root``.

    Raises
    ------
    ValueError
        If ``value`` is not inside ``old_root``.

    See Also
    --------
    relative
    is_within

    Examples
    --------
    >>> from klygo import files
    >>> files.replace_root("data/images/a.jpg", "data/images", "data/labels")
    """
    candidate = normalize(value)
    source_root = normalize(old_root)
    try:
        relative_part = candidate.relative_to(source_root)
    except ValueError:
        try:
            relative_part = candidate.resolve(strict=False).relative_to(
                source_root.resolve(strict=False)
            )
        except ValueError as exc:
            raise ValueError(f"Path {candidate} is not inside root {source_root}") from exc
    return normalize(path(new_root).joinpath(relative_part))


def with_name(value: PathInput, new_name: str) -> Path:
    """Return a path with a replaced final component.

    Parameters
    ----------
    value : str or pathlib.Path
        Path-like value to transform.
    new_name : str
        Replacement final path component.

    Returns
    -------
    pathlib.Path
        Path with the new final component.

    See Also
    --------
    with_stem
    with_extension

    Examples
    --------
    >>> from klygo import files
    >>> files.with_name("data/cat.jpg", "dog.jpg")
    """
    validate_type(new_name, str, "new_name")
    return path(value).with_name(new_name)


def with_stem(value: PathInput, new_stem: str) -> Path:
    """Return a path with a replaced stem.

    Parameters
    ----------
    value : str or pathlib.Path
        Path-like value to transform.
    new_stem : str
        Replacement filename stem.

    Returns
    -------
    pathlib.Path
        Path with the new stem.

    See Also
    --------
    with_name
    with_extension

    Examples
    --------
    >>> from klygo import files
    >>> files.with_stem("data/cat.jpg", "dog")
    """
    validate_type(new_stem, str, "new_stem")
    return path(value).with_stem(new_stem)


def extensions(value: PathInput) -> Tuple[str, ...]:
    """Return every filename suffix as a tuple.

    Parameters
    ----------
    value : str or pathlib.Path
        Path-like value to transform.

    Returns
    -------
    tuple[str, ...]
        All suffixes in filename order.

    See Also
    --------
    extension
    compound_extension

    Examples
    --------
    >>> from klygo import files
    >>> files.extensions("archive.tar.gz")
    ('.tar', '.gz')
    """
    return tuple(path(value).suffixes)


def compound_extension(value: PathInput) -> str:
    """Return every filename suffix joined together.

    Parameters
    ----------
    value : str or pathlib.Path
        Path-like value to transform.

    Returns
    -------
    str
        All suffixes joined together.

    See Also
    --------
    extensions
    extension

    Examples
    --------
    >>> from klygo import files
    >>> files.compound_extension("archive.tar.gz")
    '.tar.gz'
    """
    return "".join(extensions(value))


def _effective_extension(value: PathInput) -> str:
    candidate = path(value)
    lowered_name = candidate.name.lower()
    for suffix in _COMPOUND_SUFFIXES:
        if lowered_name.endswith(suffix):
            return candidate.name[-len(suffix):]
    return candidate.suffix


def with_extension(value: PathInput, new_extension: str) -> Path:
    """Return a path with a replaced filename extension.

    Known archive extensions ``.tar.gz``, ``.tar.xz``, and ``.tar.bz2`` are replaced as one unit.

    Parameters
    ----------
    value : str or pathlib.Path
        Path-like value to transform.
    new_extension : str
        Replacement extension, with or without a leading dot.

    Returns
    -------
    pathlib.Path
        Path with the new extension.

    Raises
    ------
    ValueError
        If the extension contains a path separator.

    See Also
    --------
    extension
    compound_extension

    Examples
    --------
    >>> from klygo import files
    >>> files.with_extension("archive.tar.gz", ".zip")
    """
    validate_type(new_extension, str, "new_extension")
    if any(separator in new_extension for separator in ("/", "\\")):
        raise ValueError("new_extension must not contain path separators")
    suffix = new_extension.strip()
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"
    candidate = path(value)
    current = _effective_extension(candidate)
    base_name = candidate.name[:-len(current)] if current else candidate.name
    return candidate.with_name(f"{base_name}{suffix}")


def unique_path(value: PathInput, separator: str = "_", start: int = 1) -> Path:
    """Return a currently unused path.

    If the requested path exists, an incrementing numeric suffix is inserted before the effective extension.

    Parameters
    ----------
    value : str or pathlib.Path
        Path-like value to transform.
    separator : str, default='_'
        Text inserted before the numeric suffix.
    start : int, default=1
        Initial numeric suffix.

    Returns
    -------
    pathlib.Path
        An unused candidate path.

    Raises
    ------
    ValueError
        If ``start`` is negative.

    See Also
    --------
    exists

    Examples
    --------
    >>> from klygo import files
    >>> output = files.unique_path("result.json")
    """
    validate_type(separator, str, "separator")
    validate_type(start, int, "start")
    if start < 0:
        raise ValueError("start must be greater than or equal to 0")
    candidate = path(value)
    if not candidate.exists():
        return candidate
    suffix = _effective_extension(candidate)
    base_name = candidate.name[:-len(suffix)] if suffix else candidate.name
    counter = start
    while True:
        numbered = candidate.with_name(f"{base_name}{separator}{counter}{suffix}")
        if not numbered.exists():
            return numbered
        counter += 1


def parents(value: PathInput) -> Tuple[Path, ...]:
    """Return every ancestor path from nearest to farthest.

    Parameters
    ----------
    value : str or pathlib.Path
        Path-like value to transform.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Ancestors from nearest to farthest.

    See Also
    --------
    parent

    Examples
    --------
    >>> from klygo import files
    >>> files.parents("data/images/cat.jpg")[0]
    """
    return tuple(path(value).parents)


def is_absolute(value: PathInput) -> bool:
    """Check whether a path is absolute.

    Parameters
    ----------
    value : str or pathlib.Path
        Path-like value to transform.

    Returns
    -------
    bool
        Whether the value is absolute.

    See Also
    --------
    resolve

    Examples
    --------
    >>> from klygo import files
    >>> files.is_absolute("data/images")
    False
    """
    return path(value).is_absolute()


def name(value: PathInput) -> str:
    """Return the final path component including its extension.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to inspect or operate on.

    Returns
    -------
    str
        Final path component.

    See Also
    --------
    stem
    extension

    Examples
    --------
    >>> from klygo import files
    >>> files.name("data/images/cat.jpg")
    'cat.jpg'
    """
    return path(value).name


def stem(value: PathInput) -> str:
    """Return the final path component without its last extension.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to inspect or operate on.

    Returns
    -------
    str
        Final component without its last suffix.

    See Also
    --------
    name
    extension

    Examples
    --------
    >>> from klygo import files
    >>> files.stem("archive.tar.gz")
    'archive.tar'
    """
    return path(value).stem


def extension(value: PathInput) -> str:
    """Return the final filename extension.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to inspect or operate on.

    Returns
    -------
    str
        Final suffix, including the leading dot.

    See Also
    --------
    extensions
    compound_extension

    Examples
    --------
    >>> from klygo import files
    >>> files.extension("archive.tar.gz")
    '.gz'
    """
    return path(value).suffix


def parent(value: PathInput) -> Path:
    """Return the immediate parent path.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to inspect or operate on.

    Returns
    -------
    pathlib.Path
        Immediate parent path.

    See Also
    --------
    parents

    Examples
    --------
    >>> from klygo import files
    >>> files.parent("data/images/cat.jpg")
    """
    return path(value).parent


__all__ = [
    "path", "join", "normalize", "resolve", "relative", "is_within",
    "common_path", "replace_root", "with_name", "with_stem",
    "with_extension", "unique_path", "extensions", "compound_extension",
    "parents", "is_absolute", "name", "stem", "extension", "parent",
]






