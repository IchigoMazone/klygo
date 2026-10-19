"""File metadata, sizing, hashing, and comparison."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, Union

from klygo.utils.formatting import human_size as _human_size
from klygo.validators import validate_type

from ._typing import PathInput
from .filesystem import exists, find, is_dir, is_file
from .paths import extension, name, parent, path as _to_path, stem


def size(path: PathInput, human: bool = False) -> Union[int, str]:
    """Calculate the size of a file or directory tree.

    Directory sizes are the sum of all descendant regular files.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Path to inspect or operate on.
    human : bool, default=False
        Return a human-readable size rather than an integer.

    Returns
    -------
    int or str
        Bytes, or a formatted size when ``human=True``.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.

    See Also
    --------
    info

    Examples
    --------
    >>> from klygo import files
    >>> files.size("model.onnx", human=True)
    """
    candidate = _to_path(path)
    if not exists(candidate):
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    total = (
        candidate.stat().st_size
        if is_file(candidate)
        else sum(item.stat().st_size for item in find(candidate, "*", recursive=True))
    )
    return _human_size(total) if human else total


def hash(path: PathInput, algorithm: str = "sha256") -> str:
    """Calculate a streaming checksum for a file.

    Files are processed in chunks, so large files are not loaded fully into memory.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Path to inspect or operate on.
    algorithm : str, default='sha256'
        Algorithm accepted by ``hashlib.new``.

    Returns
    -------
    str
        Lowercase hexadecimal checksum.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the path is not a file or the algorithm is unknown.

    See Also
    --------
    compare

    Examples
    --------
    >>> from klygo import files
    >>> digest = files.hash("model.onnx", "sha256")
    """
    candidate = _to_path(path)
    if not exists(candidate):
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    if not is_file(candidate):
        raise ValueError(f"Path must be a file to hash: {candidate}")
    hasher = hashlib.new(algorithm)
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def info(path: PathInput, *, verbose: bool = False) -> Dict[str, Any]:
    """Collect common metadata for a file or directory.

    The result includes names, type flags, byte and human sizes, and timestamps.
    Directory size requires traversal of its descendants.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Path to inspect or operate on.
    verbose : bool, default=False
        Print the metadata as a Rich table while returning the dictionary.

    Returns
    -------
    dict[str, Any]
        Filesystem metadata with stable keys.

    Raises
    ------
    TypeError
        If ``verbose`` is not a boolean.
    FileNotFoundError
        If the path does not exist.

    See Also
    --------
    size
    hash

    Examples
    --------
    >>> from klygo import files
    >>> metadata = files.info("model.onnx", verbose=True)
    """
    validate_type(verbose, bool, "verbose")
    candidate = _to_path(path)
    if not exists(candidate):
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    stat = candidate.stat()
    byte_size = size(candidate)
    metadata = {
        "name": name(candidate),
        "stem": stem(candidate),
        "extension": extension(candidate),
        "parent": parent(candidate),
        "size": byte_size,
        "human_size": _human_size(byte_size),
        "is_file": is_file(candidate),
        "is_dir": is_dir(candidate),
        "created": datetime.fromtimestamp(stat.st_ctime),
        "modified": datetime.fromtimestamp(stat.st_mtime),
    }
    if verbose:
        from rich.console import Console
        from rich.table import Table

        table = Table(title=f"File information: {candidate}")
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        for key, value in metadata.items():
            table.add_row(key, str(value))
        Console().print(table)
    return metadata


def compare(path1: PathInput, path2: PathInput, by: str = "content") -> bool:
    """Compare two files by checksum or binary content.

    A size check is performed first. Comparison then uses either checksums or chunked binary reads.

    Parameters
    ----------
    path1 : str or os.PathLike[str]
        Path to the first file.
    path2 : str or os.PathLike[str]
        Path to the second file.
    by : {'hash', 'content'}, default='content'
        File comparison strategy.

    Returns
    -------
    bool
        Whether both files contain identical bytes.

    Raises
    ------
    FileNotFoundError
        If either file does not exist.
    ValueError
        If ``by`` is not a supported strategy.

    See Also
    --------
    hash

    Examples
    --------
    >>> from klygo import files
    >>> files.compare("a.bin", "b.bin", by="content")
    """
    if by not in ("hash", "content"):
        raise ValueError(f"Invalid compare mode: {by!r}. Expected 'hash' or 'content'.")
    first = _to_path(path1)
    second = _to_path(path2)
    if not exists(first):
        raise FileNotFoundError(f"Path does not exist: {first}")
    if not exists(second):
        raise FileNotFoundError(f"Path does not exist: {second}")
    if first.stat().st_size != second.stat().st_size:
        return False
    if by == "hash":
        return hash(first) == hash(second)
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            left_chunk = left.read(65536)
            right_chunk = right.read(65536)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


__all__ = ["size", "hash", "info", "compare"]
