"""File metadata, sizing, hashing, and comparison."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

from klygo.archive.human_size import human_size as _human_size

PathInput = Union[str, Path]


def size(path: PathInput, human: bool = False) -> Union[int, str]:
    """Calculate the size of a file or directory tree.

    Directory sizes are the sum of all descendant regular files.

    Parameters
    ----------
    path : str or pathlib.Path
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
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    total = (
        candidate.stat().st_size
        if candidate.is_file()
        else sum(item.stat().st_size for item in candidate.rglob("*") if item.is_file())
    )
    return _human_size(total) if human else total


def hash(path: PathInput, algorithm: str = "md5") -> str:
    """Calculate a streaming checksum for a file.

    Files are processed in chunks, so large files are not loaded fully into memory.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to inspect or operate on.
    algorithm : str, default='md5'
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
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"Path must be a file to hash: {candidate}")
    hasher = hashlib.new(algorithm)
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def info(path: PathInput) -> Dict[str, Any]:
    """Collect common metadata for a file or directory.

    The result includes names, type flags, byte and human sizes, timestamps, and a checksum for regular files.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to inspect or operate on.

    Returns
    -------
    dict[str, Any]
        Filesystem metadata.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.

    See Also
    --------
    size
    hash

    Examples
    --------
    >>> from klygo import files
    >>> metadata = files.info("model.onnx")
    """
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    stat = candidate.stat()
    byte_size = size(candidate)
    return {
        "name": candidate.name,
        "stem": candidate.stem,
        "extension": candidate.suffix,
        "parent": candidate.parent,
        "size": byte_size,
        "human_size": _human_size(byte_size),
        "is_file": candidate.is_file(),
        "is_dir": candidate.is_dir(),
        "created": datetime.fromtimestamp(stat.st_ctime),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "hash": hash(candidate) if candidate.is_file() else None,
    }


def compare(path1: PathInput, path2: PathInput, by: str = "hash") -> bool:
    """Compare two files by checksum or binary content.

    A size check is performed first. Comparison then uses either checksums or chunked binary reads.

    Parameters
    ----------
    path1 : str or pathlib.Path
        Path to the first file.
    path2 : str or pathlib.Path
        Path to the second file.
    by : {'hash', 'content'}, default='hash'
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
    first = Path(path1)
    second = Path(path2)
    if not first.exists():
        raise FileNotFoundError(f"Path does not exist: {first}")
    if not second.exists():
        raise FileNotFoundError(f"Path does not exist: {second}")
    if first.stat().st_size != second.stat().st_size:
        return False
    if by == "hash":
        return hash(first) == hash(second)
    if by == "content":
        with first.open("rb") as left, second.open("rb") as right:
            while True:
                left_chunk = left.read(65536)
                right_chunk = right.read(65536)
                if left_chunk != right_chunk:
                    return False
                if not left_chunk:
                    return True
    raise ValueError(f"Invalid compare mode: {by!r}. Expected 'hash' or 'content'.")


__all__ = ["size", "hash", "info", "compare"]






