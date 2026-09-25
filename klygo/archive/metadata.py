"""Archive member discovery, metadata, validation, and comparison."""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Union

from klygo.archive._utils import resolve_backend

def list_files(archive_path: Union[str, Path]) -> List[str]:
    """Return the member names stored in an archive.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Archive to inspect or modify.

    Returns
    -------
    list[str]
        Stored member names in backend order.

    Raises
    ------
    FileNotFoundError
        If the archive does not exist.

    See Also
    --------
    iter_files
    search

    Examples
    --------
    >>> import klygo.archive as archive
    >>> names = archive.list_files("dataset.zip")
    """
    path, backend = resolve_backend(archive_path)
    return backend.list_files(path)


def iter_files(archive_path: Union[str, Path]) -> Iterator[str]:
    """Iterate over archive member names lazily.

    The backend controls iteration and avoids building an additional result list.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Archive to inspect or modify.

    Returns
    -------
    Iterator[str]
        Lazy stream of stored member names.

    See Also
    --------
    list_files

    Examples
    --------
    >>> import klygo.archive as archive
    >>> for name in archive.iter_files("dataset.zip"):
    ...     print(name)
    """
    path, backend = resolve_backend(archive_path)
    yield from backend.iter_files(path)


def search(
    archive_path: Union[str, Path],
    pattern: str,
    regex: bool = False,
    case_sensitive: bool = True,
) -> List[str]:
    """Search archive member names using a glob or regular expression.

    Glob matching is the default. Set ``regex=True`` to use a regular expression.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Archive to inspect or modify.
    pattern : str
        Glob pattern or regular expression.
    regex : bool, default=False
        Interpret ``pattern`` as a regular expression.
    case_sensitive : bool, default=True
        Match letter case exactly.

    Returns
    -------
    list[str]
        Member names matching the requested pattern.

    See Also
    --------
    list_files
    extract_matching

    Examples
    --------
    >>> import klygo.archive as archive
    >>> archive.search("dataset.zip", "images/*.jpg")
    """
    path, backend = resolve_backend(archive_path)
    return backend.search(
        archive_path=path,
        pattern=pattern,
        regex=regex,
        case_sensitive=case_sensitive,
    )


def get_info(archive_path: Union[str, Path]) -> Dict[str, Any]:
    """Return detailed archive metadata and compression statistics.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Archive to inspect or modify.

    Returns
    -------
    dict[str, Any]
        Format, size, ratio, encryption, and member statistics.

    Raises
    ------
    FileNotFoundError
        If the archive does not exist.

    See Also
    --------
    verify
    test

    Examples
    --------
    >>> import klygo.archive as archive
    >>> metadata = archive.get_info("dataset.zip")
    """
    path, backend = resolve_backend(archive_path)
    return backend.get_info(path)


def test(archive_path: Union[str, Path], raise_exception: bool = False) -> bool:
    """Test archive integrity.

    Corruption normally returns ``False``; ``raise_exception=True`` converts backend failures into ``ValueError``.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Archive to inspect or modify.
    raise_exception : bool, default=False
        Raise ``ValueError`` instead of returning ``False`` on corruption.

    Returns
    -------
    bool
        ``True`` when the archive passes integrity checks.

    Raises
    ------
    ValueError
        If integrity fails and ``raise_exception=True``.

    See Also
    --------
    verify

    Examples
    --------
    >>> import klygo.archive as archive
    >>> archive.test("dataset.zip")
    True
    """
    path, backend = resolve_backend(archive_path)
    return backend.test(path, raise_exception=raise_exception)


def verify(archive_path: Union[str, Path]) -> Dict[str, Any]:
    """Build a high-level archive verification report.

    The report combines integrity status with format, member count, size, and encryption metadata.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Archive to inspect or modify.

    Returns
    -------
    dict[str, Any]
        Summary containing validity and archive metadata.

    See Also
    --------
    test
    get_info

    Examples
    --------
    >>> import klygo.archive as archive
    >>> archive.verify("dataset.zip")["valid"]
    True
    """
    path, backend = resolve_backend(archive_path)
    is_valid = backend.test(path, raise_exception=False)
    info = backend.get_info(path)

    return {
        "valid": is_valid,
        "format": info.get("format"),
        "file_count": info.get("file_count", 0),
        "archive_size": info.get("archive_size", 0),
        "human_archive_size": info.get("human_archive_size", ""),
        "encrypted": info.get("encrypted", False),
    }


# Ngăn pytest thu thập nhầm API này khi được import vào module test.
test.__test__ = False

def compare(archive1: Union[str, Path], archive2: Union[str, Path]) -> Dict[str, List[str]]:
    """Compare the member-name sets of two archives.

    File contents are not compared; this function compares normalized member-name sets.

    Parameters
    ----------
    archive1 : str or pathlib.Path
        First archive.
    archive2 : str or pathlib.Path
        Second archive.

    Returns
    -------
    dict[str, list[str]]
        Added, removed, and common member names.

    See Also
    --------
    list_files

    Examples
    --------
    >>> import klygo.archive as archive
    >>> difference = archive.compare("v1.zip", "v2.zip")
    """
    path1, backend1 = resolve_backend(archive1)
    path2, backend2 = resolve_backend(archive2)

    set1 = set(backend1.list_files(path1))
    set2 = set(backend2.list_files(path2))

    return {
        "added_files": sorted(set2 - set1),
        "removed_files": sorted(set1 - set2),
        "common_files": sorted(set1 & set2),
    }
