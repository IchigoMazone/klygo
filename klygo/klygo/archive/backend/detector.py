"""Detect archive formats and construct their backend adapters."""

from pathlib import Path
from typing import Callable, Dict, Union

import klygo.files as file_utils
from klygo.archive.backend.base import ArchiveBackend
from klygo.archive.backend.zip_backend import ZipBackend
from klygo.archive.backend.tar_backend import TarBackend
from klygo.archive.backend.gzip_backend import GZipBackend
from klygo.archive.backend.sevenzip_backend import SevenZipBackend
from klygo.archive.backend.rar_backend import RarBackend

SUPPORTED_EXTENSIONS = {
    ".zip": "zip",
    ".tar": "tar",
    ".tar.gz": "tar.gz",
    ".tgz": "tar.gz",
    ".tar.xz": "tar.xz",
    ".txz": "tar.xz",
    ".tar.bz2": "tar.bz2",
    ".tbz2": "tar.bz2",
    ".7z": "7z",
    ".gz": "gz",
    ".rar": "rar",
}


BACKEND_FACTORIES: Dict[str, Callable[[], ArchiveBackend]] = {
    "zip": ZipBackend,
    "tar": lambda: TarBackend("tar"),
    "tar.gz": lambda: TarBackend("tar.gz"),
    "tar.xz": lambda: TarBackend("tar.xz"),
    "tar.bz2": lambda: TarBackend("tar.bz2"),
    "gz": GZipBackend,
    "7z": SevenZipBackend,
    "rar": RarBackend,
}

FORMAT_ALIASES = {
    "tgz": "tar.gz",
    "txz": "tar.xz",
    "tbz2": "tar.bz2",
}


def detect_format(path: Union[str, Path]) -> str:
    """Detect an archive format from its name and magic bytes.

    Known compound extensions are checked first. Existing files are also inspected for ZIP, GZip, XZ, 7Z, RAR, and TAR signatures.

    Parameters
    ----------
    path : str or pathlib.Path
        Archive path or candidate filename.

    Returns
    -------
    str
        Canonical format identifier such as ``zip`` or ``tar.gz``.

    Raises
    ------
    ValueError
        If neither the extension nor file signature is supported.

    See Also
    --------
    is_archive

    Examples
    --------
    >>> import klygo.archive as archive
    >>> archive.detect_format("dataset.tar.gz")
    'tar.gz'
    """
    filepath = file_utils.path(path)
    filename = filepath.name.lower()

    # Compound extensions check first
    for ext in (".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tbz2"):
        if filename.endswith(ext):
            return SUPPORTED_EXTENSIONS[ext]

    # File magic bytes check if file exists
    if file_utils.is_file(filepath):
        try:
            with open(filepath, "rb") as f:
                header = f.read(512)
                if header.startswith(b"PK\x03\x04") or header.startswith(b"PK\x05\x06"):
                    return "zip"
                if header.startswith(b"\x1f\x8b"):
                    return "tar.gz" if filename.endswith((".tar.gz", ".tgz")) else "gz"
                if header.startswith(b"\xfd7zXZ\x00"):
                    return "tar.xz"
                if header.startswith(b"7z\xbc\xaf\x27\x1c"):
                    return "7z"
                if header.startswith(b"Rar!\x1a\x07"):
                    return "rar"
                if len(header) >= 262 and header[257:262] == b"ustar":
                    return "tar"
        except Exception:
            pass

    # Suffix fallback
    suffix = filepath.suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[suffix]

    raise ValueError(
        f"Unsupported or unrecognized archive format for file '{filepath}'. "
        f"Supported formats: {sorted(set(SUPPORTED_EXTENSIONS.values()))}"
    )


def is_archive(path: Union[str, Path]) -> bool:
    """Check whether a path appears to be a supported archive.

    The function is intentionally non-raising and returns ``False`` for missing, unsupported, or unreadable paths.

    Parameters
    ----------
    path : str or pathlib.Path
        Archive path or candidate filename.

    Returns
    -------
    bool
        Whether detection succeeds with a supported format.

    See Also
    --------
    detect_format

    Examples
    --------
    >>> import klygo.archive as archive
    >>> archive.is_archive("dataset.zip")
    True
    """
    try:
        fmt = detect_format(path)
        return fmt in SUPPORTED_EXTENSIONS.values()
    except Exception:
        return False


def get_backend(
    path: Union[str, Path],
    format_hint: str = None,
) -> ArchiveBackend:
    """Construct the backend selected by a format hint or archive path.

    Parameters
    ----------
    path : str or pathlib.Path
        Archive path used for detection when ``format_hint`` is omitted. The
        path need not exist when its extension identifies the format.
    format_hint : str or None, default=None
        Explicit format name. Canonical names and the aliases ``tgz``, ``txz``,
        and ``tbz2`` are accepted. A hint takes precedence over ``path``.

    Returns
    -------
    ArchiveBackend
        A new backend instance.

    Raises
    ------
    ValueError
        If detection fails or the requested format has no registered backend.

    Examples
    --------
    >>> from klygo.archive.backend import get_backend
    >>> type(get_backend("bundle.zip")).__name__
    'ZipBackend'
    >>> get_backend("unused", format_hint="tgz").format_name
    'tar.gz'
    """
    requested = (format_hint or detect_format(path)).lower()
    fmt = FORMAT_ALIASES.get(requested, requested)
    try:
        factory = BACKEND_FACTORIES[fmt]
    except KeyError as exc:
        raise ValueError(f"No backend available for format '{requested}'.") from exc
    return factory()
