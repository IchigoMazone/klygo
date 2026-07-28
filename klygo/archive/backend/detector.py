from pathlib import Path
from typing import Union

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


def detect_format(path: Union[str, Path]) -> str:
    """
    Detect archive format from file magic bytes or extension.
    """
    filepath = Path(path)
    filename = filepath.name.lower()

    # Compound extensions check first
    for ext in (".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tbz2"):
        if filename.endswith(ext):
            return SUPPORTED_EXTENSIONS[ext]

    # File magic bytes check if file exists
    if filepath.exists() and filepath.is_file():
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
    """
    Check if a file path points to a supported archive format.
    """
    try:
        fmt = detect_format(path)
        return fmt in SUPPORTED_EXTENSIONS.values()
    except Exception:
        return False


def get_backend(path: Union[str, Path], format_hint: str = None) -> ArchiveBackend:
    """
    Get the appropriate ArchiveBackend instance for a given path or format hint.
    """
    fmt = (format_hint or detect_format(path)).lower()

    if fmt == "zip":
        return ZipBackend()
    elif fmt in ("tar", "tar.gz", "tar.xz", "tar.bz2", "tgz", "txz", "tbz2"):
        return TarBackend(format_name=fmt)
    elif fmt == "gz":
        return GZipBackend()
    elif fmt == "7z":
        return SevenZipBackend()
    elif fmt == "rar":
        return RarBackend()

    raise ValueError(f"No backend available for format '{fmt}'.")
