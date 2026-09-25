"""Archive backend contracts, built-in adapters, and format selection.

Most callers should use :mod:`klygo.archive`. Import this package directly when
writing a backend, inspecting capabilities, or deliberately selecting a
specific adapter.

Examples
--------
>>> from klygo.archive.backend import get_backend
>>> backend = get_backend("dataset.tar.gz")
>>> backend.format_name
'tar.gz'
>>> backend.capabilities.compress
True
"""

from klygo.archive.backend.base import (
    ArchiveBackend,
    BackendCapabilities,
    UnsupportedOperationError,
    UnsupportedOptionError,
)
from klygo.archive.backend.detector import detect_format, is_archive, get_backend
from klygo.archive.backend.zip_backend import ZipBackend
from klygo.archive.backend.tar_backend import TarBackend
from klygo.archive.backend.gzip_backend import GZipBackend
from klygo.archive.backend.sevenzip_backend import SevenZipBackend
from klygo.archive.backend.rar_backend import RarBackend

__all__ = [
    "ArchiveBackend",
    "BackendCapabilities",
    "UnsupportedOperationError",
    "UnsupportedOptionError",
    "detect_format",
    "is_archive",
    "get_backend",
    "ZipBackend",
    "TarBackend",
    "GZipBackend",
    "SevenZipBackend",
    "RarBackend",
]
