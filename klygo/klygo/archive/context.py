from pathlib import Path
from typing import Union, List, Iterator, Dict, Any, Optional

import klygo.files as files
from klygo.archive._utils import resolve_backend
from klygo.archive.backend import detect_format


class ArchiveFile:
    """Object-oriented interface to a single archive.

    Format detection and backend selection happen once during construction.
    The object can then list, search, inspect, validate, and extract members.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Existing archive to open.

    Attributes
    ----------
    archive_path : pathlib.Path
        Normalized archive path.
    format : str
        Canonical detected format.
    backend : ArchiveBackend
        Backend responsible for archive operations.

    Examples
    --------
    >>> import klygo.archive as archive
    >>> with archive.open("dataset.zip") as opened:
    ...     print(opened.format)
    ...     names = opened.list_files()
    """

    def __init__(self, archive_path: Union[str, Path]):
        """Initialize the wrapper and select its backend."""
        self.archive_path = files.path(archive_path)
        self.format = detect_format(self.archive_path)
        _, self.backend = resolve_backend(self.archive_path, format_hint=self.format)

    def list_files(self) -> List[str]:
        """Return all stored member names."""
        return self.backend.list_files(self.archive_path)

    def iter_files(self) -> Iterator[str]:
        """Yield stored member names lazily."""
        yield from self.backend.iter_files(self.archive_path)

    def search(
        self,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
    ) -> List[str]:
        """Search member names using a glob or regular expression."""
        return self.backend.search(
            self.archive_path, pattern, regex=regex, case_sensitive=case_sensitive
        )

    def extract(
        self,
        output_dir: Union[str, Path] = ".",
        password: Optional[str] = None,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        """Extract all archive members into ``output_dir``."""
        self.backend.validate_option("extract", "password", password, None)
        self.backend.extract(
            self.archive_path,
            files.path(output_dir),
            password=password,
            overwrite=overwrite,
            verbose=verbose,
        )

    def extract_file(
        self,
        filename: str,
        output_dir: Union[str, Path] = ".",
        password: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """Extract one member into ``output_dir`` using streaming I/O."""
        self.backend.validate_option("extract", "password", password, None)
        self.backend.extract_file(
            self.archive_path,
            filename,
            files.path(output_dir),
            password=password,
            overwrite=overwrite,
        )

    def get_info(self) -> Dict[str, Any]:
        """Return archive metadata and compression statistics."""
        return self.backend.get_info(self.archive_path)

    def test(self, raise_exception: bool = False) -> bool:
        """Run the backend integrity check."""
        return self.backend.test(self.archive_path, raise_exception=raise_exception)

    def __enter__(self) -> "ArchiveFile":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


def open_archive(archive_path: Union[str, Path]) -> ArchiveFile:
    """Create an ``ArchiveFile`` context-manager wrapper.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Archive to inspect or modify.

    Returns
    -------
    ArchiveFile
        Context-manager wrapper for the archive.

    See Also
    --------
    ArchiveFile

    Examples
    --------
    >>> import klygo.archive as archive
    >>> with archive.open("dataset.zip") as opened:
    ...     names = opened.list_files()
    """
    return ArchiveFile(archive_path)
