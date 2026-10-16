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

    Returns
    -------
    ArchiveFile
        Initialized archive wrapper.

    Raises
    ------
    ValueError
        If the archive format is unsupported.

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
        """Return all stored member names.

        Parameters
        ----------
        None
            Uses the archive bound to this instance.

        Returns
        -------
        list[str]
            Stored member names.

        Raises
        ------
        FileNotFoundError
            If the archive file is missing.

        Examples
        --------
        >>> opened = ArchiveFile("dataset.zip")
        >>> names = opened.list_files()
        """
        return self.backend.list_files(self.archive_path)

    def iter_files(self) -> Iterator[str]:
        """Yield stored member names lazily.

        Parameters
        ----------
        None
            Uses the archive bound to this instance.

        Returns
        -------
        Iterator[str]
            Lazy stream of stored member names.

        Raises
        ------
        FileNotFoundError
            If the archive file is missing when iterated.

        Examples
        --------
        >>> opened = ArchiveFile("dataset.zip")
        >>> names = list(opened.iter_files())
        """
        yield from self.backend.iter_files(self.archive_path)

    def search(
        self,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
    ) -> List[str]:
        """Search member names using a glob or regular expression.

        Parameters
        ----------
        pattern : str
            Glob pattern or regular expression.
        regex : bool, default=False
            Interpret ``pattern`` as a regular expression.
        case_sensitive : bool, default=True
            Match letter case exactly.

        Returns
        -------
        list[str]
            Matching member names.

        Raises
        ------
        FileNotFoundError
            If the archive file is missing.

        Examples
        --------
        >>> opened = ArchiveFile("dataset.zip")
        >>> names = opened.search("images/*.jpg")
        """
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
        """Extract all archive members into ``output_dir``.

        Parameters
        ----------
        output_dir : str or pathlib.Path, default='.'
            Destination directory.
        password : str or None, default=None
            Password for supported encrypted archives.
        overwrite : bool, default=False
            Replace existing output files.
        verbose : bool, default=True
            Display extraction progress.

        Returns
        -------
        None
            Files are extracted as a side effect.

        Raises
        ------
        FileNotFoundError
            If the archive file is missing.
        FileExistsError
            If an output file exists and ``overwrite`` is false.

        Examples
        --------
        >>> opened = ArchiveFile("dataset.zip")
        >>> opened.extract("output")
        """
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
        """Extract one member into ``output_dir`` using streaming I/O.

        Parameters
        ----------
        filename : str
            Name of the member to extract.
        output_dir : str or pathlib.Path, default='.'
            Destination directory.
        password : str or None, default=None
            Password for supported encrypted archives.
        overwrite : bool, default=False
            Replace an existing output file.

        Returns
        -------
        None
            The member is extracted as a side effect.

        Raises
        ------
        FileNotFoundError
            If the archive or requested member is missing.
        FileExistsError
            If the output exists and ``overwrite`` is false.

        Examples
        --------
        >>> opened = ArchiveFile("dataset.zip")
        >>> opened.extract_file("images/cat.jpg", "output")
        """
        self.backend.validate_option("extract", "password", password, None)
        self.backend.extract_file(
            self.archive_path,
            filename,
            files.path(output_dir),
            password=password,
            overwrite=overwrite,
        )

    def get_info(self) -> Dict[str, Any]:
        """Return archive metadata and compression statistics.

        Parameters
        ----------
        None
            Uses the archive bound to this instance.

        Returns
        -------
        dict[str, Any]
            Archive format, size, member count, and compression details.

        Raises
        ------
        FileNotFoundError
            If the archive file is missing.

        Examples
        --------
        >>> opened = ArchiveFile("dataset.zip")
        >>> info = opened.get_info()
        """
        return self.backend.get_info(self.archive_path)

    def test(self, raise_exception: bool = False) -> bool:
        """Run the backend integrity check.

        Parameters
        ----------
        raise_exception : bool, default=False
            Raise on integrity failure instead of returning false.

        Returns
        -------
        bool
            Whether the archive passes its integrity check.

        Raises
        ------
        Exception
            If integrity validation fails and ``raise_exception`` is true;
            the exception type depends on the archive backend.

        Examples
        --------
        >>> opened = ArchiveFile("dataset.zip")
        >>> valid = opened.test()
        """
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

    Raises
    ------
    ValueError
        If the archive format is unsupported.

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
