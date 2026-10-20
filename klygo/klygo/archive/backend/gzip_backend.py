"""Built-in single-stream GZip backend."""

import fnmatch
import gzip
import re
import shutil
from pathlib import Path
from typing import Iterator, Any, Dict, List, Optional, Union

import klygo.files as file_utils
from klygo.archive.backend.base import ArchiveBackend, BackendCapabilities
from klygo.utils.formatting import human_size
from klygo.utils.progress import ProgressBar


class GZipBackend(ArchiveBackend):
    """Compress and extract one file as a GZip stream.

    GZip is not a multi-member filesystem container in this API. The logical
    member name is derived from the archive filename by removing ``.gz``.
    Creation and extraction are supported; add, remove, merge, and split are
    rejected by the base capability contract. Use :class:`TarBackend` with
    ``format_name="tar.gz"`` for directories or multiple files.

    Attributes
    ----------
    format_name : str
        Always ``"gz"``.
    capabilities : BackendCapabilities
        Enables creation and the ``compresslevel`` option; mutation flags are
        disabled because GZip represents one stream rather than a container.

    Raises
    ------
    ValueError
        For directories, invalid compression levels, corrupt streams in strict
        testing, or a requested filename that is not the logical member.
    UnsupportedOptionError
        If callers change container-only options such as method or filters.
    FileExistsError
        When output exists and overwrite is disabled.

    Notes
    -----
    Metadata cannot know the original uncompressed size without reading the
    complete stream, so the normalized report uses conservative size values.

    Examples
    --------
    >>> from klygo.archive.backend import GZipBackend
    >>> backend = GZipBackend()
    >>> backend.list_files(__import__("pathlib").Path("report.txt.gz"))
    ['report.txt']
    """

    format_name = "gz"
    capabilities = BackendCapabilities(
        compress=True,
        compress_options=frozenset({"compresslevel"}),
    )

    def compress(
        self,
        source: Path,
        output_path: Path,
        compresslevel: int = 6,
        method: Optional[str] = None,
        follow_symlinks: bool = False,
        include_root: bool = True,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        """Compress one regular file into a GZip stream.

        Levels 0 through 9 are supported. Directories and container-specific
        options are rejected instead of being silently ignored.
        """
        self.validate_option("compress", "method", method, None)
        self.validate_option("compress", "follow_symlinks", follow_symlinks, False)
        self.validate_option("compress", "include_root", include_root, True)
        if not 0 <= compresslevel <= 9:
            raise ValueError("compresslevel must be between 0 and 9")
        if file_utils.exists(output_path) and not overwrite:
            raise FileExistsError(f"output_path already exists: {output_path}")

        if source.is_dir():
            raise ValueError("GZip supports one file only. Use tar.gz for directories.")

        file_utils.mkdir(file_utils.parent(output_path))
        with ProgressBar(total=1, desc=f"{output_path.name}: compressing", verbose=verbose) as progress:
            with open(source, "rb") as f_in, gzip.open(output_path, "wb", compresslevel=compresslevel) as f_out:
                shutil.copyfileobj(f_in, f_out)
            progress.update()

    def extract(
        self,
        archive_path: Path,
        output_dir: Path,
        password: Optional[str] = None,
        include: Optional[Union[str, List[str]]] = None,
        exclude: Optional[Union[str, List[str]]] = None,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        """Extract the single stream using the archive stem as output name."""
        self.validate_option("extract", "password", password, None)
        self.validate_option("extract", "include", include, None)
        self.validate_option("extract", "exclude", exclude, None)
        file_utils.mkdir(output_dir)
        out_name = archive_path.stem if archive_path.suffix.lower() == ".gz" else archive_path.name
        target = output_dir / out_name

        if file_utils.exists(target) and not overwrite:
            raise FileExistsError(f"File already exists: {target}. Use overwrite=True.")

        with ProgressBar(total=1, desc=f"{archive_path.name}: extracting", verbose=verbose) as progress:
            with gzip.open(archive_path, "rb") as f_in, open(target, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            progress.update()

    def extract_file(
        self,
        archive_path: Path,
        filename: str,
        output_dir: Path,
        password: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """Extract the stream only when ``filename`` matches its logical name."""
        self.validate_option("extract", "password", password, None)
        expected = self.list_files(archive_path)[0]
        if filename != expected:
            raise KeyError(f"'{filename}' not found in archive. Available member: '{expected}'.")
        self.extract(archive_path, output_dir, password=password, overwrite=overwrite, verbose=False)

    def list_files(self, archive_path: Path) -> List[str]:
        """Return the one logical member name derived from ``archive_path``."""
        out_name = archive_path.stem if archive_path.suffix.lower() == ".gz" else archive_path.name
        return [out_name]

    def iter_files(self, archive_path: Path) -> Iterator[str]:
        """Yield the single logical GZip member name."""
        yield from self.list_files(archive_path)

    def search(
        self,
        archive_path: Path,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
    ) -> List[str]:
        """Match the logical member name using a glob or regular expression."""
        names = self.list_files(archive_path)
        name = names[0] if case_sensitive else names[0].lower()
        requested = pattern if case_sensitive else pattern.lower()
        matched = bool(re.search(requested, name)) if regex else fnmatch.fnmatch(name, requested)
        return names if matched else []

    def get_info(self, archive_path: Path) -> Dict[str, Any]:
        """Return normalized metadata for the single GZip stream."""
        archive_size = archive_path.stat().st_size
        return {
            "path": str(archive_path),
            "format": "gz",
            "compression_algorithm": "deflate",
            "encrypted": False,
            "comment": "",
            "file_count": 1,
            "directory_count": 0,
            "uncompressed_size": archive_size,
            "human_uncompressed_size": human_size(archive_size),
            "compressed_size": archive_size,
            "human_compressed_size": human_size(archive_size),
            "compress_ratio": 0.0,
            "archive_size": archive_size,
            "human_archive_size": human_size(archive_size),
            "largest_file": archive_path.stem,
            "smallest_file": archive_path.stem,
        }

    def test(self, archive_path: Path, raise_exception: bool = False) -> bool:
        """Decompress the complete stream to verify framing and checksum data."""
        try:
            with gzip.open(archive_path, "rb") as f:
                while f.read(1024 * 1024):
                    pass
            return True
        except Exception as e:
            if raise_exception:
                raise ValueError(f"GZip archive corrupted: {e}")
            return False
