"""Optional read-only RAR backend powered by :mod:`rarfile`."""

import fnmatch
import re
from pathlib import Path
from typing import Iterator, Any, Dict, List, Optional, Union

import klygo.files as file_utils
from klygo.archive.backend.base import ArchiveBackend, BackendCapabilities
from klygo.utils.formatting import human_size
from klygo.utils.progress import ProgressBar


class RarBackend(ArchiveBackend):
    """Inspect and extract RAR archives through optional ``rarfile`` support.

    The backend is read-only and advertises no creation or mutation capability.
    Passwords are accepted for extraction. Importing and inspecting the class
    does not require ``rarfile``; archive operations provide an actionable
    :class:`ImportError` if the dependency is unavailable.
    """

    format_name = "rar"
    capabilities = BackendCapabilities(
        extract_options=frozenset({"password"}),
    )

    def _check_rarfile(self):
        try:
            import rarfile
            return rarfile
        except ImportError:
            raise ImportError(
                "Support for .rar format requires the 'rarfile' package. "
                "Install it using 'pip install \"klygo[rarfile]\"'."
            )

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
        """Extract every RAR member, optionally with a password.

        Changed include or exclude filters are rejected because selective RAR
        extraction is not yet part of this backend's public contract.
        """
        self.validate_option("extract", "include", include, None)
        self.validate_option("extract", "exclude", exclude, None)
        rarfile = self._check_rarfile()
        file_utils.mkdir(output_dir)
        with rarfile.RarFile(archive_path, mode="r") as rf:
            if password:
                rf.setpassword(password)
            names = rf.namelist()
            if not overwrite:
                existing = [name for name in names if file_utils.exists(output_dir / name)]
                if existing:
                    raise FileExistsError(
                        f"Archive members already exist: {existing[:5]}. Use overwrite=True."
                    )
            with ProgressBar(total=1, desc=f"{archive_path.name}: extracting", verbose=verbose) as progress:
                rf.extractall(path=output_dir)
                progress.update()

    def extract_file(
        self,
        archive_path: Path,
        filename: str,
        output_dir: Path,
        password: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """Extract one exact RAR member while enforcing overwrite policy."""
        rarfile = self._check_rarfile()
        file_utils.mkdir(output_dir)
        target = output_dir / file_utils.name(filename)
        if file_utils.exists(target) and not overwrite:
            raise FileExistsError(f"File already exists: {target}. Use overwrite=True.")
        with rarfile.RarFile(archive_path, mode="r") as rf:
            if password:
                rf.setpassword(password)
            rf.extract(filename, path=output_dir)

    def list_files(self, archive_path: Path) -> List[str]:
        """Return member names reported by ``rarfile``."""
        rarfile = self._check_rarfile()
        with rarfile.RarFile(archive_path, mode="r") as rf:
            return rf.namelist()

    def iter_files(self, archive_path: Path) -> Iterator[str]:
        """Yield RAR member names from the managed list result."""
        yield from self.list_files(archive_path)

    def search(
        self,
        archive_path: Path,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
    ) -> List[str]:
        """Search RAR member names with glob or regular-expression matching."""
        results = []
        for name in self.iter_files(archive_path):
            n = name if case_sensitive else name.lower()
            p = pattern if case_sensitive else pattern.lower()
            if regex and re.search(p, n):
                results.append(name)
            elif not regex and fnmatch.fnmatch(n, p):
                results.append(name)
        return results

    def get_info(self, archive_path: Path) -> Dict[str, Any]:
        """Normalize RAR sizes, counts, encryption state, and archive comment."""
        rarfile = self._check_rarfile()
        with rarfile.RarFile(archive_path, mode="r") as rf:
            infolist = rf.infolist()
            total_uncompressed = sum(info.file_size for info in infolist)
            total_compressed = sum(info.compress_size for info in infolist)
            size = archive_path.stat().st_size
            return {
                "path": str(archive_path),
                "format": "rar",
                "compression_algorithm": "rar",
                "encrypted": rf.needs_password(),
                "comment": rf.comment or "",
                "file_count": len(infolist),
                "directory_count": 0,
                "uncompressed_size": total_uncompressed,
                "human_uncompressed_size": human_size(total_uncompressed),
                "compressed_size": total_compressed,
                "human_compressed_size": human_size(total_compressed),
                "compress_ratio": round((1 - total_compressed / total_uncompressed) * 100, 2) if total_uncompressed > 0 else 0.0,
                "archive_size": size,
                "human_archive_size": human_size(size),
                "largest_file": None,
                "smallest_file": None,
            }

    def test(self, archive_path: Path, raise_exception: bool = False) -> bool:
        """Run ``rarfile`` integrity testing and optionally raise on failure."""
        rarfile = self._check_rarfile()
        try:
            with rarfile.RarFile(archive_path, mode="r") as rf:
                return rf.testrar() is None
        except Exception as e:
            if raise_exception:
                raise ValueError(f"RAR archive corrupted: {e}")
            return False
