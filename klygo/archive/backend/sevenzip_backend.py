"""Optional 7-Zip backend powered by :mod:`py7zr`."""

import fnmatch
import re
from pathlib import Path
from typing import Iterator, Any, Dict, List, Optional, Union

import klygo.files as file_utils
from klygo.archive.backend.base import ArchiveBackend, BackendCapabilities
from klygo.utils.formatting import human_size
from klygo.utils.progress import ProgressBar


class SevenZipBackend(ArchiveBackend):
    """Read and create 7-Zip archives through the optional ``py7zr`` package.

    The adapter can be imported and inspected without ``py7zr`` installed.
    Operations that open a 7-Zip archive raise an actionable :class:`ImportError`
    when the dependency is unavailable. Mutation operations are intentionally
    disabled until they can provide the same conflict and atomicity guarantees
    as ZIP and TAR.
    """

    format_name = "7z"
    capabilities = BackendCapabilities(
        compress=True,
        compress_options=frozenset({"include_root"}),
        extract_options=frozenset({"password"}),
    )

    def _check_py7zr(self):
        try:
            import py7zr
            return py7zr
        except ImportError:
            raise ImportError(
                "Support for .7z format requires the 'py7zr' package. "
                "Install it using 'pip install \"klygo[py7zr]\"'."
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
        """Create a 7-Zip archive through ``py7zr``.

        ``include_root`` controls directory layout. Changed compression method,
        level, and symlink options are rejected until implemented explicitly.
        """
        self.validate_option("compress", "compresslevel", compresslevel, 6)
        self.validate_option("compress", "method", method, None)
        self.validate_option("compress", "follow_symlinks", follow_symlinks, False)
        py7zr = self._check_py7zr()
        if file_utils.exists(output_path) and not overwrite:
            raise FileExistsError(f"output_path already exists: {output_path}")

        file_utils.mkdir(file_utils.parent(output_path))
        with ProgressBar(total=1, desc=f"{output_path.name}: compressing", verbose=verbose) as progress:
            with py7zr.SevenZipFile(output_path, mode="w") as archive:
                if source.is_dir():
                    if include_root:
                        archive.writeall(source, arcname=source.name)
                    else:
                        for child in sorted(source.iterdir()):
                            if child.is_dir():
                                archive.writeall(child, arcname=child.name)
                            else:
                                archive.write(child, arcname=child.name)
                else:
                    archive.write(source, arcname=source.name)
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
        """Extract every 7-Zip member, optionally using a password.

        Include and exclude filters are currently unsupported and changed
        values raise ``UnsupportedOptionError``.
        """
        self.validate_option("extract", "include", include, None)
        self.validate_option("extract", "exclude", exclude, None)
        py7zr = self._check_py7zr()
        file_utils.mkdir(output_dir)

        with py7zr.SevenZipFile(archive_path, mode="r", password=password) as archive:
            names = archive.getnames()
            if not overwrite:
                existing = [name for name in names if file_utils.exists(output_dir / name)]
                if existing:
                    raise FileExistsError(
                        f"Archive members already exist: {existing[:5]}. Use overwrite=True."
                    )
            with ProgressBar(total=1, desc=f"{archive_path.name}: extracting", verbose=verbose) as progress:
                archive.extractall(path=output_dir)
                progress.update()

    def extract_file(
        self,
        archive_path: Path,
        filename: str,
        output_dir: Path,
        password: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """Extract one exact 7-Zip member through ``py7zr``."""
        py7zr = self._check_py7zr()
        file_utils.mkdir(output_dir)
        target = output_dir / file_utils.name(filename)
        if file_utils.exists(target) and not overwrite:
            raise FileExistsError(f"File already exists: {target}. Use overwrite=True.")

        with py7zr.SevenZipFile(archive_path, mode="r", password=password) as archive:
            archive.extract(path=output_dir, targets=[filename])

    def list_files(self, archive_path: Path) -> List[str]:
        """Return member names reported by ``py7zr``."""
        py7zr = self._check_py7zr()
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            return archive.getnames()

    def iter_files(self, archive_path: Path) -> Iterator[str]:
        """Yield 7-Zip member names from the managed list result."""
        yield from self.list_files(archive_path)

    def search(
        self,
        archive_path: Path,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
    ) -> List[str]:
        """Search 7-Zip member names using glob or regex matching."""
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
        """Normalize 7-Zip metadata, sizes, encryption, and member counts."""
        py7zr = self._check_py7zr()
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            info = archive.archiveinfo()
            files = archive.getnames()
            size = archive_path.stat().st_size
            return {
                "path": str(archive_path),
                "format": "7z",
                "compression_algorithm": "lzma2",
                "encrypted": archive.password_protected,
                "comment": "",
                "file_count": len(files),
                "directory_count": 0,
                "uncompressed_size": info.uncompressed if hasattr(info, "uncompressed") else size,
                "human_uncompressed_size": human_size(info.uncompressed) if hasattr(info, "uncompressed") else human_size(size),
                "compressed_size": size,
                "human_compressed_size": human_size(size),
                "compress_ratio": 0.0,
                "archive_size": size,
                "human_archive_size": human_size(size),
                "largest_file": files[0] if files else None,
                "smallest_file": files[-1] if files else None,
            }

    def test(self, archive_path: Path, raise_exception: bool = False) -> bool:
        """Delegate archive integrity testing to ``py7zr``."""
        py7zr = self._check_py7zr()
        try:
            with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                return archive.test()
        except Exception as e:
            if raise_exception:
                raise ValueError(f"7z archive corrupted: {e}")
            return False
