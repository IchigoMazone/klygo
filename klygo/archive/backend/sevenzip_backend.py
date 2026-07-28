import shutil
from pathlib import Path
from typing import Iterator, Any, Dict, List, Optional, Union, Literal

from klygo.archive.backend.base import ArchiveBackend
from klygo.archive.human_size import human_size


class SevenZipBackend(ArchiveBackend):
    """
    7z Archive Backend supporting .7z format (requires optional py7zr library).
    """

    def _check_py7zr(self):
        try:
            import py7zr
            return py7zr
        except ImportError:
            raise ImportError(
                "Support for .7z format requires the 'py7zr' package. "
                "Please install it using 'pip install py7zr'."
            )

    def compress(
        self,
        source: Path,
        output_path: Path,
        compresslevel: int = 6,
        method: Optional[str] = None,
        preserve_timestamp: bool = True,
        preserve_permissions: bool = True,
        follow_symlinks: bool = False,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        py7zr = self._check_py7zr()
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"output_path already exists: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(output_path, mode="w") as archive:
            if source.is_dir():
                archive.writeall(source, arcname=source.name)
            else:
                archive.write(source, arcname=source.name)

    def extract(
        self,
        archive_path: Path,
        output_dir: Path,
        password: Optional[str] = None,
        include: Optional[Union[str, List[str]]] = None,
        exclude: Optional[Union[str, List[str]]] = None,
        preserve_timestamp: bool = True,
        preserve_permissions: bool = True,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        py7zr = self._check_py7zr()
        output_dir.mkdir(parents=True, exist_ok=True)

        with py7zr.SevenZipFile(archive_path, mode="r", password=password) as archive:
            archive.extractall(path=output_dir)

    def extract_file(
        self,
        archive_path: Path,
        filename: str,
        output_dir: Path,
        password: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        py7zr = self._check_py7zr()
        output_dir.mkdir(parents=True, exist_ok=True)

        with py7zr.SevenZipFile(archive_path, mode="r", password=password) as archive:
            archive.extract(path=output_dir, targets=[filename])

    def list_files(self, archive_path: Path) -> List[str]:
        py7zr = self._check_py7zr()
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            return archive.getnames()

    def iter_files(self, archive_path: Path) -> Iterator[str]:
        yield from self.list_files(archive_path)

    def search(
        self,
        archive_path: Path,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
    ) -> List[str]:
        import fnmatch, re
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
        py7zr = self._check_py7zr()
        try:
            with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                return archive.test()
        except Exception as e:
            if raise_exception:
                raise ValueError(f"7z archive corrupted: {e}")
            return False

    def add(
        self,
        archive_path: Path,
        files: List[Path],
        on_conflict: Literal["rename", "overwrite", "skip"] = "rename",
        verbose: bool = True,
    ) -> None:
        py7zr = self._check_py7zr()
        with py7zr.SevenZipFile(archive_path, mode="a") as archive:
            for fp in files:
                archive.write(fp, arcname=fp.name)

    def remove(self, archive_path: Path, files: List[str]) -> None:
        raise NotImplementedError("Removing files from 7z archive is not directly supported.")

    def merge(
        self,
        archive_paths: List[Path],
        output_path: Path,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        raise NotImplementedError("Merging 7z archives is not supported directly.")

    def split_by_size(
        self,
        archive_path: Path,
        size: float,
        output_dir: Path,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> List[str]:
        raise NotImplementedError("Splitting 7z archive is not supported.")
