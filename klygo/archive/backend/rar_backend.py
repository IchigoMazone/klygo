from pathlib import Path
from typing import Iterator, Any, Dict, List, Optional, Union, Literal

from klygo.archive.backend.base import ArchiveBackend
from klygo.archive.human_size import human_size


class RarBackend(ArchiveBackend):
    """
    Rar Archive Backend supporting read/extract operations for .rar format.
    """

    def _check_rarfile(self):
        try:
            import rarfile
            return rarfile
        except ImportError:
            raise ImportError(
                "Support for .rar format requires the 'rarfile' package. "
                "Please install it using 'pip install rarfile'."
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
        raise NotImplementedError("Creating RAR archives is not supported (RAR is a proprietary format). Use .zip or .7z.")

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
        rarfile = self._check_rarfile()
        output_dir.mkdir(parents=True, exist_ok=True)
        with rarfile.RarFile(archive_path, mode="r") as rf:
            if password:
                rf.setpassword(password)
            rf.extractall(path=output_dir)

    def extract_file(
        self,
        archive_path: Path,
        filename: str,
        output_dir: Path,
        password: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        rarfile = self._check_rarfile()
        output_dir.mkdir(parents=True, exist_ok=True)
        with rarfile.RarFile(archive_path, mode="r") as rf:
            if password:
                rf.setpassword(password)
            rf.extract(filename, path=output_dir)

    def list_files(self, archive_path: Path) -> List[str]:
        rarfile = self._check_rarfile()
        with rarfile.RarFile(archive_path, mode="r") as rf:
            return rf.namelist()

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
        rarfile = self._check_rarfile()
        try:
            with rarfile.RarFile(archive_path, mode="r") as rf:
                return rf.testrar() is None
        except Exception as e:
            if raise_exception:
                raise ValueError(f"RAR archive corrupted: {e}")
            return False

    def add(
        self,
        archive_path: Path,
        files: List[Path],
        on_conflict: Literal["rename", "overwrite", "skip"] = "rename",
        verbose: bool = True,
    ) -> None:
        raise NotImplementedError("Modifying RAR archive is not supported.")

    def remove(self, archive_path: Path, files: List[str]) -> None:
        raise NotImplementedError("Modifying RAR archive is not supported.")

    def merge(
        self,
        archive_paths: List[Path],
        output_path: Path,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        raise NotImplementedError("Merging RAR archives is not supported.")

    def split_by_size(
        self,
        archive_path: Path,
        size: float,
        output_dir: Path,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> List[str]:
        raise NotImplementedError("Splitting RAR archive is not supported.")
