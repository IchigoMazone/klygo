import gzip
import shutil
from pathlib import Path
from typing import Iterator, Any, Dict, List, Optional, Union, Literal

from klygo.archive.backend.base import ArchiveBackend
from klygo.archive.human_size import human_size


class GZipBackend(ArchiveBackend):
    """
    GZip Archive Backend supporting single file .gz compression/decompression.
    """

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
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"output_path already exists: {output_path}")

        if source.is_dir():
            raise ValueError("GZip backend only supports single files. For directories use tar.gz format.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(source, "rb") as f_in, gzip.open(output_path, "wb", compresslevel=compresslevel) as f_out:
            shutil.copyfileobj(f_in, f_out)

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
        output_dir.mkdir(parents=True, exist_ok=True)
        out_name = archive_path.stem if archive_path.suffix.lower() == ".gz" else archive_path.name
        target = output_dir / out_name

        if target.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {target}. Use overwrite=True.")

        with gzip.open(archive_path, "rb") as f_in, open(target, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    def extract_file(
        self,
        archive_path: Path,
        filename: str,
        output_dir: Path,
        password: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        self.extract(archive_path, output_dir, password=password, overwrite=overwrite, verbose=False)

    def list_files(self, archive_path: Path) -> List[str]:
        out_name = archive_path.stem if archive_path.suffix.lower() == ".gz" else archive_path.name
        return [out_name]

    def iter_files(self, archive_path: Path) -> Iterator[str]:
        yield from self.list_files(archive_path)

    def search(
        self,
        archive_path: Path,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
    ) -> List[str]:
        files = self.list_files(archive_path)
        return files if pattern in ("*", files[0]) else []

    def get_info(self, archive_path: Path) -> Dict[str, Any]:
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
        try:
            with gzip.open(archive_path, "rb") as f:
                while f.read(1024 * 1024):
                    pass
            return True
        except Exception as e:
            if raise_exception:
                raise ValueError(f"GZip archive corrupted: {e}")
            return False

    def add(
        self,
        archive_path: Path,
        files: List[Path],
        on_conflict: Literal["rename", "overwrite", "skip"] = "rename",
        verbose: bool = True,
    ) -> None:
        raise NotImplementedError("GZip backend does not support adding files to an existing .gz file.")

    def remove(self, archive_path: Path, files: List[str]) -> None:
        raise NotImplementedError("GZip backend does not support removing files from a .gz file.")

    def merge(
        self,
        archive_paths: List[Path],
        output_path: Path,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        raise NotImplementedError("GZip backend does not support merging multiple .gz files directly.")

    def split_by_size(
        self,
        archive_path: Path,
        size: float,
        output_dir: Path,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> List[str]:
        raise NotImplementedError("GZip backend does not support splitting.")
