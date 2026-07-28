import fnmatch
import re
import shutil
import tarfile
from pathlib import Path
from typing import Iterator, Any, Dict, List, Optional, Union, Literal

from klygo.archive.backend.base import ArchiveBackend
from klygo.archive.human_size import human_size
from klygo.archive.progress import ArchiveProgress


def _is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """Check for Zip Slip / Path Traversal vulnerability."""
    try:
        target_path.resolve().relative_to(base_dir.resolve())
        return True
    except ValueError:
        return False


def _get_tar_mode(fmt: str, write: bool = False) -> str:
    action = "w" if write else "r"
    if fmt in ("tar.gz", "tgz"):
        return f"{action}:gz"
    if fmt in ("tar.xz", "txz"):
        return f"{action}:xz"
    if fmt in ("tar.bz2", "tbz2"):
        return f"{action}:bz2"
    return f"{action}:*" if not write else "w:"


class TarBackend(ArchiveBackend):
    """
    Tar Archive Backend supporting TAR, TAR.GZ, TAR.XZ, TAR.BZ2 formats.
    """

    def __init__(self, format_name: str = "tar"):
        self.format_name = format_name

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

        mode = _get_tar_mode(self.format_name, write=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _file_generator():
            if source.is_dir():
                for p in source.rglob("*"):
                    if p.is_file() or (not follow_symlinks and p.is_symlink()):
                        yield p
            else:
                yield source

        files_to_compress = list(_file_generator()) if verbose else None
        total_count = len(files_to_compress) if files_to_compress else 1

        with ArchiveProgress(total=total_count, desc=f"{output_path.name}: compressing", verbose=verbose) as pbar:
            with tarfile.open(output_path, mode=mode) as tar:
                gen = files_to_compress if files_to_compress is not None else _file_generator()
                for file_path in gen:
                    arcname = (
                        file_path.relative_to(source.parent)
                        if source.is_dir()
                        else file_path.name
                    )
                    tar.add(file_path, arcname=str(arcname), recursive=False)
                    pbar.update(1)

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
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        includes = [include] if isinstance(include, str) else (include or [])
        excludes = [exclude] if isinstance(exclude, str) else (exclude or [])

        mode = _get_tar_mode(self.format_name, write=False)
        with tarfile.open(archive_path, mode=mode) as tar:
            members = tar.getmembers()

            filtered_members = []
            for m in members:
                name = m.name
                if includes and not any(fnmatch.fnmatch(name, pat) for pat in includes):
                    continue
                if excludes and any(fnmatch.fnmatch(name, pat) for pat in excludes):
                    continue
                filtered_members.append(m)

            if not overwrite:
                existing = [m for m in filtered_members if (output_dir / m.name).exists()]
                if existing:
                    names = ", ".join(m.name for m in existing[:5])
                    suffix = f"… (+{len(existing) - 5} more)" if len(existing) > 5 else ""
                    raise FileExistsError(f"Files already exist in output directory: {names}{suffix}. Use overwrite=True.")

            with ArchiveProgress(total=len(filtered_members), desc=f"{archive_path.name}: extracting", verbose=verbose) as pbar:
                for member in filtered_members:
                    target_path = output_dir / member.name
                    if not _is_safe_path(output_dir, target_path):
                        raise ValueError(f"Unsafe path detected in archive (Zip Slip): '{member.name}'")
                    tar.extract(member, path=output_dir)
                    pbar.update(1)

    def extract_file(
        self,
        archive_path: Path,
        filename: str,
        output_dir: Path,
        password: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        mode = _get_tar_mode(self.format_name, write=False)

        with tarfile.open(archive_path, mode=mode) as tar:
            try:
                member = tar.getmember(filename)
            except KeyError:
                raise KeyError(f"'{filename}' not found in archive. Use list_files() to see available files.")

            target = output_dir / Path(filename).name
            if target.exists() and not overwrite:
                raise FileExistsError(f"File already exists: {target}. Use overwrite=True.")

            src = tar.extractfile(member)
            if src is None:
                raise ValueError(f"Cannot extract member '{filename}' (may be directory).")
            with src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

    def list_files(self, archive_path: Path) -> List[str]:
        mode = _get_tar_mode(self.format_name, write=False)
        with tarfile.open(archive_path, mode=mode) as tar:
            return tar.getnames()

    def iter_files(self, archive_path: Path) -> Iterator[str]:
        mode = _get_tar_mode(self.format_name, write=False)
        with tarfile.open(archive_path, mode=mode) as tar:
            for member in tar:
                yield member.name

    def search(
        self,
        archive_path: Path,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
    ) -> List[str]:
        results = []
        for name in self.iter_files(archive_path):
            n = name if case_sensitive else name.lower()
            p = pattern if case_sensitive else pattern.lower()
            match = bool(re.search(p, n)) if regex else fnmatch.fnmatch(n, p)
            if match:
                results.append(name)
        return results

    def get_info(self, archive_path: Path) -> Dict[str, Any]:
        mode = _get_tar_mode(self.format_name, write=False)
        with tarfile.open(archive_path, mode=mode) as tar:
            members = tar.getmembers()
            files = [m for m in members if m.isfile()]
            dirs = [m for m in members if m.isdir()]

            total_uncompressed = sum(m.size for m in members)
            archive_size = archive_path.stat().st_size

            largest = max(files, key=lambda x: x.size).name if files else None
            smallest = min(files, key=lambda x: x.size).name if files else None

            return {
                "path": str(archive_path),
                "format": self.format_name,
                "compression_algorithm": self.format_name,
                "encrypted": False,
                "comment": "",
                "file_count": len(files),
                "directory_count": len(dirs),
                "uncompressed_size": total_uncompressed,
                "human_uncompressed_size": human_size(total_uncompressed),
                "compressed_size": archive_size,
                "human_compressed_size": human_size(archive_size),
                "compress_ratio": round((1 - archive_size / total_uncompressed) * 100, 2) if total_uncompressed > 0 else 0.0,
                "archive_size": archive_size,
                "human_archive_size": human_size(archive_size),
                "largest_file": largest,
                "smallest_file": smallest,
            }

    def test(self, archive_path: Path, raise_exception: bool = False) -> bool:
        mode = _get_tar_mode(self.format_name, write=False)
        try:
            with tarfile.open(archive_path, mode=mode) as tar:
                for member in tar:
                    pass
            return True
        except Exception as e:
            if raise_exception:
                raise ValueError(f"Tar archive corrupted: {e}")
            return False

    def add(
        self,
        archive_path: Path,
        files: List[Path],
        on_conflict: Literal["rename", "overwrite", "skip"] = "rename",
        verbose: bool = True,
    ) -> None:
        # TAR appends entries cleanly in write/append mode
        mode = "a:" if self.format_name == "tar" else _get_tar_mode(self.format_name, write=True)
        with tarfile.open(archive_path, mode=mode) as tar:
            with ArchiveProgress(total=len(files), desc=f"{archive_path.name}: adding", verbose=verbose) as pbar:
                for fp in files:
                    tar.add(fp, arcname=fp.name)
                    pbar.update(1)

    def remove(self, archive_path: Path, files: List[str]) -> None:
        to_remove = set(files)
        mode_read = _get_tar_mode(self.format_name, write=False)
        mode_write = _get_tar_mode(self.format_name, write=True)

        tmp_path = archive_path.with_suffix(".tmp.tar")
        with tarfile.open(archive_path, mode=mode_read) as src_tar, tarfile.open(tmp_path, mode=mode_write) as dst_tar:
            for member in src_tar.getmembers():
                if member.name not in to_remove:
                    fobj = src_tar.extractfile(member) if member.isfile() else None
                    dst_tar.addfile(member, fobj)

        tmp_path.replace(archive_path)

    def merge(
        self,
        archive_paths: List[Path],
        output_path: Path,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"output_path already exists: {output_path}")

        written_names = set()
        mode_write = _get_tar_mode(self.format_name, write=True)
        with tarfile.open(output_path, mode=mode_write) as dst_tar:
            with ArchiveProgress(total=len(archive_paths), desc=f"{output_path.name}: merging", verbose=verbose) as pbar:
                for src in archive_paths:
                    mode_read = _get_tar_mode(src.suffix.lstrip("."), write=False)
                    with tarfile.open(src, mode=mode_read) as src_tar:
                        for member in src_tar.getmembers():
                            if member.name in written_names:
                                continue
                            written_names.add(member.name)
                            fobj = src_tar.extractfile(member) if member.isfile() else None
                            dst_tar.addfile(member, fobj)
                    pbar.update(1)


    def split_by_size(
        self,
        archive_path: Path,
        size: float,
        output_dir: Path,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> List[str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        max_bytes = int(size * 1024 * 1024)

        stem = archive_path.stem
        suffix = archive_path.suffix
        parts: List[str] = []
        part_num = 1

        mode_read = _get_tar_mode(self.format_name, write=False)
        mode_write = _get_tar_mode(self.format_name, write=True)

        with tarfile.open(archive_path, mode=mode_read) as src_tar:
            members = src_tar.getmembers()
            with ArchiveProgress(total=len(members), desc=f"{archive_path.name}: splitting", verbose=verbose) as pbar:
                current_members: List = []
                current_size = 0

                def _flush() -> None:
                    nonlocal part_num
                    part_path = output_dir / f"{stem}_part_{part_num:03d}{suffix}"
                    if part_path.exists() and not overwrite:
                        raise FileExistsError(f"Output already exists: {part_path}. Use overwrite=True.")
                    with tarfile.open(part_path, mode=mode_write) as ptar:
                        for m in current_members:
                            fobj = src_tar.extractfile(m) if m.isfile() else None
                            ptar.addfile(m, fobj)
                    parts.append(str(part_path))
                    part_num += 1

                for member in members:
                    if current_members and (current_size + member.size) > max_bytes:
                        _flush()
                        current_members = []
                        current_size = 0
                    current_members.append(member)
                    current_size += member.size
                    pbar.update(1)

                if current_members:
                    _flush()

        return parts
