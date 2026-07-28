import fnmatch
import re
import shutil
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED, ZIP_STORED, ZIP_BZIP2, ZIP_LZMA
from typing import Iterator, Any, Dict, List, Optional, Union, Literal

from klygo.archive.backend.base import ArchiveBackend
from klygo.archive.human_size import human_size
from klygo.archive.progress import ArchiveProgress


def _is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """Check for Zip Slip vulnerability (Path Traversal)."""
    try:
        target_path.resolve().relative_to(base_dir.resolve())
        return True
    except ValueError:
        return False


def _match_pattern(name: str, pattern: str, regex: bool = False, case_sensitive: bool = True) -> bool:
    if not case_sensitive:
        name = name.lower()
        pattern = pattern.lower()

    if regex:
        return bool(re.search(pattern, name))
    return fnmatch.fnmatch(name, pattern)


class ZipBackend(ArchiveBackend):
    """
    Zip Archive Backend supporting ZIP format.
    """

    COMPRESSION_METHODS = {
        "deflated": ZIP_DEFLATED,
        "stored": ZIP_STORED,
        "bzip2": ZIP_BZIP2,
        "lzma": ZIP_LZMA,
    }

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

        comp_type = self.COMPRESSION_METHODS.get((method or "deflated").lower(), ZIP_DEFLATED)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generator traversal to avoid loading all Paths into RAM
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
            with ZipFile(output_path, mode="w", compression=comp_type, compresslevel=compresslevel) as zf:
                gen = files_to_compress if files_to_compress is not None else _file_generator()
                for file_path in gen:
                    arcname = (
                        file_path.relative_to(source.parent)
                        if source.is_dir()
                        else file_path.name
                    )
                    zf.write(file_path, arcname=str(arcname))
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
        pwd_bytes = password.encode("utf-8") if password else None
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        includes = [include] if isinstance(include, str) else (include or [])
        excludes = [exclude] if isinstance(exclude, str) else (exclude or [])

        with ZipFile(archive_path, mode="r") as zf:
            members = zf.infolist()

            # Filter members
            filtered_members = []
            for m in members:
                name = m.filename
                if includes and not any(fnmatch.fnmatch(name, pat) for pat in includes):
                    continue
                if excludes and any(fnmatch.fnmatch(name, pat) for pat in excludes):
                    continue
                filtered_members.append(m)

            if not overwrite:
                existing = [m for m in filtered_members if (output_dir / m.filename).exists()]
                if existing:
                    names = ", ".join(m.filename for m in existing[:5])
                    suffix = f"… (+{len(existing) - 5} more)" if len(existing) > 5 else ""
                    raise FileExistsError(f"Files already exist in output directory: {names}{suffix}. Use overwrite=True.")

            with ArchiveProgress(total=len(filtered_members), desc=f"{archive_path.name}: extracting", verbose=verbose) as pbar:
                for member in filtered_members:
                    target_path = output_dir / member.filename
                    if not _is_safe_path(output_dir, target_path):
                        raise ValueError(f"Unsafe path detected in archive (Zip Slip): '{member.filename}'")
                    zf.extract(member, path=output_dir, pwd=pwd_bytes)
                    pbar.update(1)

    def extract_file(
        self,
        archive_path: Path,
        filename: str,
        output_dir: Path,
        password: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        pwd_bytes = password.encode("utf-8") if password else None
        output_dir.mkdir(parents=True, exist_ok=True)

        with ZipFile(archive_path, mode="r") as zf:
            try:
                member = zf.getinfo(filename)
            except KeyError:
                raise KeyError(f"'{filename}' not found in archive. Use list_files() to see available files.")

            target = output_dir / Path(filename).name
            if target.exists() and not overwrite:
                raise FileExistsError(f"File already exists: {target}. Use overwrite=True.")

            with zf.open(member, pwd=pwd_bytes) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

    def list_files(self, archive_path: Path) -> List[str]:
        with ZipFile(archive_path, mode="r") as zf:
            return zf.namelist()

    def iter_files(self, archive_path: Path) -> Iterator[str]:
        with ZipFile(archive_path, mode="r") as zf:
            for info in zf.infolist():
                yield info.filename

    def search(
        self,
        archive_path: Path,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
    ) -> List[str]:
        results = []
        for name in self.iter_files(archive_path):
            if _match_pattern(name, pattern, regex=regex, case_sensitive=case_sensitive):
                results.append(name)
        return results

    def get_info(self, archive_path: Path) -> Dict[str, Any]:
        with ZipFile(archive_path, mode="r") as zf:
            members = zf.infolist()
            files = [m for m in members if not m.is_dir()]
            dirs = [m for m in members if m.is_dir()]

            total_uncompressed = sum(m.file_size for m in members)
            total_compressed = sum(m.compress_size for m in members)
            archive_size = archive_path.stat().st_size

            ratio = (
                round((1 - total_compressed / total_uncompressed) * 100, 2)
                if total_uncompressed > 0
                else 0.0
            )

            is_encrypted = any(m.flag_bits & 0x1 > 0 for m in members)
            largest = max(files, key=lambda x: x.file_size).filename if files else None
            smallest = min(files, key=lambda x: x.file_size).filename if files else None

            methods = set(m.compress_type for m in members)
            algo = "DEFLATED" if 8 in methods else ("STORED" if 0 in methods else "MIXED")

            return {
                "path": str(archive_path),
                "format": "zip",
                "compression_algorithm": algo,
                "encrypted": is_encrypted,
                "comment": zf.comment.decode("utf-8", errors="ignore"),
                "file_count": len(files),
                "directory_count": len(dirs),
                "uncompressed_size": total_uncompressed,
                "human_uncompressed_size": human_size(total_uncompressed),
                "compressed_size": total_compressed,
                "human_compressed_size": human_size(total_compressed),
                "compress_ratio": ratio,
                "archive_size": archive_size,
                "human_archive_size": human_size(archive_size),
                "largest_file": largest,
                "smallest_file": smallest,
            }

    def test(self, archive_path: Path, raise_exception: bool = False) -> bool:
        with ZipFile(archive_path, mode="r") as zf:
            bad_file = zf.testzip()

        if bad_file is not None:
            if raise_exception:
                raise ValueError(f"Archive is corrupted. First bad file: '{bad_file}'")
            return False
        return True

    def add(
        self,
        archive_path: Path,
        files: List[Path],
        on_conflict: Literal["rename", "overwrite", "skip"] = "rename",
        verbose: bool = True,
    ) -> None:
        all_files: List[tuple[Path, str]] = []
        for fp in files:
            if fp.is_dir():
                for child in sorted(fp.rglob("*")):
                    if child.is_file():
                        all_files.append((child, str(child.relative_to(fp.parent))))
            else:
                all_files.append((fp, fp.name))

        # Build final list of (abs_path, arcname) resolving conflicts
        with ZipFile(archive_path, mode="r") as zf:
            existing = set(zf.namelist())

        resolved: List[tuple[Path, str]] = []
        to_overwrite: set[str] = set()
        for abs_path, arcname in all_files:
            if arcname in existing:
                if on_conflict == "skip":
                    continue
                elif on_conflict == "rename":
                    stem = Path(arcname).stem
                    suffix = Path(arcname).suffix
                    arcname = f"{stem}_dup{suffix}"
                elif on_conflict == "overwrite":
                    to_overwrite.add(arcname)
            resolved.append((abs_path, arcname))

        if not resolved:
            return

        # If overwriting entries, must rebuild archive
        if to_overwrite:
            tmp_path = archive_path.with_suffix(".tmp.zip")
            with ZipFile(archive_path, mode="r") as src_zf, \
                 ZipFile(tmp_path, mode="w", compression=ZIP_DEFLATED) as dst_zf:
                # Copy existing entries not being overwritten
                for item in src_zf.infolist():
                    if item.filename not in to_overwrite:
                        with src_zf.open(item) as src, dst_zf.open(item, mode="w") as dst:
                            shutil.copyfileobj(src, dst)
                # Write new/overwritten entries
                with ArchiveProgress(total=len(resolved), desc=f"{archive_path.name}: adding", verbose=verbose) as pbar:
                    for abs_path, arcname in resolved:
                        dst_zf.write(abs_path, arcname=arcname)
                        pbar.update(1)
            tmp_path.replace(archive_path)
        else:
            # Simple append — no overwrite needed
            with ZipFile(archive_path, mode="a", compression=ZIP_DEFLATED) as zf:
                with ArchiveProgress(total=len(resolved), desc=f"{archive_path.name}: adding", verbose=verbose) as pbar:
                    for abs_path, arcname in resolved:
                        zf.write(abs_path, arcname=arcname)
                        pbar.update(1)


    def remove(self, archive_path: Path, files: List[str]) -> None:
        to_remove = set(files)
        with ZipFile(archive_path, mode="r") as zf:
            names = set(zf.namelist())
            missing = to_remove - names
            if missing:
                raise KeyError(f"Files not found in archive: {sorted(missing)}.")

            tmp_path = archive_path.with_suffix(".tmp.zip")
            with ZipFile(tmp_path, mode="w", compression=ZIP_DEFLATED) as tmp_zf:
                for item in zf.infolist():
                    if item.filename not in to_remove:
                        with zf.open(item) as src, tmp_zf.open(item, mode="w") as dst:
                            shutil.copyfileobj(src, dst)

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

        src_members: Dict[Path, List] = {}
        total = 0
        for src in archive_paths:
            with ZipFile(src, mode="r") as zf:
                src_members[src] = zf.infolist()
                total += len(src_members[src])

        written_names = set()
        with ArchiveProgress(total=total, desc=f"{output_path.name}: merging", verbose=verbose) as pbar:
            with ZipFile(output_path, mode="w", compression=ZIP_DEFLATED) as out_zf:
                for src in archive_paths:
                    with ZipFile(src, mode="r") as src_zf:
                        for item in src_members[src]:
                            if item.filename in written_names:
                                pbar.update(1)
                                continue
                            written_names.add(item.filename)
                            with src_zf.open(item) as src_file, out_zf.open(item, mode="w") as dst_file:
                                shutil.copyfileobj(src_file, dst_file)
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

        with ZipFile(archive_path, mode="r") as src_zf:
            members = src_zf.infolist()
            with ArchiveProgress(total=len(members), desc=f"{archive_path.name}: splitting", verbose=verbose) as pbar:
                current_members: List = []
                current_size = 0

                def _flush() -> None:
                    nonlocal part_num
                    part_path = output_dir / f"{stem}_part_{part_num:03d}{suffix}"
                    if part_path.exists() and not overwrite:
                        raise FileExistsError(f"Output already exists: {part_path}. Use overwrite=True.")
                    with ZipFile(part_path, mode="w", compression=ZIP_DEFLATED) as pzf:
                        for m in current_members:
                            with src_zf.open(m) as src_file, pzf.open(m, mode="w") as dst_file:
                                shutil.copyfileobj(src_file, dst_file)
                    parts.append(str(part_path))
                    part_num += 1

                for member in members:
                    if current_members and (current_size + member.compress_size) > max_bytes:
                        _flush()
                        current_members = []
                        current_size = 0
                    current_members.append(member)
                    current_size += member.compress_size
                    pbar.update(1)

                if current_members:
                    _flush()

        return parts
