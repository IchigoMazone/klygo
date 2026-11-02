"""Built-in ZIP backend with creation, extraction, and mutation support."""

import fnmatch
import re
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED, ZIP_BZIP2, ZIP_LZMA
from typing import Iterator, Any, Dict, List, Optional, Union, Literal

import klygo.files as file_utils
from klygo.archive.backend.base import ArchiveBackend, BackendCapabilities
from klygo.utils.formatting import human_size
from klygo.utils.progress import ProgressBar


def _is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """Check for Zip Slip vulnerability (Path Traversal)."""
    return file_utils.is_within(target_path, base_dir)


def _match_pattern(name: str, pattern: str, regex: bool = False, case_sensitive: bool = True) -> bool:
    if not case_sensitive:
        name = name.lower()
        pattern = pattern.lower()

    if regex:
        return bool(re.search(pattern, name))
    return fnmatch.fnmatch(name, pattern)


class ZipBackend(ArchiveBackend):
    """Read and write ZIP archives using Python's standard library.

    ZIP is the most feature-complete built-in backend. It supports creation,
    selective extraction, passwords for reading encrypted members, member
    mutation, same-format merging, and logical size-based splitting.

    Attributes
    ----------
    format_name : str
        Always ``"zip"``.
    capabilities : BackendCapabilities
        Declares every mutating operation plus ZIP-specific compression,
        extraction, and conflict options.
    COMPRESSION_METHODS : dict[str, int]
        Mapping from public method names to :mod:`zipfile` constants.

    Raises
    ------
    FileExistsError
        When creation, extraction, or splitting would replace existing data
        without explicit overwrite permission.
    ValueError
        For invalid compression settings, corrupt archives, or unsafe member
        paths that could escape the extraction directory.
    KeyError
        When an exact requested member does not exist.

    Notes
    -----
    Direct methods accept :class:`pathlib.Path` objects. Applications normally
    call :mod:`klygo.archive`, which performs path normalization and dispatch.

    Examples
    --------
    >>> from pathlib import Path
    >>> from klygo.archive.backend import ZipBackend
    >>> backend = ZipBackend()
    >>> backend.capabilities.add
    True
    >>> backend.compress(Path("dataset"), Path("dataset.zip"), verbose=False)
    """

    COMPRESSION_METHODS = {
        "deflated": ZIP_DEFLATED,
        "stored": ZIP_STORED,
        "bzip2": ZIP_BZIP2,
        "lzma": ZIP_LZMA,
    }
    format_name = "zip"
    capabilities = BackendCapabilities(
        compress=True,
        add=True,
        remove=True,
        merge=True,
        split=True,
        compress_options=frozenset(
            {"compresslevel", "method", "follow_symlinks", "include_root"}
        ),
        extract_options=frozenset({"password", "include", "exclude"}),
        add_options=frozenset({"on_conflict"}),
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
        """Create a ZIP archive from one file or directory.

        ``method`` accepts ``deflated``, ``stored``, ``bzip2``, or ``lzma``.
        Directory members include the source root unless ``include_root`` is
        false. Existing outputs require ``overwrite=True``.
        """
        if file_utils.exists(output_path) and not overwrite:
            raise FileExistsError(f"output_path already exists: {output_path}")

        method_name = (method or "deflated").lower()
        if method_name not in self.COMPRESSION_METHODS:
            raise ValueError(
                f"Unsupported ZIP compression method '{method}'. "
                f"Choose from {sorted(self.COMPRESSION_METHODS)}."
            )
        if not 0 <= compresslevel <= 9:
            raise ValueError("compresslevel must be between 0 and 9")
        comp_type = self.COMPRESSION_METHODS[method_name]
        file_utils.mkdir(file_utils.parent(output_path))

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

        with ProgressBar(total=total_count, desc=f"{output_path.name}: compressing", verbose=verbose) as pbar:
            with ZipFile(output_path, mode="w", compression=comp_type, compresslevel=compresslevel) as zf:
                gen = files_to_compress if files_to_compress is not None else _file_generator()
                for file_path in gen:
                    if source.is_dir():
                        base = source.parent if include_root else source
                        arcname = file_path.relative_to(base)
                    else:
                        arcname = file_path.name
                    zf.write(file_path, arcname=str(arcname))
                    pbar.update(1)

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
        """Extract selected ZIP members with overwrite and path-safety checks.

        ``include`` and ``exclude`` accept one glob or a list of globs. Exclude
        rules are applied after include rules. ``password`` is encoded as UTF-8
        for encrypted ZIP members.
        """
        pwd_bytes = password.encode("utf-8") if password else None
        output_dir = output_dir.resolve()
        file_utils.mkdir(output_dir)

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
                existing = [m for m in filtered_members if file_utils.exists(output_dir / m.filename)]
                if existing:
                    names = ", ".join(m.filename for m in existing[:5])
                    suffix = f"… (+{len(existing) - 5} more)" if len(existing) > 5 else ""
                    raise FileExistsError(f"Files already exist in output directory: {names}{suffix}. Use overwrite=True.")

            with ProgressBar(total=len(filtered_members), desc=f"{archive_path.name}: extracting", verbose=verbose) as pbar:
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
        """Stream one exact ZIP member to the output directory."""
        pwd_bytes = password.encode("utf-8") if password else None
        file_utils.mkdir(output_dir)

        with ZipFile(archive_path, mode="r") as zf:
            try:
                member = zf.getinfo(filename)
            except KeyError:
                raise KeyError(f"'{filename}' not found in archive. Use list_files() to see available file_utils.")

            target = output_dir / file_utils.name(filename)
            if file_utils.exists(target) and not overwrite:
                raise FileExistsError(f"File already exists: {target}. Use overwrite=True.")

            with zf.open(member, pwd=pwd_bytes) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

    def list_files(self, archive_path: Path) -> List[str]:
        """Return ZIP member names in central-directory order."""
        with ZipFile(archive_path, mode="r") as zf:
            return zf.namelist()

    def iter_files(self, archive_path: Path) -> Iterator[str]:
        """Yield ZIP member names while the archive handle is managed internally."""
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
        """Search ZIP member names using a glob or regular expression."""
        results = []
        for name in self.iter_files(archive_path):
            if _match_pattern(name, pattern, regex=regex, case_sensitive=case_sensitive):
                results.append(name)
        return results

    def get_info(self, archive_path: Path) -> Dict[str, Any]:
        """Return normalized ZIP sizes, counts, ratio, encryption, and extrema."""
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
        """Run the ZIP CRC check and optionally raise for the first bad member."""
        try:
            with ZipFile(archive_path, mode="r") as zf:
                bad_file = zf.testzip()
        except Exception as exc:
            if raise_exception:
                raise ValueError(f"ZIP archive is corrupted: {exc}") from exc
            return False

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
        """Add files with ``rename``, ``overwrite``, or ``skip`` conflict policy.

        Overwrite rebuilds the archive so an older duplicate member cannot
        remain visible. Rename generates deterministic ``_dupN`` names.
        """
        all_files: List[tuple[Path, str]] = []
        for fp in files:
            if fp.is_dir():
                for child in sorted(fp.rglob("*")):
                    if child.is_file():
                        all_files.append((child, child.relative_to(fp.parent).as_posix()))
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
                    member_path = PurePosixPath(arcname)
                    index = 1
                    while True:
                        candidate = member_path.with_name(
                            f"{member_path.stem}_dup{index}{member_path.suffix}"
                        ).as_posix()
                        if candidate not in existing:
                            arcname = candidate
                            break
                        index += 1
                elif on_conflict == "overwrite":
                    to_overwrite.add(arcname)
            resolved.append((abs_path, arcname))
            existing.add(arcname)

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
                with ProgressBar(total=len(resolved), desc=f"{archive_path.name}: adding", verbose=verbose) as pbar:
                    for abs_path, arcname in resolved:
                        dst_zf.write(abs_path, arcname=arcname)
                        pbar.update(1)
            tmp_path.replace(archive_path)
        else:
            # Simple append — no overwrite needed
            with ZipFile(archive_path, mode="a", compression=ZIP_DEFLATED) as zf:
                with ProgressBar(total=len(resolved), desc=f"{archive_path.name}: adding", verbose=verbose) as pbar:
                    for abs_path, arcname in resolved:
                        zf.write(abs_path, arcname=arcname)
                        pbar.update(1)


    def remove(self, archive_path: Path, files: List[str]) -> None:
        """Rebuild a ZIP archive without the named members."""
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
        """Merge ZIP archives, keeping the first occurrence of each member name."""
        if file_utils.exists(output_path) and not overwrite:
            raise FileExistsError(f"output_path already exists: {output_path}")

        src_members: Dict[Path, List] = {}
        total = 0
        for src in archive_paths:
            with ZipFile(src, mode="r") as zf:
                src_members[src] = zf.infolist()
                total += len(src_members[src])

        written_names = set()
        with ProgressBar(total=total, desc=f"{output_path.name}: merging", verbose=verbose) as pbar:
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
        """Group ZIP members into independently readable size-limited parts."""
        file_utils.mkdir(output_dir)
        max_bytes = int(size * 1024 * 1024)

        stem = archive_path.stem
        suffix = archive_path.suffix
        parts: List[str] = []
        part_num = 1

        with ZipFile(archive_path, mode="r") as src_zf:
            members = src_zf.infolist()
            with ProgressBar(total=len(members), desc=f"{archive_path.name}: splitting", verbose=verbose) as pbar:
                current_members: List = []
                current_size = 0

                def _flush() -> None:
                    nonlocal part_num
                    part_path = output_dir / f"{stem}_part_{part_num:03d}{suffix}"
                    if file_utils.exists(part_path) and not overwrite:
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
