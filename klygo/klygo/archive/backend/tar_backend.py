"""Built-in TAR backend for raw and compressed TAR containers."""

import fnmatch
import re
import shutil
import tarfile
from pathlib import Path
from typing import Iterator, Any, Dict, List, Optional, Union, Literal

import klygo.files as file_utils
from klygo.archive.backend.base import ArchiveBackend, BackendCapabilities
from klygo.utils.formatting import human_size
from klygo.utils.progress import ProgressBar


def _is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """Check for Zip Slip / Path Traversal vulnerability."""
    return file_utils.is_within(target_path, base_dir)


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
    """Read and write TAR, TAR.GZ, TAR.XZ, and TAR.BZ2 archives.

    Parameters
    ----------
    format_name : {'tar', 'tar.gz', 'tar.xz', 'tar.bz2'}, default='tar'
        Container and compression variant handled by the instance.

    Attributes
    ----------
    format_name : str
        Canonical selected TAR variant.
    capabilities : BackendCapabilities
        Per-instance declaration. Compressed variants advertise
        ``compresslevel`` while raw TAR does not.

    Raises
    ------
    ValueError
        If ``format_name`` is not a supported TAR variant, extraction detects
        an unsafe path, or archive integrity validation fails in strict mode.
    FileExistsError
        When an operation would replace existing output without permission.
    KeyError
        When an exact member requested for extraction or removal is absent.

    Notes
    -----
    Adding to compressed TAR archives rebuilds the container atomically because
    compressed streams cannot be appended safely. Extraction uses the standard
    library's data filter and performs an additional containment check.

    Examples
    --------
    >>> from pathlib import Path
    >>> from klygo.archive.backend import TarBackend
    >>> backend = TarBackend("tar.gz")
    >>> backend.format_name
    'tar.gz'
    >>> "compresslevel" in backend.capabilities.compress_options
    True
    """

    def __init__(self, format_name: str = "tar"):
        self.format_name = format_name
        compress_options = {"follow_symlinks", "include_root"}
        if format_name != "tar":
            compress_options.add("compresslevel")
        self.capabilities = BackendCapabilities(
            compress=True,
            add=True,
            remove=True,
            merge=True,
            split=True,
            compress_options=frozenset(compress_options),
            extract_options=frozenset({"include", "exclude"}),
            add_options=frozenset({"on_conflict"}),
        )

    capabilities = BackendCapabilities(
        compress=True,
        add=True,
        remove=True,
        merge=True,
        split=True,
        compress_options=frozenset({"follow_symlinks", "include_root"}),
        extract_options=frozenset({"include", "exclude"}),
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
        """Create the configured TAR variant from a file or directory.

        Compressed variants accept levels 1 through 9. Raw TAR rejects a
        changed compression level. ``include_root`` controls the leading source
        directory, and symlink traversal follows ``follow_symlinks``.
        """
        self.validate_option("compress", "compresslevel", compresslevel, 6)
        self.validate_option("compress", "method", method, None)
        if "compresslevel" in self.capabilities.compress_options and not 1 <= compresslevel <= 9:
            raise ValueError("compresslevel must be between 1 and 9 for compressed TAR archives.")
        if file_utils.exists(output_path) and not overwrite:
            raise FileExistsError(f"output_path already exists: {output_path}")

        mode = _get_tar_mode(self.format_name, write=True)
        file_utils.mkdir(file_utils.parent(output_path))

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
            open_options: Dict[str, Any] = {}
            if self.format_name in {"tar.gz", "tgz", "tar.bz2", "tbz2"}:
                open_options["compresslevel"] = compresslevel
            elif self.format_name in {"tar.xz", "txz"}:
                open_options["preset"] = compresslevel
            with tarfile.open(output_path, mode=mode, **open_options) as tar:
                gen = files_to_compress if files_to_compress is not None else _file_generator()
                for file_path in gen:
                    if source.is_dir():
                        base = source.parent if include_root else source
                        arcname = file_path.relative_to(base)
                    else:
                        arcname = file_path.name
                    tar.add(file_path, arcname=str(arcname), recursive=False)
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
        """Extract filtered TAR members using safe data-filter semantics.

        Include and exclude values are glob patterns. TAR has no password
        support, so a changed password is rejected through capability checks.
        """
        self.validate_option("extract", "password", password, None)
        output_dir = output_dir.resolve()
        file_utils.mkdir(output_dir)

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
                existing = [m for m in filtered_members if file_utils.exists(output_dir / m.name)]
                if existing:
                    names = ", ".join(m.name for m in existing[:5])
                    suffix = f"… (+{len(existing) - 5} more)" if len(existing) > 5 else ""
                    raise FileExistsError(f"Files already exist in output directory: {names}{suffix}. Use overwrite=True.")

            with ProgressBar(total=len(filtered_members), desc=f"{archive_path.name}: extracting", verbose=verbose) as pbar:
                for member in filtered_members:
                    target_path = output_dir / member.name
                    if not _is_safe_path(output_dir, target_path):
                        raise ValueError(f"Unsafe path detected in archive (Zip Slip): '{member.name}'")
                    tar.extract(member, path=output_dir, filter="data")
                    pbar.update(1)

    def extract_file(
        self,
        archive_path: Path,
        filename: str,
        output_dir: Path,
        password: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """Stream one regular TAR member to the output directory."""
        self.validate_option("extract", "password", password, None)
        file_utils.mkdir(output_dir)
        mode = _get_tar_mode(self.format_name, write=False)

        with tarfile.open(archive_path, mode=mode) as tar:
            try:
                member = tar.getmember(filename)
            except KeyError:
                raise KeyError(f"'{filename}' not found in archive. Use list_files() to see available file_utils.")

            target = output_dir / file_utils.name(filename)
            if file_utils.exists(target) and not overwrite:
                raise FileExistsError(f"File already exists: {target}. Use overwrite=True.")

            src = tar.extractfile(member)
            if src is None:
                raise ValueError(f"Cannot extract member '{filename}' (may be directory).")
            with src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

    def list_files(self, archive_path: Path) -> List[str]:
        """Return all TAR member names in stored order."""
        mode = _get_tar_mode(self.format_name, write=False)
        with tarfile.open(archive_path, mode=mode) as tar:
            return tar.getnames()

    def iter_files(self, archive_path: Path) -> Iterator[str]:
        """Yield TAR member names from the sequential archive stream."""
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
        """Search TAR member names with glob or regular-expression matching."""
        results = []
        for name in self.iter_files(archive_path):
            n = name if case_sensitive else name.lower()
            p = pattern if case_sensitive else pattern.lower()
            match = bool(re.search(p, n)) if regex else fnmatch.fnmatch(n, p)
            if match:
                results.append(name)
        return results

    def get_info(self, archive_path: Path) -> Dict[str, Any]:
        """Return normalized TAR member counts, sizes, ratio, and extrema."""
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
        """Read the complete TAR member stream and report structural validity."""
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
        """Add roots by rebuilding the archive with explicit conflict rules.

        Python's ``tarfile`` cannot append to compressed TAR streams. Rebuilding
        works uniformly for TAR, TAR.GZ, TAR.XZ, and TAR.BZ2 and also makes
        ``overwrite`` a real replacement instead of creating duplicate members.
        """
        self.validate_option("add", "on_conflict", on_conflict, "rename")
        for item in files:
            if not file_utils.exists(item):
                raise FileNotFoundError(f"Input path does not exist: {item}")

        mode_read = _get_tar_mode(self.format_name, write=False)
        mode_write = _get_tar_mode(self.format_name, write=True)
        tmp_path = file_utils.unique_path(
            archive_path.with_name(f"{archive_path.name}.tmp")
        )

        try:
            with tarfile.open(archive_path, mode=mode_read) as source:
                existing = set(source.getnames())
                additions: List[tuple[Path, str]] = []
                replaced_roots: set[str] = set()
                reserved = set(existing)

                for item in files:
                    root_name = item.name
                    collision = root_name in reserved or any(
                        name.startswith(f"{root_name}/") for name in reserved
                    )
                    if collision and on_conflict == "skip":
                        continue
                    if collision and on_conflict == "overwrite":
                        replaced_roots.add(root_name)
                    elif collision:
                        stem = file_utils.stem(root_name)
                        suffix = file_utils.extension(root_name)
                        index = 1
                        while True:
                            candidate = f"{stem}_dup{index}{suffix}"
                            if candidate not in reserved and not any(
                                name.startswith(f"{candidate}/") for name in reserved
                            ):
                                root_name = candidate
                                break
                            index += 1
                    additions.append((item, root_name))
                    reserved.add(root_name)

                if not additions:
                    return

                with tarfile.open(tmp_path, mode=mode_write) as destination:
                    for member in source.getmembers():
                        if any(
                            member.name == root
                            or member.name.startswith(f"{root}/")
                            for root in replaced_roots
                        ):
                            continue
                        stream = source.extractfile(member) if member.isfile() else None
                        destination.addfile(member, stream)

                    with ProgressBar(
                        total=len(additions),
                        desc=f"{archive_path.name}: adding",
                        verbose=verbose,
                    ) as pbar:
                        for item, arcname in additions:
                            destination.add(item, arcname=arcname)
                            pbar.update(1)
            tmp_path.replace(archive_path)
        finally:
            file_utils.remove(tmp_path, missing_ok=True)

    def remove(self, archive_path: Path, files: List[str]) -> None:
        """Rebuild the configured TAR variant without exact named members."""
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
        """Merge TAR archives while keeping the first occurrence of each name."""
        if file_utils.exists(output_path) and not overwrite:
            raise FileExistsError(f"output_path already exists: {output_path}")

        written_names = set()
        mode_write = _get_tar_mode(self.format_name, write=True)
        with tarfile.open(output_path, mode=mode_write) as dst_tar:
            with ProgressBar(total=len(archive_paths), desc=f"{output_path.name}: merging", verbose=verbose) as pbar:
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
        """Group TAR members into independently readable size-limited parts."""
        file_utils.mkdir(output_dir)
        max_bytes = int(size * 1024 * 1024)

        stem = archive_path.stem
        suffix = archive_path.suffix
        parts: List[str] = []
        part_num = 1

        mode_read = _get_tar_mode(self.format_name, write=False)
        mode_write = _get_tar_mode(self.format_name, write=True)

        with tarfile.open(archive_path, mode=mode_read) as src_tar:
            members = src_tar.getmembers()
            with ProgressBar(total=len(members), desc=f"{archive_path.name}: splitting", verbose=verbose) as pbar:
                current_members: List = []
                current_size = 0

                def _flush() -> None:
                    nonlocal part_num
                    part_path = output_dir / f"{stem}_part_{part_num:03d}{suffix}"
                    if file_utils.exists(part_path) and not overwrite:
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
