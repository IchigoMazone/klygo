"""In-place archive mutation and archive transformation operations."""

import tempfile
from pathlib import Path
from typing import List, Literal, Union

import klygo.files as fs
from klygo.archive._utils import ensure_writable, resolve_backend
from klygo.archive.backend import detect_format


def add(
    archive_path: Union[str, Path],
    files: Union[str, Path, List[Union[str, Path]]],
    verbose: bool = True,
    on_conflict: Literal["rename", "overwrite", "skip"] = "rename",
) -> None:
    """Add files or directories to an existing archive.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Archive to inspect or modify.
    files : path-like or sequence
        Filesystem inputs for ``add`` or member names for ``remove``.
    verbose : bool, default=True
        Display archive progress.
    on_conflict : {'rename', 'overwrite', 'skip'}, default='rename'
        Strategy used when an added member name already exists.

    Returns
    -------
    None
        The archive is modified as a side effect.

    Raises
    ------
    FileNotFoundError
        If an input path does not exist.

    See Also
    --------
    remove

    Examples
    --------
    >>> import klygo.archive as archive
    >>> archive.add("dataset.zip", "new-label.txt", verbose=False)
    """
    path, backend = resolve_backend(archive_path)

    if on_conflict not in {"rename", "overwrite", "skip"}:
        raise ValueError("on_conflict must be 'rename', 'overwrite', or 'skip'.")

    if isinstance(files, (str, Path)):
        files_list = [fs.path(files)]
    else:
        files_list = [fs.path(item) for item in files]

    backend.require_operation("add")
    backend.validate_option("add", "on_conflict", on_conflict, "rename")
    backend.add(
        archive_path=path,
        files=files_list,
        on_conflict=on_conflict,
        verbose=verbose,
    )


def remove(
    archive_path: Union[str, Path],
    files: Union[str, List[str]],
) -> None:
    """Remove named members from an existing archive.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Archive to inspect or modify.
    files : path-like or sequence
        Filesystem inputs for ``add`` or member names for ``remove``.

    Returns
    -------
    None
        The archive is modified as a side effect.

    Raises
    ------
    KeyError
        If a requested member is absent.

    See Also
    --------
    add

    Examples
    --------
    >>> import klygo.archive as archive
    >>> archive.remove("dataset.zip", "obsolete.txt")
    """
    path, backend = resolve_backend(archive_path)

    if isinstance(files, str):
        files_list = [files]
    else:
        files_list = files

    backend.require_operation("remove")
    backend.remove(archive_path=path, files=files_list)


def _rebuild_archive(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    *,
    compresslevel: int,
    overwrite: bool,
    verbose: bool,
) -> None:
    """Extract and recompress an archive while preserving member paths."""
    src, src_backend = resolve_backend(source_path)
    dst, dst_backend = resolve_backend(target_path)
    dst_backend.require_operation("compress")
    ensure_writable(dst, overwrite=overwrite, parameter="target_path")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = fs.path(tmpdir)
        src_backend.extract(src, tmp_path, verbose=verbose)
        dst_backend.compress(
            tmp_path,
            dst,
            compresslevel=compresslevel,
            include_root=False,
            overwrite=overwrite,
            verbose=verbose,
        )


def merge(
    archive_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Merge multiple archives into one destination archive.

    Archives with matching formats use the backend fast path. Cross-format inputs are extracted into a temporary directory and recompressed.

    Parameters
    ----------
    archive_paths : list[str or pathlib.Path]
        At least two input archives.
    output_path : str or pathlib.Path
        Destination archive path.
    overwrite : bool, default=False
        Allow existing output entries to be replaced.
    verbose : bool, default=True
        Display archive progress.

    Returns
    -------
    None
        The destination archive is created as a side effect.

    Raises
    ------
    FileExistsError
        If the output exists and overwrite is disabled.

    See Also
    --------
    convert

    Examples
    --------
    >>> import klygo.archive as archive
    >>> archive.merge(["a.zip", "b.zip"], "merged.zip")
    """
    out_path = fs.path(output_path)
    sources = [fs.path(item) for item in archive_paths]

    out_fmt = detect_format(out_path)
    src_fmts = [detect_format(p) for p in sources]
    all_same_format = all(f == out_fmt for f in src_fmts)
    _, dst_backend = resolve_backend(out_path, format_hint=out_fmt)

    if all_same_format:
        # Fast path — same format, merge directly
        dst_backend.require_operation("merge")
        dst_backend.merge(
            archive_paths=sources,
            output_path=out_path,
            overwrite=overwrite,
            verbose=verbose,
        )
    else:
        # Cross-format path — extract all to temp dir, then compress
        dst_backend.require_operation("compress")
        ensure_writable(out_path, overwrite=overwrite, parameter="output_path")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = fs.path(tmpdir)
            for src, src_fmt in zip(sources, src_fmts):
                _, src_backend = resolve_backend(src, format_hint=src_fmt)
                src_backend.extract(src, tmp_path, verbose=verbose)
            dst_backend.compress(
                tmp_path,
                out_path,
                include_root=False,
                overwrite=overwrite,
                verbose=verbose,
            )



def split_by_size(
    archive_path: Union[str, Path],
    size: Union[int, float],
    output_dir: Union[str, Path] = ".",
    overwrite: bool = False,
    verbose: bool = True,
) -> List[str]:
    """Split an archive into size-limited archive parts.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Archive to inspect or modify.
    size : int or float
        Maximum size of each part in megabytes.
    output_dir : str or pathlib.Path, default='.'
        Directory receiving extracted files or archive parts.
    overwrite : bool, default=False
        Allow existing output entries to be replaced.
    verbose : bool, default=True
        Display archive progress.

    Returns
    -------
    list[str]
        Paths to the generated archive parts.

    Raises
    ------
    ValueError
        If ``size`` is not positive or the backend cannot split.

    See Also
    --------
    merge

    Examples
    --------
    >>> import klygo.archive as archive
    >>> parts = archive.split_by_size("large.zip", 100, "parts")
    """
    path, backend = resolve_backend(archive_path)
    out_dir = fs.path(output_dir)
    backend.require_operation("split")
    return backend.split_by_size(
        archive_path=path,
        size=float(size),
        output_dir=out_dir,
        overwrite=overwrite,
        verbose=verbose,
    )


def convert(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Convert an archive to another format.

    Members are extracted into a temporary directory and compressed using the destination backend.

    Parameters
    ----------
    source_path : str or pathlib.Path
        Source archive path.
    target_path : str or pathlib.Path
        Destination archive path.
    overwrite : bool, default=False
        Allow existing output entries to be replaced.
    verbose : bool, default=True
        Display archive progress.

    Returns
    -------
    None
        The target archive is created as a side effect.

    Raises
    ------
    FileExistsError
        If the target exists and overwrite is disabled.

    See Also
    --------
    recompress

    Examples
    --------
    >>> import klygo.archive as archive
    >>> archive.convert("dataset.zip", "dataset.tar.gz")
    """
    _rebuild_archive(
        source_path,
        target_path,
        compresslevel=6,
        overwrite=overwrite,
        verbose=verbose,
    )


def recompress(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    compresslevel: int = 6,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Rebuild an archive with a different compression level or format.

    Members are extracted into a temporary directory and compressed with ``compresslevel``.

    Parameters
    ----------
    source_path : str or pathlib.Path
        Source archive path.
    target_path : str or pathlib.Path
        Destination archive path.
    compresslevel : int, default=6
        Compression level, usually from 1 through 9.
    overwrite : bool, default=False
        Allow existing output entries to be replaced.
    verbose : bool, default=True
        Display archive progress.

    Returns
    -------
    None
        The target archive is created as a side effect.

    Raises
    ------
    FileExistsError
        If the target exists and overwrite is disabled.

    See Also
    --------
    convert

    Examples
    --------
    >>> import klygo.archive as archive
    >>> archive.recompress("input.zip", "compact.zip", compresslevel=9)
    """
    _rebuild_archive(
        source_path,
        target_path,
        compresslevel=compresslevel,
        overwrite=overwrite,
        verbose=verbose,
    )


def copy(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    overwrite: bool = False,
) -> None:
    """Copy an archive without extracting or changing it.

    The implementation delegates filesystem behavior and overwrite handling to ``klygo.files.copy``.

    Parameters
    ----------
    source_path : str or pathlib.Path
        Source archive path.
    target_path : str or pathlib.Path
        Destination archive path.
    overwrite : bool, default=False
        Allow existing output entries to be replaced.

    Returns
    -------
    None
        The destination file is created as a side effect.

    Raises
    ------
    FileNotFoundError
        If the source does not exist.
    FileExistsError
        If the target exists and overwrite is disabled.

    See Also
    --------
    klygo.files.copy

    Examples
    --------
    >>> import klygo.archive as archive
    >>> archive.copy("dataset.zip", "backup/dataset.zip")
    """
    fs.copy(source_path, target_path, overwrite=overwrite)
