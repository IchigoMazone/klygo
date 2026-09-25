"""Archive creation and extraction operations."""

from pathlib import Path
from typing import List, Optional, Union

import klygo.files as files
from klygo.archive._utils import resolve_backend

def compress(
    source: Union[str, Path],
    output_path: Union[str, Path],
    format: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = True,
    compresslevel: int = 6,
    method: Optional[str] = None,
    follow_symlinks: bool = False,
    include_root: bool = True,
) -> None:
    """Create an archive from a file or directory.

    The output backend is selected from ``format`` or the destination extension. ZIP, TAR, TAR.GZ, TAR.XZ, TAR.BZ2, GZ, and optionally 7Z can be written.

    Parameters
    ----------
    source : str or pathlib.Path
        File or directory to archive.
    output_path : str or pathlib.Path
        Destination archive path.
    format : str or None, default=None
        Explicit output format; inferred when omitted.
    overwrite : bool, default=False
        Allow existing output entries to be replaced.
    verbose : bool, default=True
        Display archive progress.
    compresslevel : int, default=6
        Compression level, usually from 1 through 9.
    method : str or None, default=None
        Backend-specific compression method.
    follow_symlinks : bool, default=False
        Follow symbolic links instead of archiving link entries.
    include_root : bool, default=True
        Include the source directory name as the top-level archive member.
        Set this to ``False`` to archive only the directory contents. This
        option has no effect when ``source`` is a single file.

    Returns
    -------
    None
        The archive is created as a side effect.

    Raises
    ------
    FileNotFoundError
        If ``source`` does not exist.
    FileExistsError
        If the destination exists and overwrite is disabled.
    ValueError
        If the format or backend parameters are invalid.

    See Also
    --------
    extract
    convert

    Examples
    --------
    >>> import klygo.archive as archive
    >>> archive.compress("dataset", "dataset.zip", verbose=False)
    >>> archive.compress("dataset", "contents-only.zip", include_root=False)
    """
    source_path = files.path(source)

    if not files.exists(source_path):
        raise FileNotFoundError(f"source does not exist: {source_path}")

    output_path, backend = resolve_backend(output_path, format_hint=format)
    backend.require_operation("compress")
    backend.validate_option("compress", "compresslevel", compresslevel, 6)
    backend.validate_option("compress", "method", method, None)
    backend.validate_option("compress", "follow_symlinks", follow_symlinks, False)
    backend.validate_option("compress", "include_root", include_root, True)
    backend.compress(
        source=source_path,
        output_path=output_path,
        compresslevel=compresslevel,
        method=method,
        follow_symlinks=follow_symlinks,
        include_root=include_root,
        overwrite=overwrite,
        verbose=verbose,
    )


def extract(
    archive_path: Union[str, Path],
    output_dir: Union[str, Path] = ".",
    overwrite: bool = False,
    verbose: bool = True,
    password: Optional[str] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
) -> None:
    """Extract selected or all members from an archive.

    Member filters are applied before extraction. Backends reject path traversal entries and enforce the overwrite policy.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Archive to inspect or modify.
    output_dir : str or pathlib.Path, default='.'
        Directory receiving extracted files or archive parts.
    overwrite : bool, default=False
        Allow existing output entries to be replaced.
    verbose : bool, default=True
        Display archive progress.
    password : str or None, default=None
        Password for encrypted formats that support it.
    include : str, list[str], or None, default=None
        Glob pattern or patterns to include.
    exclude : str, list[str], or None, default=None
        Glob pattern or patterns to exclude.

    Returns
    -------
    None
        Members are extracted as a side effect.

    Raises
    ------
    FileNotFoundError
        If the archive does not exist.
    FileExistsError
        If an extracted target exists and overwrite is disabled.
    ValueError
        If an unsafe member path is detected.

    See Also
    --------
    extract_file
    extract_matching

    Examples
    --------
    >>> import klygo.archive as archive
    >>> archive.extract("dataset.zip", "output", verbose=False)
    """
    path, backend = resolve_backend(archive_path)
    out = files.path(output_dir)
    backend.validate_option("extract", "password", password, None)
    backend.validate_option("extract", "include", include, None)
    backend.validate_option("extract", "exclude", exclude, None)
    backend.extract(
        archive_path=path,
        output_dir=out,
        password=password,
        include=include,
        exclude=exclude,
        overwrite=overwrite,
        verbose=verbose,
    )


def extract_file(
    archive_path: Union[str, Path],
    filename: str,
    output_dir: Union[str, Path] = ".",
    overwrite: bool = False,
    password: Optional[str] = None,
) -> None:
    """Extract one archive member using streaming I/O.

    Only the requested member is copied to the output directory. The destination uses the member basename.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Archive to inspect or modify.
    filename : str
        Exact member name stored in the archive.
    output_dir : str or pathlib.Path, default='.'
        Directory receiving extracted files or archive parts.
    overwrite : bool, default=False
        Allow existing output entries to be replaced.
    password : str or None, default=None
        Password for encrypted formats that support it.

    Returns
    -------
    None
        The selected member is extracted as a side effect.

    Raises
    ------
    KeyError
        If the member does not exist.
    FileExistsError
        If the destination exists and overwrite is disabled.

    See Also
    --------
    extract
    list_files

    Examples
    --------
    >>> import klygo.archive as archive
    >>> archive.extract_file("dataset.zip", "labels/a.txt", "output")
    """
    path, backend = resolve_backend(archive_path)
    out = files.path(output_dir)
    backend.validate_option("extract", "password", password, None)
    backend.extract_file(
        archive_path=path,
        filename=filename,
        output_dir=out,
        password=password,
        overwrite=overwrite,
    )


def extract_matching(
    archive_path: Union[str, Path],
    pattern: str,
    output_dir: Union[str, Path] = ".",
    overwrite: bool = False,
    password: Optional[str] = None,
) -> None:
    """Extract members matching a wildcard pattern.

    This is a convenience wrapper around ``extract`` with ``include=pattern`` and progress output disabled.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Archive to inspect or modify.
    pattern : str
        Glob pattern or regular expression.
    output_dir : str or pathlib.Path, default='.'
        Directory receiving extracted files or archive parts.
    overwrite : bool, default=False
        Allow existing output entries to be replaced.
    password : str or None, default=None
        Password for encrypted formats that support it.

    Returns
    -------
    None
        Matching members are extracted as a side effect.

    See Also
    --------
    extract
    search

    Examples
    --------
    >>> import klygo.archive as archive
    >>> archive.extract_matching("dataset.zip", "*.jpg", "images")
    """
    extract(
        archive_path=archive_path,
        output_dir=output_dir,
        include=pattern,
        overwrite=overwrite,
        password=password,
        verbose=False,
    )
