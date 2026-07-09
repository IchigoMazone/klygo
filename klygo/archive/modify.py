from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path

from tqdm import tqdm

from klygo.validators.archive import Add, Remove


def add(
    source: str | Path,
    files: "str | Path | list",
    verbose: bool = True,
) -> None:
    """Append one or more files or directories to an existing archive.

    If a directory is passed, all files inside it are added recursively,
    preserving the internal structure relative to the directory's parent.
    If an added file's arcname already exists inside the archive, a
    ``_dup`` suffix is appended to avoid silent overwrites.

    Parameters
    ----------
    source : str or Path
        Path to the existing archive file to append to.
    files : str, Path, or list of (str | Path)
        A single file/directory path or a list of paths to add.
        Each path must exist on the filesystem.
    verbose : bool, optional
        If *True* (default), display a coloured progress bar and print
        the count of added files on completion.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``source``, ``verbose``, or any item in ``files`` has the wrong type.
    FileNotFoundError
        If ``source`` or any file in ``files`` does not exist.
    ValueError
        If ``source`` is not a supported archive format.

    Examples
    --------
    Add a single file:

    >>> add("data.zip", "new_file.txt")

    Add multiple files at once:

    >>> add("data.zip", ["file_a.csv", "file_b.csv"])

    Add an entire directory:

    >>> add("data.zip", "extra_images/")
    """

    params = Add(source=source, files=files, verbose=verbose)

    # Expand directories to individual files
    all_files: list[tuple[Path, str]] = []  # (absolute_path, arcname)
    for fp in params.files:
        if fp.is_dir():
            for child in sorted(fp.rglob("*")):
                if child.is_file():
                    all_files.append((child, str(child.relative_to(fp.parent))))
        else:
            all_files.append((fp, fp.name))

    with ZipFile(params.source, mode="a", compression=ZIP_DEFLATED) as zf:
        existing = set(zf.namelist())
        bar = (
            tqdm(
                total=len(all_files) or 1,
                desc="Adding",
                unit="file",
                colour="yellow",
                bar_format="{l_bar}{bar:30}{r_bar}",
            )
            if verbose
            else None
        )
        for abs_path, arcname in all_files:
            # Avoid silently overwriting; suffix with _dup if name conflicts
            if arcname in existing:
                stem = Path(arcname).stem
                suffix = Path(arcname).suffix
                arcname = f"{stem}_dup{suffix}"
            zf.write(abs_path, arcname=arcname)
            if bar is not None:
                bar.update(1)
        if bar is not None:
            if not all_files:
                bar.update(1)
            bar.close()

    if verbose:
        print(f"Done. Added {len(all_files)} file(s) to '{params.source}'")


def remove(
    source: str | Path,
    files: "str | list[str]",
) -> None:
    """Delete one or more files from an archive.

    Because the ZIP format does not support in-place deletion, the
    archive is rewritten to a temporary sibling file and then atomically
    replaces the original. The temporary file is always cleaned up,
    even if an error occurs.

    Parameters
    ----------
    source : str or Path
        Path to the archive file to modify.
    files : str or list of str
        Arcname of the file to remove, or a list of arcnames, exactly
        as returned by :func:`list_files`.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``source`` or any item in ``files`` has the wrong type.
    FileNotFoundError
        If ``source`` does not exist.
    ValueError
        If ``source`` is not a supported archive format.
    KeyError
        If one or more names in ``files`` are not found inside the archive.

    Examples
    --------
    Remove a single file:

    >>> remove("data.zip", "images/frame_001.jpg")

    Remove multiple files:

    >>> remove("data.zip", ["images/frame_001.jpg", "data.yaml"])
    """

    params = Remove(source=source, files=files)
    to_remove = set(params.files)

    with ZipFile(params.source, mode="r") as zf:
        names = set(zf.namelist())
        missing = to_remove - names
        if missing:
            raise KeyError(
                f"Files not found in archive: {sorted(missing)}. "
                f"Use list_files() to see available files."
            )

        tmp_path = params.source.with_suffix(".tmp.zip")
        with ZipFile(tmp_path, mode="w", compression=ZIP_DEFLATED) as tmp_zf:
            for item in zf.infolist():
                if item.filename not in to_remove:
                    tmp_zf.writestr(item, zf.read(item.filename))

    tmp_path.replace(params.source)
