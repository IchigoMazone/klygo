from zipfile import ZipFile
from pathlib import Path

from tqdm import tqdm

from klygo.validators.archive import Extract, ExtractFile


def extract(
    source: str | Path,
    output: str | Path = ".",
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Extract all contents of an archive to a directory.

    The full directory structure stored inside the archive is
    recreated under ``output``. The output directory is created
    automatically if it does not exist.

    Parameters
    ----------
    source : str or Path
        Path to the archive file to extract (e.g. ``"data.zip"``).
    output : str or Path, optional
        Directory where the archive contents will be extracted.
        Defaults to the current working directory (``"."``).
    overwrite : bool, optional
        If *True*, overwrite files that already exist in ``output``.
        If *False* (default) and conflicting files exist, raises
        ``FileExistsError`` listing the first offending names.
    verbose : bool, optional
        If *True* (default), display a coloured progress bar and print
        the total number of extracted files on completion.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If any argument has the wrong type.
    FileNotFoundError
        If ``source`` does not exist.
    ValueError
        If ``source`` is not a supported archive format.
    FileExistsError
        If files in the archive already exist at the destination
        and ``overwrite`` is *False*.

    Examples
    --------
    Extract to a specific folder:

    >>> extract("data.zip", output="data/")

    Extract and overwrite any existing files:

    >>> extract("data.zip", output="data/", overwrite=True)

    Extract silently:

    >>> extract("data.zip", output="data/", verbose=False)
    """

    params = Extract(
        source=source,
        output=output,
        overwrite=overwrite,
        verbose=verbose,
    )

    source_path: Path = params.source
    output_path: Path = params.output

    output_path.mkdir(parents=True, exist_ok=True)

    with ZipFile(source_path, mode="r") as zf:
        members = zf.infolist()

        if not overwrite:
            existing = [
                m for m in members
                if (output_path / m.filename).exists()
            ]
            if existing:
                names = ", ".join(m.filename for m in existing[:5])
                suffix = f"… (+{len(existing) - 5} more)" if len(existing) > 5 else ""
                raise FileExistsError(
                    f"Files already exist in output directory: {names}{suffix}. "
                    "Use overwrite=True to replace them."
                )

        iterator = (
            tqdm(
                members,
                desc="Extracting",
                unit="file",
                colour="cyan",
                bar_format="{l_bar}{bar:30}{r_bar}",
            )
            if verbose
            else iter(members)
        )

        for member in iterator:
            zf.extract(member, path=output_path)

    if verbose:
        print(f"Done. Extracted {len(members)} file(s) to '{output_path}'")


def extract_file(
    source: str | Path,
    filename: str,
    output: str | Path = ".",
    overwrite: bool = False,
) -> None:
    """Extract a single file from an archive without unpacking everything.

    Use :func:`list_files` to discover the exact ``filename`` string
    as it is stored inside the archive, then pass that value here.

    Parameters
    ----------
    source : str or Path
        Path to the archive file.
    filename : str
        Path of the target file *inside* the archive, exactly as returned
        by :func:`list_files` (e.g. ``"images/frame_001.jpg"``).
    output : str or Path, optional
        Directory where the file will be written.
        Defaults to the current working directory (``"."``).
    overwrite : bool, optional
        If *True*, overwrite the file if it already exists at the destination.
        Default is *False*.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If any argument has the wrong type.
    FileNotFoundError
        If ``source`` does not exist.
    KeyError
        If ``filename`` is not found inside the archive.
    FileExistsError
        If the destination file already exists and ``overwrite`` is *False*.

    Examples
    --------
    >>> extract_file("data.zip", "images/frame_001.jpg", output="out/")
    """

    params = ExtractFile(
        source=source,
        filename=filename,
        output=output,
        overwrite=overwrite,
    )

    output_path: Path = params.output
    output_path.mkdir(parents=True, exist_ok=True)

    with ZipFile(params.source, mode="r") as zf:
        names = zf.namelist()
        if params.filename not in names:
            raise KeyError(
                f"'{params.filename}' not found in archive. "
                f"Use list_files() to see available files."
            )

        target = output_path / Path(params.filename).name
        if target.exists() and not params.overwrite:
            raise FileExistsError(
                f"file already exists: {target}. Use overwrite=True."
            )

        zf.extract(params.filename, path=output_path)
