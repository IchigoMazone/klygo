from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path

from tqdm import tqdm

from klygo.validators.archive import Compress
from klygo.archive._utils import _human_size


def compress(
    source: str | Path,
    output: str | Path,
    format: str = "zip",
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Compress a file or directory into a ZIP archive.

    When ``source`` is a directory, all files are added recursively
    while preserving the internal directory structure. Parent-level
    directory name is kept as the root inside the archive.

    Parameters
    ----------
    source : str or Path
        Path to the file or directory to compress.
    output : str or Path
        Destination path for the archive (e.g. ``"data.zip"``).
        Parent directories are created automatically if they do not exist.
    format : str, optional
        Archive format. Currently only ``"zip"`` is supported.
        Default is ``"zip"``.
    overwrite : bool, optional
        If *True*, silently overwrite ``output`` when it already exists.
        If *False* (default) and ``output`` exists, raises ``FileExistsError``.
    verbose : bool, optional
        If *True* (default), display a coloured progress bar and print
        the final archive size on completion.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If any argument has the wrong type.
    FileNotFoundError
        If ``source`` does not exist.
    FileExistsError
        If ``output`` already exists and ``overwrite`` is *False*.
    ValueError
        If ``format`` is not supported, or ``output`` has the wrong extension.

    Examples
    --------
    Compress a single file:

    >>> compress("report.csv", "report.zip")

    Compress an entire directory, overwriting any existing archive:

    >>> compress("my_dataset/", "dataset.zip", overwrite=True)

    Compress silently (no progress bar):

    >>> compress("images/", "images.zip", verbose=False)
    """

    params = Compress(
        source=source,
        output=output,
        format=format,
        overwrite=overwrite,
        verbose=verbose,
    )

    source_path: Path = params.source
    output_path: Path = params.output

    # Collect all files to compress
    if source_path.is_dir():
        all_files: list[Path] = sorted(
            f for f in source_path.rglob("*") if f.is_file()
        )
    else:
        all_files = [source_path]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output_path, mode="w", compression=ZIP_DEFLATED) as zf:
        bar = (
            tqdm(
                total=len(all_files) or 1,
                desc="Compressing",
                unit="file",
                colour="green",
                bar_format="{l_bar}{bar:30}{r_bar}",
            )
            if verbose
            else None
        )
        for file in all_files:
            arcname = (
                file.relative_to(source_path.parent)
                if source_path.is_dir()
                else file.name
            )
            zf.write(file, arcname=arcname)
            if bar is not None:
                bar.update(1)
        if bar is not None:
            if not all_files:
                bar.update(1)
            bar.close()

    if verbose:
        total_size = output_path.stat().st_size
        print(f"Done. Archive size: {_human_size(total_size)}")
