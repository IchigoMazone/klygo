from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path

from tqdm import tqdm

from klygo.validators.archive import Merge, Split
from klygo.archive._utils import _human_size


def merge(
    sources: list,
    output: str | Path,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Combine two or more archives into a single new archive.

    All files from every source archive are written sequentially into
    ``output``. Metadata (compression method, timestamps) from the
    original members is preserved. If duplicate arcnames exist across
    the input archives, both copies are included as-is.

    Parameters
    ----------
    sources : list of (str | Path)
        Ordered list of at least 2 archive paths to merge.
    output : str or Path
        Destination path for the merged archive (must end in ``.zip``).
    overwrite : bool, optional
        If *True*, overwrite ``output`` when it already exists.
        Default is *False*.
    verbose : bool, optional
        If *True* (default), display a coloured progress bar and print
        the final merged archive size on completion.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``sources`` is not a list with at least 2 items, or any
        argument has the wrong type.
    FileNotFoundError
        If any path in ``sources`` does not exist.
    ValueError
        If any source is not a supported format, or ``output`` does not
        end with ``.zip``.
    FileExistsError
        If ``output`` already exists and ``overwrite`` is *False*.

    Examples
    --------
    >>> merge(["part1.zip", "part2.zip"], "combined.zip")

    Merge three archives, overwriting the output if it exists:

    >>> merge(["a.zip", "b.zip", "c.zip"], "all.zip", overwrite=True)
    """

    params = Merge(sources=sources, output=output, overwrite=overwrite)

    # Pre-collect members per source for accurate total count
    src_members: dict[Path, list] = {}
    total = 0
    for src in params.sources:
        with ZipFile(src, mode="r") as zf:
            src_members[src] = zf.infolist()
            total += len(src_members[src])

    bar = (
        tqdm(
            total=total or 1,
            desc="Merging",
            unit="file",
            colour="magenta",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        if verbose
        else None
    )

    with ZipFile(params.output, mode="w", compression=ZIP_DEFLATED) as out_zf:
        for src in params.sources:
            with ZipFile(src, mode="r") as src_zf:
                for item in src_members[src]:
                    out_zf.writestr(item, src_zf.read(item.filename))
                    if bar is not None:
                        bar.update(1)

    if bar is not None:
        if total == 0:
            bar.update(1)
        bar.close()

    if verbose:
        size = params.output.stat().st_size
        print(
            f"Done. Merged {len(params.sources)} archives -> "
            f"{_human_size(size)}"
        )


def split(
    source: str | Path,
    size: int,
    output_dir: str | Path = ".",
    overwrite: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Split a large archive into multiple smaller part files.

    Files are distributed across parts greedily: a new part is started
    whenever adding the next file would cause the current part's
    *compressed* size to exceed ``size``. Each individual file always
    goes into exactly one part (files are never split across archives).

    Output files are named ``{stem}_part_NNN{suffix}`` where ``NNN``
    is a zero-padded three-digit counter (e.g. ``data_part_001.zip``).

    Parameters
    ----------
    source : str or Path
        Path to the archive file to split.
    size : int
        Maximum *compressed* size in bytes for each output part.
        Must be a positive integer.
    output_dir : str or Path, optional
        Directory where the part files will be written.
        Created automatically if it does not exist.
        Defaults to the current working directory (``"."``).
    overwrite : bool, optional
        If *True*, overwrite existing part files.
        Default is *False*.
    verbose : bool, optional
        If *True* (default), display a coloured progress bar and print
        the total number of parts created on completion.

    Returns
    -------
    list[str]
        Ordered list of paths to the created part files
        (e.g. ``["out/data_part_001.zip", "out/data_part_002.zip", …]``).

    Raises
    ------
    TypeError
        If any argument has the wrong type.
    FileNotFoundError
        If ``source`` does not exist.
    ValueError
        If ``source`` is not a supported archive format, or ``size`` is
        not a positive integer.
    FileExistsError
        If a part file already exists and ``overwrite`` is *False*.

    Examples
    --------
    Split into 5 MB parts:

    >>> parts = split("large_dataset.zip", size=5_000_000, output_dir="parts/")
    >>> print(parts)
    ['parts/large_dataset_part_001.zip', 'parts/large_dataset_part_002.zip']
    """

    params = Split(
        source=source,
        size=size,
        output_dir=output_dir,
        overwrite=overwrite,
    )

    params.output_dir.mkdir(parents=True, exist_ok=True)

    stem = params.source.stem
    suffix = params.source.suffix
    parts: list[str] = []
    part_num = 1

    with ZipFile(params.source, mode="r") as src_zf:
        members = src_zf.infolist()

        bar = (
            tqdm(
                total=len(members) or 1,
                desc="Splitting",
                unit="file",
                colour="blue",
                bar_format="{l_bar}{bar:30}{r_bar}",
            )
            if verbose
            else None
        )

        current_members: list = []
        current_size = 0

        def _flush() -> None:
            nonlocal part_num
            part_path = params.output_dir / f"{stem}_part_{part_num:03d}{suffix}"
            if part_path.exists() and not params.overwrite:
                raise FileExistsError(
                    f"output already exists: {part_path}. "
                    "Use overwrite=True."
                )
            with ZipFile(part_path, mode="w", compression=ZIP_DEFLATED) as pzf:
                for m in current_members:
                    pzf.writestr(m, src_zf.read(m.filename))
            parts.append(str(part_path))
            part_num += 1

        for member in members:
            if current_members and (current_size + member.compress_size) > params.size:
                _flush()
                current_members = []
                current_size = 0
            current_members.append(member)
            current_size += member.compress_size
            if bar is not None:
                bar.update(1)

        if current_members:
            _flush()

        if bar is not None:
            if not members:
                bar.update(1)
            bar.close()

    if verbose:
        print(f"Done. Split into {len(parts)} part(s).")

    return parts
