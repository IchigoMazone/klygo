from zipfile import ZipFile
from pathlib import Path

from klygo.validators.archive import GetInfo, Test


def get_info(
    source: str | Path,
) -> dict:
    """Return a metadata summary of an archive.

    Reads all member headers in the ZIP central directory — no
    decompression is performed, so this is fast even for very large archives.

    Parameters
    ----------
    source : str or Path
        Path to the archive file.

    Returns
    -------
    dict
        A dictionary with the following keys:

        * ``"path"`` (*str*) — absolute path to the archive.
        * ``"format"`` (*str*) — archive format, e.g. ``"zip"``.
        * ``"file_count"`` (*int*) — number of files inside the archive.
        * ``"uncompressed_size"`` (*int*) — total size of all files before
          compression, in bytes.
        * ``"compressed_size"`` (*int*) — total size of all files after
          compression, in bytes.
        * ``"compress_ratio"`` (*float*) — space saved by compression,
          expressed as a percentage (0–100). A value of ``0.0`` means
          no compression was applied.
        * ``"archive_size"`` (*int*) — actual size of the ``.zip`` file
          on disk (includes ZIP metadata overhead).

    Raises
    ------
    TypeError
        If ``source`` is not a ``str`` or ``Path``.
    FileNotFoundError
        If ``source`` does not exist.
    ValueError
        If ``source`` is not a supported archive format.

    Examples
    --------
    >>> info = get_info("data.zip")
    >>> print(info["file_count"], info["compress_ratio"])
    723 0.64
    """

    params = GetInfo(source=source)

    with ZipFile(params.source, mode="r") as zf:
        members = zf.infolist()
        total_uncompressed = sum(m.file_size for m in members)
        total_compressed = sum(m.compress_size for m in members)

    ratio = (
        round((1 - total_compressed / total_uncompressed) * 100, 2)
        if total_uncompressed > 0
        else 0.0
    )

    return {
        "path": str(params.source),
        "format": params.source.suffix.lstrip("."),
        "file_count": len(members),
        "uncompressed_size": total_uncompressed,
        "compressed_size": total_compressed,
        "compress_ratio": ratio,
        "archive_size": params.source.stat().st_size,
    }


def test(
    source: str | Path,
) -> bool:
    """Verify the integrity of an archive using CRC checksum validation.

    Each file inside the archive is read and its CRC32 checksum is compared
    against the value stored in the ZIP header. This catches bit-rot,
    partial downloads, and corrupted archives.

    Parameters
    ----------
    source : str or Path
        Path to the archive file to test.

    Returns
    -------
    bool
        *True* if every file in the archive passes its CRC check.

    Raises
    ------
    TypeError
        If ``source`` is not a ``str`` or ``Path``.
    FileNotFoundError
        If ``source`` does not exist.
    ValueError
        If ``source`` is not a supported archive format, or if the
        archive is corrupted. The error message includes the name of
        the first file that failed the checksum.

    Examples
    --------
    >>> ok = test("data.zip")
    >>> print(ok)
    True
    """

    params = Test(source=source)

    with ZipFile(params.source, mode="r") as zf:
        bad_file = zf.testzip()

    if bad_file is not None:
        raise ValueError(
            f"archive is corrupted. First bad file: '{bad_file}'"
        )

    return True
