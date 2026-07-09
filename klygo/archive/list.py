from zipfile import ZipFile
from pathlib import Path
import fnmatch

from klygo.validators.archive import ListFiles, Search


def list_files(
    source: str | Path,
) -> list[str]:
    """Return a list of all file paths stored inside an archive.

    The returned strings are the *arcnames* — the paths as they are
    stored inside the ZIP (e.g. ``"images/frame_001.jpg"``). These
    values can be passed directly to :func:`extract_file`, :func:`remove`,
    and :func:`search`.

    Parameters
    ----------
    source : str or Path
        Path to the archive file (e.g. ``"data.zip"``).

    Returns
    -------
    list[str]
        Ordered list of file paths as stored inside the archive.

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
    >>> files = list_files("data.zip")
    >>> print(files[:3])
    ['images/frame_0.jpg', 'images/frame_1.jpg', 'data.yaml']
    """

    params = ListFiles(source=source)

    with ZipFile(params.source, mode="r") as zf:
        names = zf.namelist()

    return names


def search(
    source: str | Path,
    pattern: str,
) -> list[str]:
    """Find files inside an archive whose paths match a glob pattern.

    Matching is performed with :func:`fnmatch.fnmatch` against each
    arcname stored in the archive. The search is case-sensitive on
    case-sensitive filesystems.

    Parameters
    ----------
    source : str or Path
        Path to the archive file.
    pattern : str
        Glob pattern to match against file paths inside the archive.
        Supports the standard wildcards:

        * ``*``  — matches any sequence of characters (not ``/``).
        * ``?``  — matches any single character.
        * ``[seq]`` — matches any character in *seq*.

        Example: ``"images/*.jpg"`` matches all JPEG files under the
        ``images/`` folder.

    Returns
    -------
    list[str]
        Arcnames of the files that matched ``pattern``, in the order
        they appear inside the archive. Returns an empty list if
        nothing matches.

    Raises
    ------
    TypeError
        If ``source`` or ``pattern`` has the wrong type.
    FileNotFoundError
        If ``source`` does not exist.
    ValueError
        If ``source`` is not a supported archive format.

    Examples
    --------
    >>> results = search("data.zip", "images/frame_1*.jpg")
    >>> print(len(results))
    111

    >>> search("data.zip", "*.yaml")
    ['data.yaml']
    """

    params = Search(source=source, pattern=pattern)

    with ZipFile(params.source, mode="r") as zf:
        names = zf.namelist()

    return [name for name in names if fnmatch.fnmatch(name, params.pattern)]
