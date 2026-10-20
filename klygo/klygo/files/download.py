"""Remote and local file download helpers."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

from klygo.utils.progress import ProgressBar
from klygo.validators import validate_type

from .filesystem import copy

PathInput = Union[str, Path]


def download(
    source: PathInput,
    output_dir: PathInput = ".",
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Download a URL or copy a local file while preserving its name.

    HTTP, HTTPS, FTP, local files, and Google Colab are supported. Directories are intentionally rejected.

    Parameters
    ----------
    source : str or pathlib.Path
        Source URL or filesystem path.
    output_dir : str or pathlib.Path, default='.'
        Directory in which the output file is created.
    overwrite : bool, default=False
        Allow an existing destination to be replaced.
    verbose : bool, default=True
        Display a progress indicator.

    Returns
    -------
    pathlib.Path
        Path to the downloaded or copied file.

    Raises
    ------
    FileNotFoundError
        If a local source does not exist.
    FileExistsError
        If the destination exists and overwrite is disabled.
    ValueError
        If a local source is a directory.

    See Also
    --------
    copy
    move

    Examples
    --------
    >>> from klygo import files
    >>> model = files.download(url, output_dir="weights")
    """
    validate_type(source, (str, Path), "source")
    validate_type(output_dir, (str, Path), "output_dir")
    validate_type(overwrite, bool, "overwrite")
    validate_type(verbose, bool, "verbose")

    source_text = str(source)
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    if source_text.startswith(("http://", "https://", "ftp://")):
        parsed = urlparse(source_text)
        destination = destination_dir / (os.path.basename(parsed.path) or "downloaded_file")
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {destination}. Use overwrite=True to replace it."
            )
        request = urllib.request.Request(source_text, headers={"User-Agent": "klygo/2.0"})
        with urllib.request.urlopen(request) as response:
            total_size = int(response.headers.get("content-length", 0))
            with destination.open("wb") as stream, ProgressBar(
                total=total_size if total_size > 0 else None,
                desc=f"Downloading {destination.name}",
                unit="B",
                unit_scale=True,
                verbose=verbose,
                colour="cyan",
            ) as progress:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    stream.write(chunk)
                    progress.update(len(chunk))
        return destination

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file does not exist: {source_path}")
    if source_path.is_dir():
        raise ValueError(f"download() only supports single files, not directories: {source_path}")

    try:
        from google.colab import files as colab_files

        colab_files.download(str(source_path))
        return source_path
    except ImportError:
        pass

    destination = destination_dir / source_path.name
    if destination.resolve() != source_path.resolve():
        copy(source_path, destination, overwrite=overwrite)
    return destination


__all__ = ["download"]






