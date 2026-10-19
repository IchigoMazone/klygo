"""Remote and local file download helpers."""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from klygo.utils.progress import ProgressBar
from klygo.validators import validate_type

from ._typing import PathInput
from .filesystem import copy


def download(
    source: PathInput,
    output_dir: PathInput = ".",
    overwrite: bool = False,
    verbose: bool = True,
    *,
    timeout: float = 30.0,
    retries: int = 0,
    expected_sha256: str | None = None,
) -> Path:
    """Download a URL or copy a local file while preserving its name.

    HTTP, HTTPS, FTP, local files, and Google Colab are supported. Directories are intentionally rejected.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Source URL or filesystem path.
    output_dir : str or os.PathLike[str], default='.'
        Directory in which the output file is created.
    overwrite : bool, default=False
        Allow an existing destination to be replaced.
    verbose : bool, default=True
        Display a progress indicator.
    timeout : float, default=30.0
        Network timeout in seconds.
    retries : int, default=0
        Additional attempts after a network or incomplete-transfer error.
    expected_sha256 : str or None, default=None
        Expected SHA-256 digest, checked before the file is published.

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
        If a local source is a directory, an option is invalid, or a checksum
        does not match.
    OSError
        If a transfer is incomplete or cannot be published.

    See Also
    --------
    copy
    move

    Examples
    --------
    >>> from klygo import files
    >>> model = files.download(url, output_dir="weights")
    """
    validate_type(source, (str, os.PathLike), "source")
    validate_type(output_dir, (str, os.PathLike), "output_dir")
    validate_type(overwrite, bool, "overwrite")
    validate_type(verbose, bool, "verbose")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive number of seconds")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a non-negative integer")
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None
    ):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")

    source_text = os.fspath(source)
    validate_type(source_text, str, "source")
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    if source_text.startswith(("http://", "https://", "ftp://")):
        parsed = urlparse(source_text)
        destination = destination_dir / (os.path.basename(parsed.path) or "downloaded_file")
        if (destination.exists() or destination.is_symlink()) and not overwrite:
            raise FileExistsError(
                f"File already exists: {destination}. Use overwrite=True to replace it."
            )
        request = urllib.request.Request(source_text, headers={"User-Agent": "klygo/2.0"})
        for attempt in range(retries + 1):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination_dir
            )
            temporary = Path(temporary_name)
            os.close(descriptor)
            try:
                try:
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        declared = response.headers.get("content-length")
                        total_size = int(declared) if declared is not None else None
                        digest = hashlib.sha256()
                        downloaded = 0
                        with temporary.open("wb") as stream, ProgressBar(
                            total=total_size,
                            desc=f"Downloading {destination.name}",
                            unit="B",
                            unit_scale=True,
                            verbose=verbose,
                            colour="cyan",
                        ) as progress:
                            while chunk := response.read(8192):
                                stream.write(chunk)
                                digest.update(chunk)
                                downloaded += len(chunk)
                                progress.update(len(chunk))
                    if total_size is not None and downloaded != total_size:
                        raise OSError(f"Incomplete download: expected {total_size} bytes, received {downloaded}")
                except (OSError, urllib.error.URLError):
                    if attempt >= retries:
                        raise
                    time.sleep(min(2 ** attempt, 5))
                    continue
                if expected_sha256 is not None and digest.hexdigest() != expected_sha256.lower():
                    raise ValueError("Downloaded file does not match expected_sha256")
                if overwrite:
                    os.replace(temporary, destination)
                else:
                    os.link(temporary, destination)
                return destination
            finally:
                temporary.unlink(missing_ok=True)

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file does not exist: {source_path}")
    if source_path.is_dir():
        raise ValueError(f"download() only supports single files, not directories: {source_path}")
    if expected_sha256 is not None:
        from .metadata import hash as file_hash

        if file_hash(source_path, "sha256") != expected_sha256.lower():
            raise ValueError("Source file does not match expected_sha256")

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






