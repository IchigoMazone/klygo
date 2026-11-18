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
from urllib.parse import parse_qs, unquote, urlencode, urlparse

from klygo.utils.progress import ProgressBar
from klygo.validators import validate_type

from ._typing import PathInput
from .filesystem import copy, exists, is_dir, is_symlink, mkdir, remove
from .metadata import hash as file_hash
from .paths import name, path as to_path, resolve


_GENERIC_URL_NAMES = {"", "view", "edit", "preview", "open", "uc", "download"}
_GDRIVE_HOSTS = {"drive.google.com", "drive.usercontent.google.com", "docs.google.com"}


def _extract_gdrive_file_id(url: str) -> str | None:
    """Extract a Google Drive file ID from common share, open, or uc URLs."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host not in _GDRIVE_HOSTS:
        return None
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", parsed.path)
    if match:
        return match.group(1)
    query_ids = parse_qs(parsed.query).get("id")
    if query_ids:
        return query_ids[0]
    return None


def _normalize_download_url(url: str) -> tuple[str, bool]:
    """Rewrite known viewer URLs (such as Google Drive /view) to direct download URLs."""
    file_id = _extract_gdrive_file_id(url)
    if file_id is not None:
        return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t", True
    return url, False


def _extract_gdrive_confirm_url(html_text: str, fallback_file_id: str | None) -> str | None:
    """Parse Google Drive's virus-scan warning HTML form into a direct download URL."""
    action_match = re.search(r'<form[^>]+id=["\']download-form["\'][^>]+action=["\']([^"\']+)["\']', html_text, re.I)
    if action_match:
        action_url = action_match.group(1).replace("&amp;", "&")
        inputs = dict(re.findall(r'<input[^>]+type=["\']hidden["\'][^>]+name=["\']([^"\']+)["\'][^>]+value=["\']([^"\']*)["\']', html_text, re.I))
        if inputs:
            sep = "&" if "?" in action_url else "?"
            return f"{action_url}{sep}{urlencode(inputs)}"
        return action_url
    confirm_match = re.search(r'confirm=([0-9A-Za-z_-]+)', html_text)
    if confirm_match and fallback_file_id:
        return f"https://drive.usercontent.google.com/download?id={fallback_file_id}&export=download&confirm={confirm_match.group(1)}"
    return None


def _filename_from_headers(headers: object) -> str | None:
    """Extract a sanitized filename from Content-Disposition headers when present."""
    disposition = headers.get("content-disposition") if hasattr(headers, "get") else None
    if not disposition or not isinstance(disposition, str):
        return None
    utf8_match = re.search(r"filename\*\s*=\s*(?:UTF-8|utf-8)''([^;\r\n]+)", disposition)
    if utf8_match:
        candidate = unquote(utf8_match.group(1)).strip().strip('"\'')
    else:
        plain_match = re.search(r'filename\s*=\s*"([^"]+)"|filename\s*=\s*([^;\r\n]+)', disposition)
        if not plain_match:
            return None
        candidate = (plain_match.group(1) or plain_match.group(2) or "").strip().strip('"\'')
    sanitized = Path(candidate).name
    return sanitized if sanitized and sanitized not in {".", ".."} else None


def _resolve_remote_filename(initial_url: str, response: object, fallback: str = "downloaded_file") -> str:
    """Determine output filename from response headers, redirected URL, or initial URL."""
    headers = getattr(response, "headers", None)
    header_name = _filename_from_headers(headers)
    if header_name:
        return header_name
    final_url = getattr(response, "geturl", lambda: None)()
    for candidate_url in (final_url, initial_url):
        if not candidate_url:
            continue
        parsed_path = urlparse(candidate_url).path
        candidate = name(parsed_path) if parsed_path else ""
        if candidate and candidate.lower() not in _GENERIC_URL_NAMES:
            return candidate
    return fallback


def download(
    source: PathInput,
    output_dir: PathInput = ".",
    overwrite: bool = False,
    verbose: bool = True,
    *,
    timeout: float = 30.0,
    retries: int = 0,
    expected_sha256: str | None = None,
) -> None:
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
    None

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
    >>> files.download(url, output_dir="weights")
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
    destination_dir = to_path(output_dir)
    mkdir(destination_dir, parents=True, exist_ok=True)

    if source_text.startswith(("http://", "https://", "ftp://")):
        download_url, is_gdrive = _normalize_download_url(source_text)
        gdrive_id = _extract_gdrive_file_id(source_text) if is_gdrive else None
        parsed = urlparse(download_url)
        initial_name = name(parsed.path) if parsed.path else ""
        if initial_name.lower() in _GENERIC_URL_NAMES:
            initial_name = gdrive_id or "downloaded_file"
        preliminary_dest = destination_dir / initial_name
        if (
            not is_gdrive
            and (exists(preliminary_dest) or is_symlink(preliminary_dest))
            and not overwrite
        ):
            raise FileExistsError(
                f"File already exists: {preliminary_dest}. Use overwrite=True to replace it."
            )
        headers = {"User-Agent": "klygo/2.0"}
        for attempt in range(retries + 1):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{name(preliminary_dest)}.", suffix=".tmp", dir=destination_dir
            )
            temporary = to_path(temporary_name, expand_user=False)
            os.close(descriptor)
            destination = preliminary_dest
            try:
                try:
                    request = urllib.request.Request(download_url, headers=headers)
                    response = urllib.request.urlopen(request, timeout=timeout)
                    try:
                        if is_gdrive and "text/html" in (response.headers.get("content-type", "") or "").lower():
                            html_text = response.read(524288).decode("utf-8", errors="ignore")
                            confirm_url = _extract_gdrive_confirm_url(html_text, gdrive_id)
                            if confirm_url:
                                response.close()
                                response = urllib.request.urlopen(
                                    urllib.request.Request(confirm_url, headers=headers),
                                    timeout=timeout,
                                )
                            if "text/html" in (response.headers.get("content-type", "") or "").lower():
                                raise OSError(
                                    "Google Drive download returned an HTML page instead of a file; "
                                    "check sharing permissions or download quota."
                                )
                        resolved_name = _resolve_remote_filename(
                            source_text, response, fallback=initial_name
                        )
                        destination = destination_dir / resolved_name
                        if (exists(destination) or is_symlink(destination)) and not overwrite:
                            raise FileExistsError(
                                f"File already exists: {destination}. Use overwrite=True to replace it."
                            )
                        declared = response.headers.get("content-length")
                        total_size = int(declared) if declared is not None else None
                        digest = hashlib.sha256()
                        downloaded = 0
                        with temporary.open("wb") as stream, ProgressBar(
                            total=total_size,
                            desc=f"Downloading {name(destination)}",
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
                    finally:
                        response.close()
                    if total_size is not None and downloaded != total_size:
                        raise OSError(f"Incomplete download: expected {total_size} bytes, received {downloaded}")
                except FileExistsError:
                    raise
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
                return
            finally:
                remove(temporary, missing_ok=True)

    source_path = to_path(source)
    if not exists(source_path):
        raise FileNotFoundError(f"Source file does not exist: {source_path}")
    if is_dir(source_path):
        raise ValueError(f"download() only supports single files, not directories: {source_path}")
    if expected_sha256 is not None and file_hash(source_path, "sha256") != expected_sha256.lower():
        raise ValueError("Source file does not match expected_sha256")

    try:
        from google.colab import files as colab_files

        colab_files.download(str(source_path))
        return
    except ImportError:
        pass

    destination = destination_dir / name(source_path)
    if resolve(destination) != resolve(source_path):
        copy(source_path, destination, overwrite=overwrite)


__all__ = ["download"]
