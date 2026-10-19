# `files.download`

```python
files.download(source, output_dir='.', overwrite=False, verbose=True, *, timeout=30.0, retries=0, expected_sha256=None)
```

Download a URL or copy a local file while preserving its name.

## Contract

Supports HTTP, HTTPS, FTP, local files, and Google Colab download behavior. Remote data is written to a temporary file and published only after transfer and optional SHA-256 validation succeed. A declared content length is checked. Directories are rejected.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `source`: `str | os.PathLike[str]` — source URL or path.
- `output_dir`: `str | os.PathLike[str]` — destination directory.
- `overwrite`: `bool` — permit replacing an existing target.
- `verbose`: `bool` — enable the progress indicator.
- `timeout`: positive number of seconds, default `30.0` — network request timeout.
- `retries`: non-negative integer, default `0` — additional attempts after network or incomplete-transfer errors.
- `expected_sha256`: 64-character hexadecimal string or `None` — verify content before publication.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

Raises `FileNotFoundError` for a missing local source, `ValueError` for invalid options, directory sources, or checksum mismatches, and `FileExistsError` when replacement is disabled. Incomplete transfers and network errors propagate after retries. A failed transfer leaves any previous destination intact.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.download(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`download.py`](../../../examples/files/download.py) for an executable example.

## Tests

See [`test_download.py`](../../../test/files/test_download.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.download."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "source.bin"
    source.write_bytes(b"klygo")
    downloaded = files.download(source, root / "downloads", verbose=False)
    print(downloaded, downloaded.read_bytes())


# Additional cases
# Existing destinations are protected unless overwrite=True.
with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "weights.bin"
    source.write_bytes(b"v1")
    destination = files.download(source, root / "cache", verbose=False)
    try:
        files.download(source, root / "cache", verbose=False)
    except FileExistsError:
        print("destination protected:", destination)
```
