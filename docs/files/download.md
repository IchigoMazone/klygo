# `files.download`

```python
files.download(source, output_dir='.', overwrite=False, verbose=True)
```

Download a URL or copy a local file while preserving its name.

## Contract

Supports HTTP, HTTPS, FTP, local files, and Google Colab download behavior. Directories are rejected.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `source`: `str | pathlib.Path` — source URL or path.
- `output_dir`: `str | pathlib.Path` — destination directory.
- `overwrite`: `bool` — permit replacing an existing target.
- `verbose`: `bool` — enable the progress indicator.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

Raises `FileNotFoundError` for a missing local source, `ValueError` for a directory source, and `FileExistsError` when replacement is disabled. Network errors propagate unchanged.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.download(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`download.py`](../../examples/files/download.py) for an executable example.

## Tests

See [`test_download.py`](../../test/files/test_download.py) for the executable behavioral contract.

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
