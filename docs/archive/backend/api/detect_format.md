# `backend.detect_format`

```python
from klygo.archive.backend import detect_format

fmt = detect_format(path)
```

Detect an archive format from its extension and file signature (magic bytes).

## Contract

`detect_format` inspects compound extensions first (`.tar.gz`, `.tgz`, `.tar.xz`, `.txz`, `.tar.bz2`, `.tbz2`), then checks file header magic bytes if the file exists on disk, and finally falls back to simple suffixes. Normalized canonical format strings are returned.

## Parameters

- `path`: `str or pathlib.Path` — Archive file path or candidate name.

## Returns

`str`
    Canonical format identifier (`"zip"`, `"tar"`, `"tar.gz"`, `"tar.xz"`, `"tar.bz2"`, `"gz"`, `"7z"`, `"rar"`).

## Errors and edge cases

Raises `ValueError` if neither the filename extension nor file signature corresponds to a supported archive format.

## AI usage guidance

Use `detect_format` before dispatching to format-specific handlers or when verifying candidate archive files.

## Example

See [`detect_format.py`](../../../../examples/archive/backend/detect_format.py).

## Tests

See [`test_detect_format.py`](../../../../test/archive/backend/test_detect_format.py).

## Complete executable example

```python
from klygo.archive.backend import detect_format

assert detect_format("dataset.tgz") == "tar.gz"
assert detect_format("data.zip") == "zip"
assert detect_format("archive.7z") == "7z"
```
