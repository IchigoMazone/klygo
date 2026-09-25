# `archive.compress`

```python
archive.compress(source, output_path, format=None, overwrite=False, verbose=True, compresslevel=6, method=None, follow_symlinks=False, include_root=True)
```

Create an archive from a file or directory.

## Contract

The output backend is selected from `format` or the destination extension. ZIP, TAR, TAR.GZ, TAR.XZ, TAR.BZ2, GZ, and optionally 7Z can be written.
Archive paths accept both strings and `pathlib.Path` values. Format-specific work is delegated to the selected archive backend.

## Parameters

- `source`: `str or pathlib.Path` — File or directory to archive.
- `output_path`: `str or pathlib.Path` — Destination archive path.
- `format`: `str or None, default=None` — Explicit output format; inferred when omitted.
- `overwrite`: `bool, default=False` — Allow existing output entries to be replaced.
- `verbose`: `bool, default=True` — Display archive progress.
- `compresslevel`: `int, default=6` — Compression level, usually from 1 through 9.
- `method`: `str or None, default=None` — Backend-specific compression method.
- `follow_symlinks`: `bool, default=False` — Follow symbolic links instead of archiving link entries.
- `include_root`: `bool, default=True` — Include the source directory name as the top-level archive member. Set this to `False` to archive only the directory contents. This option has no effect when `source` is a single file.

## Returns

`None` — The archive is created as a side effect.

## Errors and edge cases

Raises `FileNotFoundError` if `source` does not exist.

Raises `FileExistsError` if the destination exists and overwrite is disabled.

Raises `ValueError` if the format or backend parameters are invalid.

## AI usage guidance

Prefer the public form `from klygo import archive` followed by `archive.compress(...)`. Use `klygo.files` for surrounding filesystem work. Do not import `archive.backend` directly unless implementing a new archive format.

## Example

See [`compress.py`](../../examples/archive/compress.py) for an executable example.

## Tests

See [`test_compress.py`](../../test/archive/test_compress.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.archive.compress."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.archive as archive
import klygo.files as files

from examples.archive._support import make_archive, make_source, member_ending


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = make_source(root)

    zip_path = root / "dataset.zip"
    archive.compress(source, zip_path, verbose=False)
    assert archive.detect_format(zip_path) == "zip"
    assert all(name.startswith("source/") for name in archive.list_files(zip_path))

    contents_only = root / "contents-only.zip"
    archive.compress(source, contents_only, include_root=False, verbose=False)
    assert "alpha.txt" in archive.list_files(contents_only)
    assert "source/alpha.txt" not in archive.list_files(contents_only)

    tar_path = root / "dataset.tar.gz"
    archive.compress(source, tar_path, compresslevel=9, verbose=False)
    assert archive.test(tar_path)

    print(zip_path, contents_only, tar_path)
```
