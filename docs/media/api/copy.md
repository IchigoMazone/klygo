# `media.copy`

```python
media.copy(source, target, overwrite=False)
```

Copy a media file or directory tree to a destination path with staged overwrite protection.

## Contract

Validates argument types, resolves `source` via `klygo.files.path()`, verifies that `source` exists, and delegates the filesystem copy operation to [`klygo.files.copy()`](../../files/api/copy.md). Parent directories of `target` are created automatically, and existing destination entries are protected unless `overwrite=True`.

Because `media.copy()` copies the underlying bytes without re-encoding, image and video bitstreams, EXIF tags, audio tracks, and container metadata are preserved identically.

## Parameters

- `source`: `str | os.PathLike[str]` — existing media file or directory to copy.
- `target`: `str | os.PathLike[str]` — destination file or directory path.
- `overwrite`: `bool`, default `False` — allow replacing an existing file or directory at `target`.

## Returns

`pathlib.Path` — resolved path to the copied destination file or directory.

## Errors and edge cases

Raises `TypeError` if `source` or `target` is not a string or `os.PathLike`, or if `overwrite` is not a `bool`. Raises `FileNotFoundError` when `source` does not exist. Raises `FileExistsError` when `target` already exists and `overwrite=False`.

## Example

See [`copy.py`](../../../examples/media/copy.py) for a standalone runnable script.

## Complete executable example

```python
"""Copy media files and directories without re-encoding."""

import tempfile
from PIL import Image
from klygo import files, media

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    source = media.save(root / "source.png", Image.new("RGB", (32, 24), "red"))

    # Copy to a new path (parent directories are created automatically)
    duplicate = media.copy(source, root / "backup" / "copy.png")
    assert files.is_file(duplicate)
    assert media.probe(duplicate)["size"] == (32, 24)
    assert files.compare(source, duplicate, by="hash")

    # Replace an existing target when overwrite=True
    updated = media.save(root / "blue.png", Image.new("RGB", (64, 48), "blue"))
    media.copy(updated, duplicate, overwrite=True)
    assert media.probe(duplicate)["size"] == (64, 48)
    print(media.probe(duplicate)["size"])
```
