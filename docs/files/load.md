# `files.load`

```python
files.load(path, as_lines=False, verbose=True)
```

Load structured data inferred from the file extension.

## Contract

Returns decoded Python data. Raises FileNotFoundError for missing paths and ValueError for directories or unsupported formats.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `path`: `str | pathlib.Path` — filesystem path to inspect or operate on.
- `as_lines`: `bool` — return TXT/LOG content as separate lines.
- `verbose`: `bool` — enable the progress indicator.

## Returns

`Any` decoded according to the input format.

## Errors and edge cases

Raises `FileNotFoundError` for a missing input and `ValueError` for a directory or unsupported extension. Parser-specific errors propagate unchanged.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.load(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`load.py`](../../examples/files/load.py) for an executable example.

## Tests

See [`test_load.py`](../../test/files/test_load.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.load."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    source = Path(directory) / "config.json"
    files.save(source, {"model": "yolo"}, overwrite=True, verbose=False)
    print(files.load(source, verbose=False))


# Additional cases
# Load line-oriented text without retaining newline characters.
with TemporaryDirectory() as directory:
    labels = Path(directory) / "labels.txt"
    labels.write_text("cat\ndog\n", encoding="utf-8")
    assert files.load(labels, as_lines=True, verbose=False) == ["cat", "dog"]
```
