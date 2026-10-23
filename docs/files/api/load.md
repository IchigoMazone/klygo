# `files.load`

```python
files.load(path, as_lines=False, verbose=True)
```

Load structured data inferred from the file extension.

## Contract

This API returns complete data in memory. For incremental JSONL/CSV processing, use [`iter_load()`](iter_load.md).

Returns decoded Python data. Raises FileNotFoundError for missing paths and ValueError for directories or unsupported formats.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `path`: `str | os.PathLike[str]` — filesystem path to inspect or operate on.
- `as_lines`: `bool` — return TXT/LOG content as separate lines.
- `verbose`: `bool` — enable the progress indicator.

## Returns

`Any` decoded according to the input format.

## Errors and edge cases

Raises `FileNotFoundError` for a missing input and `ValueError` for a directory or unsupported extension. Parser-specific errors propagate unchanged.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.load(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`load.py`](../../../examples/files/load.py) for an executable example.

## Tests

See [`test_load.py`](../../../test/files/test_load.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.load."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    source = files.path(directory) / "config.json"
    files.save(source, {"model": "yolo"}, overwrite=True, verbose=False)
    print(files.load(source, verbose=False))


# Additional cases
# Load line-oriented text without retaining newline characters.
with TemporaryDirectory() as directory:
    labels = files.path(directory) / "labels.txt"
    files.save(labels, ["cat", "dog"], verbose=False)
    assert files.load(labels, as_lines=True, verbose=False) == ["cat", "dog"]
```
