# `files.save`

```python
files.save(path, data, overwrite=False, verbose=True, indent=4, fieldnames=None)
```

Save structured data using the destination extension.

## Contract

Creates parent directories. Refuses to replace an existing file unless overwrite=True.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `path`: `str | pathlib.Path` — filesystem path to inspect or operate on.
- `data`: `Any` — Python object to serialize.
- `overwrite`: `bool` — permit replacing an existing target.
- `verbose`: `bool` — enable the progress indicator.
- `indent`: `int` — JSON indentation width.
- `fieldnames`: `list[str] | None` — explicit CSV columns.

## Returns

`None`; the filesystem is modified as a side effect.

## Errors and edge cases

Raises `FileExistsError` when replacement is disabled and `ValueError` for unsupported extensions. Serialization and permission errors propagate unchanged.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.save(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`save.py`](../../examples/files/save.py) for an executable example.

## Tests

See [`test_save.py`](../../test/files/test_save.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.save."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    output = Path(directory) / "config.yaml"
    files.save(output, {"batch": 16}, overwrite=True, verbose=False)
    print(output.read_text(encoding="utf-8"))


# Additional cases
# The extension selects the serializer; parent folders are created.
with TemporaryDirectory() as directory:
    root = Path(directory)
    files.save(root / "nested" / "records.jsonl", [{"id": 1}, {"id": 2}], verbose=False)
    assert files.load(root / "nested" / "records.jsonl", verbose=False) == [{"id": 1}, {"id": 2}]
```
