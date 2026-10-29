# `files.save`

```python
files.save(path, data, overwrite=False, verbose=True, indent=4, fieldnames=None, *, atomic=True)
```

Save structured data using the destination extension.

## Contract

Creates parent directories. Refuses an existing entry, including a broken
symlink, unless `overwrite=True`. The default `atomic=True` protects an existing
destination when serialization fails. Set `atomic=False` only when direct writing
and the possibility of a partial file are acceptable.

Markdown (`.md`) is written as UTF-8 text without parsing or rendering it.

With `atomic=True`, serialization completes in a sibling temporary file before
publication. Serialization failure leaves an existing destination unchanged.
The temporary file is removed on success or failure. With `overwrite=True`,
the destination entry is replaced; a destination symlink itself is replaced
and its target is left untouched. With `overwrite=False`, hard-link publication
refuses any entry created concurrently at the destination.

Atomic mode uses temporary-file permissions, does not preserve destination
metadata, and does not guarantee durability across power loss.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `path`: `str | os.PathLike[str]` — filesystem path to inspect or operate on.
- `data`: `Any` — Python object to serialize.
- `overwrite`: `bool` — permit replacing an existing target.
- `verbose`: `bool` — enable the progress indicator.
- `indent`: `int` — JSON indentation width.
- `fieldnames`: `list[str] | None` — explicit CSV columns.
- `atomic`: `bool`, default `True` — publish a fully serialized temporary file; keyword-only.

## Returns

`None`; the filesystem is modified as a side effect.

## Errors and edge cases

Raises `FileExistsError` when replacement is disabled and `ValueError` for unsupported extensions. Serialization and permission errors propagate unchanged. Atomic publication with `overwrite=False` requires hard-link support; an unsupported filesystem raises `OSError` without falling back to an overwrite.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.save(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`save.py`](../../../examples/files/save.py) for an executable example.

## Tests

See [`test_save.py`](../../../test/files/test_save.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.save."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    output = files.path(directory) / "config.yaml"
    files.save(output, {"batch": 16}, overwrite=True, verbose=False)
    print(files.load(output, verbose=False))


# Additional cases
# The extension selects the serializer; parent folders are created.
with TemporaryDirectory() as directory:
    root = files.path(directory)
    files.save(root / "nested" / "records.jsonl", [{"id": 1}, {"id": 2}], verbose=False)
    assert files.load(root / "nested" / "records.jsonl", verbose=False) == [{"id": 1}, {"id": 2}]

# Save Markdown without bypassing the public files API.
with TemporaryDirectory() as directory:
    report = files.path(directory) / "report.md"
    files.save(report, "# Report\n\nReady", verbose=False)
    assert files.load(report, verbose=False) == "# Report\n\nReady"

# Publish a complete replacement only after serialization succeeds.
with TemporaryDirectory() as directory:
    output = files.path(directory) / "config.json"
    files.save(output, {"version": 1}, atomic=True, verbose=False)
    try:
        files.save(output, {"invalid": object()}, overwrite=True, atomic=True, verbose=False)
    except TypeError:
        pass
    assert files.load(output, verbose=False) == {"version": 1}
```
