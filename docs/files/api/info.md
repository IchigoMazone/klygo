# `files.info`

```python
files.info(path, *, include_hash=True)
```

Collect common metadata for a file or directory.

## Contract

Returns name, stem, extension, parent, size, timestamps, type flags, and a file hash. With `include_hash=False`, the same dictionary keys are returned with `hash=None`, without reading file contents. Computing directory size still traverses its descendants.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `path`: `str | os.PathLike[str]` — filesystem path to inspect or operate on.
- `include_hash`: `bool`, default `True` — calculate the file checksum; keyword-only.

## Returns

A metadata dictionary.

## Errors and edge cases

Raises `FileNotFoundError` when the path does not exist and `TypeError` for a non-boolean `include_hash`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.info(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`info.py`](../../../examples/files/info.py) for an executable example.

## Tests

See [`test_info.py`](../../../test/files/test_info.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.info."""

from klygo import files

metadata = files.info("README.md")
print(metadata["name"], metadata["human_size"], metadata["hash"])


# Additional cases
assert metadata["is_file"] is True
assert metadata["size"] > 0
assert isinstance(metadata["hash"], str)

# Skip content hashing when only metadata is needed.
lightweight = files.info("README.md", include_hash=False)
assert lightweight["hash"] is None
assert lightweight.keys() == metadata.keys()
```
