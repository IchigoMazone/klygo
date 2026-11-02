# `files.list_entries`

```python
files.list_entries(path='.', pattern='*', recursive=False)
```

List matching files and directories below a directory.

## Contract

Returns Path objects sorted case-insensitively. The root must exist and be a directory.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `path`: `str | os.PathLike[str]` — filesystem path to inspect or operate on.
- `pattern`: `str` — glob expression.
- `recursive`: `bool` — include descendants.

## Returns

A deterministically sorted `list[pathlib.Path]`.

## Errors and edge cases

Raises `FileNotFoundError` for a missing root and `ValueError` when the root is not a directory.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.list_entries(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`list_entries.py`](../../../examples/files/list_entries.py) for an executable example.

## Tests

See [`test_list_entries.py`](../../../test/files/test_list_entries.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.list_entries."""

from klygo import files

for entry in files.list_entries("klygo", pattern="*.py"):
    print(entry)


# Additional cases
# Recursive listing includes matching entries in descendants.
recursive = files.list_entries("klygo", pattern="*.py", recursive=True)
assert all(item.suffix == ".py" for item in recursive)
```
