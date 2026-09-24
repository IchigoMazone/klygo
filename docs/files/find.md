# `files.find`

```python
files.find(path='.', pattern='*', recursive=True)
```

Find files matching a glob pattern.

## Contract

Only files are returned; matching directories are excluded. Results are sorted case-insensitively.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `path`: `str | pathlib.Path` — filesystem path to inspect or operate on.
- `pattern`: `str` — glob expression.
- `recursive`: `bool` — include descendants or recursively remove a directory.

## Returns

A deterministically sorted `list[pathlib.Path]`.

## Errors and edge cases

Raises `FileNotFoundError` when the search root does not exist.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.find(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`find.py`](../../examples/files/find.py) for an executable example.

## Tests

See [`test_find.py`](../../test/files/test_find.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.find."""

from klygo import files

for source in files.find("klygo", pattern="*.py"):
    print(source)


# Additional cases
# Disable recursion when only direct children are wanted.
direct = files.find("klygo", pattern="*.py", recursive=False)
assert all(item.parent.name == "klygo" for item in direct)
```
