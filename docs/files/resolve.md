# `files.resolve`

```python
files.resolve(value, strict=False)
```

Return an absolute normalized Path.

## Contract

strict=True requires every path component to exist.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `value`: `str | pathlib.Path` — path-like value to transform.
- `strict`: `bool` — require the resolved path to exist.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

With `strict=True`, raises `FileNotFoundError` when the path does not exist.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.resolve(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`resolve.py`](../../examples/files/resolve.py) for an executable example.

## Tests

See [`test_resolve.py`](../../test/files/test_resolve.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.resolve."""

from klygo import files

print(files.resolve("README.md", strict=True))


# Additional cases
absolute = files.resolve(".")
assert absolute.is_absolute()
try:
    files.resolve("missing-path", strict=True)
except FileNotFoundError:
    print("strict resolution requires existence")
```
