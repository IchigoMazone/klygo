# `files.common_path`

```python
files.common_path(values)
```

Return the longest common parent for multiple paths.

## Contract

Rejects an empty iterable and a single string passed in place of an iterable.

Inputs accept strings and text-returning `os.PathLike` objects (including `pathlib.Path`), using the current operating system's path rules. Byte paths are not supported.

## Parameters

- `values`: `Iterable[str | os.PathLike[str]]` — non-empty path collection.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

Raises `ValueError` for an empty input and `TypeError` when passed a single string.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.common_path(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`common_path.py`](../../../examples/files/common_path.py) for an executable example.

## Tests

See [`test_common_path.py`](../../../test/files/test_common_path.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.common_path."""

from klygo import files

print(files.common_path(["dataset/images/train", "dataset/labels/train"]))


# Additional cases
assert files.common_path(["dataset/images", "dataset/labels"]) == files.path("dataset")
try:
    files.common_path([])
except ValueError:
    print("empty collections are rejected")
```
