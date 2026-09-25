# `backend.is_archive`

```python
from klygo.archive.backend import is_archive

result = is_archive(path)
```

Check whether a path appears to be a supported archive file.

## Contract

`is_archive` is a non-raising predicate that checks if the given path has a supported extension or magic bytes. Missing, unreadable, or unsupported files safely return `False`.

## Parameters

- `path`: `str or pathlib.Path` — Archive file path or candidate name to inspect.

## Returns

`bool`
    `True` if the path is recognized as a supported archive format, `False` otherwise.

## Errors and edge cases

This function never raises exceptions; any detection or I/O failure returns `False`.

## AI usage guidance

Use `is_archive` in conditional branches or filters when scanning directories containing arbitrary mixed file types.

## Example

See [`is_archive.py`](../../../../examples/archive/backend/is_archive.py).

## Tests

See [`test_is_archive.py`](../../../../test/archive/backend/test_is_archive.py).

## Complete executable example

```python
from klygo.archive.backend import is_archive

assert is_archive("dataset.zip") is True
assert is_archive("dataset.tar.gz") is True
assert is_archive("image.png") is False
```
