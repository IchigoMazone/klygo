# `files.iter_find`

```python
files.iter_find(path='.', pattern='*', recursive=True, *, exclude_dirs=())
```

Yield matching regular files with optional directory pruning.

## Contract

Returns an unsorted generator of `Path` objects without collecting all matches.
Patterns match file basenames with `*`, `?`, and character classes, using the
host's case rules. Exclusion patterns match directory basenames and prune those
subtrees. Nested directory symlinks are not followed. File symlinks are included
when their targets are regular files; broken symlinks are excluded.

The root and options are validated immediately; scanning starts on iteration.
Traversal retains per-directory entries, so memory is not independent of the
number of entries in an individual directory. Scan errors propagate instead
of silently returning incomplete results.

Use [`find()`](find.md) for a sorted list or root-relative pathlib glob patterns.

Inputs accept strings and text-returning `os.PathLike` objects, including `Path`.
Byte paths are not supported. Paths follow the host operating system's rules;
these I/O and traversal APIs do not expand `~`. Use `files.path()` first to do so.

## Parameters

- `path`: `str | os.PathLike[str]`, default `'.'` — existing search directory.
- `pattern`: `str`, default `'*'` — filename wildcard without `/` or `\`.
- `recursive`: `bool`, default `True` — visit descendant directories.
- `exclude_dirs`: `Iterable[str]`, default `()` — directory basename wildcards; keyword-only. Pass a sequence, not a single string.

## Returns

`Generator[pathlib.Path, None, None]` in filesystem order; paths are not reserved.

## Errors and edge cases

`FileNotFoundError` for a missing root; `ValueError` for a non-directory root, empty pattern, or pattern containing a separator; `TypeError` for invalid option types. Directory scanning can raise `OSError` during iteration.

## AI usage guidance

Prefer `from klygo import files` and `files.iter_find(...)`. Use for large directory trees when sorted results are unnecessary. Use `exclude_dirs` to avoid scanning cache directories. Keep `find()` for callers expecting a list.

## Example

See [`iter_find.py`](../../../examples/files/iter_find.py) for an executable example.

## Tests

See [`test_iter_find.py`](../../../test/files/test_iter_find.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.iter_find."""

from tempfile import TemporaryDirectory
from klygo import files

with TemporaryDirectory() as directory:
    root = files.path(directory)
    files.mkdir(root / "images")
    files.mkdir(root / "cache")
    for relative_name in ("top.jpg", "images/cat.jpg", "cache/preview.jpg"):
        (root / relative_name).touch()

    # Patterns match filenames; exclusions prune directories before scanning.
    matches = files.iter_find(root, "*.jpg", exclude_dirs=("cache",))
    assert {files.as_posix(files.relative(item, root)) for item in matches} == {"top.jpg", "images/cat.jpg"}
    assert list(files.iter_find(root, "*.jpg", recursive=False)) == [root / "top.jpg"]
```
