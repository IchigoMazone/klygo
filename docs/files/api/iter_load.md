# `files.iter_load`

```python
files.iter_load(path)
```

Yield JSONL values or CSV records one at a time.

## Contract

Supports UTF-8 JSONL and CSV only. JSONL yields each decoded JSON value and
ignores blank lines. CSV uses the first record as field names and yields row
dictionaries as in [`load()`](load.md). Quoted multiline CSV fields are supported.

The input and extension are validated immediately. The stream opens on first
iteration and closes on exhaustion, an exception, or generator `close()`.
Use `contextlib.closing` when processing may stop early. Memory usage depends
on the current record, rather than the full file. No progress bar is created.

Inputs accept strings and text-returning `os.PathLike` objects, including `Path`.
Byte paths are not supported. Paths follow the host operating system's rules;
these I/O and traversal APIs do not expand `~`. Use `files.path()` first to do so.

## Parameters

- `path`: `str | os.PathLike[str]` — existing `.jsonl` or `.csv` file.

## Returns

`Generator[Any, None, None]`: decoded JSON values or CSV row dictionaries. Empty files yield no records.

## Errors and edge cases

`FileNotFoundError` for a missing input; `ValueError` for a directory or unsupported extension; `TypeError` for an invalid path input. JSON parsing errors (`json.JSONDecodeError`), decoding errors, CSV errors, and I/O errors propagate when the affected record is consumed.

## AI usage guidance

Prefer `from klygo import files` and `files.iter_load(...)`. Use for incremental JSONL/CSV processing. Keep `load()` when the complete data object is required. Close the generator explicitly if iteration ends early.

## Example

See [`iter_load.py`](../../../examples/files/iter_load.py) for an executable example.

## Tests

See [`test_iter_load.py`](../../../test/files/test_iter_load.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.iter_load."""

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files

with TemporaryDirectory() as directory:
    source = Path(directory) / "records.jsonl"
    files.save(source, [{"id": 1}, {"id": 2}], verbose=False)
    assert list(files.iter_load(source)) == [{"id": 1}, {"id": 2}]

    # Closing the iterator releases the stream when processing stops early.
    with closing(files.iter_load(source)) as records:
        first = next(records)
        assert first == {"id": 1}
        print(first)

    table = Path(directory) / "records.csv"
    files.save(table, [{"id": "1", "name": "cat"}], verbose=False)
    assert list(files.iter_load(table)) == [{"id": "1", "name": "cat"}]
```
