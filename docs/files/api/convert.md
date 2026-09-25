# `files.convert`

```python
files.convert(source, target, overwrite=False, verbose=True)
```

Convert between supported structured-data formats.

## Contract

Loads the source then saves the decoded data to the target format and returns the target Path.

All string paths and `pathlib.Path` values accepted by this API use the current operating system's path rules. Unless explicitly stated, path helpers do not access the filesystem.

## Parameters

- `source`: `str | pathlib.Path` — source URL or path.
- `target`: `str | pathlib.Path` — destination path.
- `overwrite`: `bool` — permit replacing an existing target.
- `verbose`: `bool` — enable the progress indicator.

## Returns

A `pathlib.Path` representing the result.

## Errors and edge cases

Propagates the documented errors from `load` and `save`.

## AI usage guidance

Prefer the public form `from klygo import files` followed by `files.convert(...)`. Do not import private helpers or reproduce this behavior with direct `os`/`shutil` calls inside Klygo.

## Example

See [`convert.py`](../../../examples/files/convert.py) for an executable example.

## Tests

See [`test_convert.py`](../../../test/files/test_convert.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.files.convert."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    source = Path(directory) / "data.json"
    target = Path(directory) / "data.yaml"
    files.save(source, {"value": 42}, verbose=False)
    files.convert(source, target, verbose=False)
    print(files.load(target, verbose=False))


# Additional cases
# Conversion keeps the decoded Python value unchanged.
with TemporaryDirectory() as directory:
    root = Path(directory)
    source, target = root / "settings.yaml", root / "settings.json"
    files.save(source, {"enabled": True}, verbose=False)
    files.convert(source, target, verbose=False)
    assert files.load(target, verbose=False) == {"enabled": True}
```
