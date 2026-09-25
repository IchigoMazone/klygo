# `config.load`

```python
config.load(path, verbose=True)
```

Load a structured configuration as a dot-accessible `box.Box`.

## Contract

The document root must be a mapping. Serialization is delegated to `files.load`. When `default.root` exists, nested strings beginning with `./` are resolved below that root.

## Parameters

- `path`: `str | pathlib.Path` — existing configuration file.
- `verbose`: `bool` — enable the underlying progress indicator.

## Returns

`box.Box` containing an independent loaded configuration.

## Errors and edge cases

Raises `FileNotFoundError` for a missing file and `TypeError` when the decoded root is not a mapping. Parser and unsupported-extension errors propagate.

## AI usage guidance

Prefer `from klygo import config` followed by `config.load(...)`. Do not call format parsers directly or depend on private root-expansion helpers.

## Example

See [`load.py`](../../../examples/config/load.py).

## Tests

See [`test_io.py`](../../../test/config/test_io.py).

## Complete executable example

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from klygo import config

with TemporaryDirectory() as directory:
    path = Path(directory) / "settings.json"
    config.save(path, {"model": {"batch": 16}}, verbose=False)
    assert config.load(path, verbose=False).model.batch == 16
```
