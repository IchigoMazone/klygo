# `config.save`

```python
config.save(path, data, overwrite=False, verbose=True)
```

Save mapping-based configuration data using the destination extension.

## Contract

Accepts `dict`, other mappings, or `Box`; converts them to a plain dictionary and delegates encoding to `files.save`.

## Parameters

- `path`: destination `str | pathlib.Path`.
- `data`: mapping or `Box` to encode.
- `overwrite`: replace an existing file when true.
- `verbose`: enable the progress indicator.

## Returns

The destination `pathlib.Path`.

## Errors and edge cases

Raises `TypeError` for non-mapping data, `FileExistsError` when overwrite is disabled, and `ValueError` for unsupported formats.

## AI usage guidance

Use this instead of `files.save` when the input is semantically configuration data and should remain mapping-only.

## Example

See [`save.py`](../../examples/config/save.py).

## Tests

See [`test_io.py`](../../test/config/test_io.py).

## Complete executable example

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from klygo import config

with TemporaryDirectory() as directory:
    path = Path(directory) / "settings.yaml"
    assert config.save(path, {"debug": False}, verbose=False) == path
```
