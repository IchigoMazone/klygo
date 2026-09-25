# `config.create`

```python
config.create(path, default_data=None, overwrite=False, verbose=True)
```

Create a configuration file from Klygo defaults and return its loaded content.

## Contract

Calls `defaults`, writes with `save`, then reloads with `load`, so the returned value reflects actual serialization behavior.

## Parameters

- `path`: destination configuration file.
- `default_data`: optional deep default overrides.
- `overwrite`: allow replacement.
- `verbose`: enable progress indicators.

## Returns

The created configuration as `box.Box`.

## Errors and edge cases

Existing targets raise `FileExistsError` unless overwrite is enabled. Format and validation errors propagate.

## AI usage guidance

Use `create` for a one-step default file; use `defaults` when no file should be written.

## Example

See [`create.py`](../../../examples/config/create.py).

## Tests

See [`test_creation.py`](../../../test/config/test_creation.py).

## Complete executable example

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from klygo import config

with TemporaryDirectory() as directory:
    created = config.create(Path(directory) / "settings.yaml", verbose=False)
    assert created.model.name == "yolov8n"
```
