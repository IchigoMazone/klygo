# `config.export`

```python
config.export(source, target, overwrite=False, verbose=True)
```

Export either an existing configuration file or an in-memory configuration.

## Contract

Path sources are routed through `convert`; mapping and `Box` sources are routed through `save`.

## Parameters

- `source`: `str | Path | Mapping | Box`.
- `target`: destination path.
- `overwrite`: allow target replacement.
- `verbose`: enable progress indicators.

## Returns

The destination `pathlib.Path`.

## Errors and edge cases

Raises `TypeError` for unsupported source types. File and serialization errors propagate from `convert` or `save`.

## AI usage guidance

Choose this when code accepts both paths and mappings; avoid manually branching on source type.

## Example

See [`export.py`](../../../examples/config/export.py).

## Tests

See [`test_io.py`](../../../test/config/test_io.py).

## Complete executable example

```python
"""Export an in-memory configuration."""

from tempfile import TemporaryDirectory

from klygo import config, files

with TemporaryDirectory() as directory:
    output = config.export({"seed": 7}, files.path(directory) / "settings.json", verbose=False)
    assert files.is_file(output)
```
