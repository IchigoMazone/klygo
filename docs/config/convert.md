# `config.convert`

```python
config.convert(source, target, overwrite=False, verbose=True)
```

Convert a configuration file between supported structured formats.

## Contract

Equivalent to `save(target, load(source))`; target extension selects the encoder and source data must have a mapping root.

## Parameters

- `source`: existing configuration path.
- `target`: output path with the desired extension.
- `overwrite`: allow replacement of an existing target.
- `verbose`: enable file progress indicators.

## Returns

The target `pathlib.Path`.

## Errors and edge cases

Input, parsing, unsupported-format, and overwrite errors propagate without being hidden.

## AI usage guidance

Use `config.convert` for path-to-path conversion; use `config.export` when the source may already be an in-memory mapping.

## Example

See [`convert.py`](../../examples/config/convert.py).

## Tests

See [`test_io.py`](../../test/config/test_io.py).

## Complete executable example

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from klygo import config

with TemporaryDirectory() as directory:
    root = Path(directory)
    source = config.save(root / "a.yaml", {"seed": 7}, verbose=False)
    assert config.convert(source, root / "a.toml", verbose=False).is_file()
```
