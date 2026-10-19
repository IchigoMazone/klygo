# `config.load`

```python
config.load(path, verbose=True)
```

Load a structured configuration as a dot-accessible `box.Box`.

## Contract

The document root must be a mapping. Serialization is delegated to `files.load`. TOML null markers written by `config.save` are restored recursively. When `default.root` exists, nested strings beginning with `./` are resolved below that root.

## Parameters

- `path`: `str | pathlib.Path` — existing configuration file.
- `verbose`: `bool` — enable the underlying progress indicator.

## Returns

`box.Box` containing an independent loaded configuration.

## Errors and edge cases

Raises `FileNotFoundError` for a missing file and `TypeError` when the decoded root is not a mapping. Parser and unsupported-extension errors propagate. Boolean model flags remain booleans in every structured format.

## AI usage guidance

Prefer `from klygo import config` followed by `config.load(...)`. Do not call format parsers directly or depend on private root-expansion/null-marker helpers. The returned `Box` can be passed directly to `models.configure()` or `models.load()`.

## Example

See [`load.py`](../../../examples/config/load.py).

## Tests

See [`test_io.py`](../../../test/config/test_io.py).

## Complete executable example

```python
"""Load dot-accessible settings and model metadata."""

from tempfile import TemporaryDirectory

from klygo import config, files, models


with TemporaryDirectory() as directory:
    root = files.path(directory)

    settings_path = root / "settings.json"
    config.save(settings_path, {"model": {"batch": 16}}, verbose=False)
    assert config.load(settings_path, verbose=False).model.batch == 16

    flags = models.flags(model=False, processor=True, post=True)
    schema = models.metadata(flags)
    metadata_path = root / "model.toml"
    config.save(metadata_path, schema, verbose=False)

    restored = config.load(metadata_path, verbose=False)
    assert restored.flags.model is False
    assert restored.flags.processor is True
    assert restored["class"] is None
```
