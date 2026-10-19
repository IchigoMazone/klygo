# `config.save`

```python
config.save(path, data, overwrite=False, verbose=True)
```

Save mapping-based configuration data using the destination extension.

## Contract

Accepts `dict`, other mappings, or `Box`; converts them to a plain dictionary and delegates encoding to `files.save`. For TOML, nested `None` values are represented internally by a private marker and transparently restored by `config.load`.

## Parameters

- `path`: destination `str | pathlib.Path`.
- `data`: mapping or `Box` to encode, including values returned by `models.metadata()` and `models.configure()`.
- `overwrite`: replace an existing file when true.
- `verbose`: enable the progress indicator.

## Returns

The destination `pathlib.Path`.

## Errors and edge cases

Raises `TypeError` for non-mapping data, `FileExistsError` when overwrite is disabled, and `ValueError` for unsupported formats. Applications should not create the private TOML null marker manually.

## AI usage guidance

Use this instead of `files.save` when the input is semantically configuration data and should remain mapping-only. It is the stateless choice for persisting model metadata; use `Config.create_default(..., metadata=value)` when a loaded, path-bound manager is also needed.

## Example

See [`save.py`](../../../examples/config/save.py).

## Tests

See [`test_io.py`](../../../test/config/test_io.py).

## Complete executable example

```python
"""Save ordinary configuration and model metadata by destination extension."""

from tempfile import TemporaryDirectory

from klygo import config, files, models


with TemporaryDirectory() as directory:
    root = files.path(directory)

    settings_path = root / "settings.yaml"
    assert config.save(
        settings_path,
        {"debug": False},
        verbose=False,
    ) == settings_path

    flags = models.flags(model=False, processor=True, post=True)
    schema = models.metadata(flags)
    metadata_path = root / "model.toml"
    config.save(metadata_path, schema, verbose=False)

    restored = config.load(metadata_path, verbose=False)
    assert restored.to_dict() == schema.to_dict()
```
