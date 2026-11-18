# `config.Config`

```python
config.Config(config_path)
```

Manage one configuration file through a stateful object interface.

## Contract


## Parameters

- `config_path`: source configuration `str | pathlib.Path`.
- `create_default(path, default_data=None, ...)`: create general Klygo defaults with optional deep overrides.
- `create_default(path, metadata=value, ...)`: write `models.metadata()` or `models.configure()` output exactly, without injecting general defaults.

## Returns

Methods return `Box`, plain dictionaries, strings, booleans, values, or destination paths according to their operation.

## Errors and edge cases


## AI usage guidance

Use the function API for stateless transformations and `Config` when several operations intentionally share mutable in-memory state. Use the `metadata=` mode for model schemas; passing model metadata as `default_data` would incorrectly merge it with general Klygo defaults.

## Example

See [`Config.py`](../../../examples/config/Config.py).

## Tests

See [`test_config.py`](../../../test/config/test_config.py) and [`test_model_metadata.py`](../../../test/config/test_model_metadata.py).

## Complete executable example

```python
"""Use Config for general settings and exact model metadata."""

from tempfile import TemporaryDirectory

from klygo import Config, files, models


with TemporaryDirectory() as directory:
    root = files.path(directory)

    settings = Config.create_default(root / "settings.yaml", verbose=False)
    settings.set("model.batch", 32)
    assert settings.get("model.batch") == 32

    flags = models.flags(model=False, processor=True, post=True)
    schema = models.metadata(flags)
    metadata = Config.create_default(
        root / "model.toml",
        metadata=schema,
        verbose=False,
    )

    assert metadata.get("flags.model") is False
    assert metadata.get("flags.post") is True
    assert metadata.get("class") is None
```

See the complete [model metadata guide](../model-metadata.md).
