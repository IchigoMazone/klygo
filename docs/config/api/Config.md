# `config.Config`

```python
config.Config(config_path)
```

Manage one configuration file through a stateful object interface.

## Contract

Construction stores a validated path without reading it. `read` loads state; access/composition methods update memory; `export_file` writes the current state in another format. `Config.create_default(...)` is the exception: it creates the file and returns an already-loaded manager.

## Parameters

- `config_path`: source configuration `str | pathlib.Path`.
- Key methods: `read`, `get`, `set`, `has`, `delete`, `merge`, `update`, `to_dict`, `to_json`, and `export_file`.
- `create_default(path, default_data=None, ...)`: create general Klygo defaults with optional deep overrides.
- `create_default(path, metadata=value, ...)`: write `models.metadata()` or `models.configure()` output exactly, without injecting general defaults.

## Returns

Methods return `Box`, plain dictionaries, strings, booleans, values, or destination paths according to their operation.

## Errors and edge cases

Calling state accessors before `read` operates on empty state. A manager returned by `create_default` is already populated. `default_data` and `metadata` are mutually exclusive. File errors occur when reading/exporting. `ext` remains a compatibility alias for `export_file(..., suffix=...)`.

## AI usage guidance

Use the function API for stateless transformations and `Config` when several operations intentionally share mutable in-memory state. Use the `metadata=` mode for model schemas; passing model metadata as `default_data` would incorrectly merge it with general Klygo defaults.

## Example

See [`Config.py`](../../../examples/config/Config.py).

## Tests

See [`test_Config.py`](../../../test/config/test_Config.py) and [`test_model_metadata.py`](../../../test/config/test_model_metadata.py).

## Complete executable example

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from klygo import Config

with TemporaryDirectory() as directory:
    manager = Config.create_default(Path(directory) / "settings.yaml", verbose=False)
    manager.read(verbose=False)
    manager.set("model.batch", 32)
    assert manager.get("model.batch") == 32
```

## Model metadata example

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from klygo import Config, models

flags = models.flags(model=False, processor=True, post=True)
schema = models.metadata(flags)

with TemporaryDirectory() as directory:
    manager = Config.create_default(
        Path(directory) / "model.toml",
        metadata=schema,
        verbose=False,
    )
    assert manager.get("flags.model") is False
    assert manager.get("flags.processor") is True
    assert manager.get("class") is None
```

See the complete [model metadata guide](../model-metadata.md).
