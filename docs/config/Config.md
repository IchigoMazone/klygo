# `config.Config`

```python
config.Config(config_path)
```

Manage one configuration file through a stateful object interface.

## Contract

Construction stores a validated path without reading it. `read` loads state; access/composition methods update memory; `export_file` writes the current state in another format.

## Parameters

- `config_path`: source configuration `str | pathlib.Path`.
- Key methods: `read`, `get`, `set`, `has`, `delete`, `merge`, `update`, `to_dict`, `to_json`, and `export_file`.
- `create_default(path, ...)`: class constructor that first creates a default file.

## Returns

Methods return `Box`, plain dictionaries, strings, booleans, values, or destination paths according to their operation.

## Errors and edge cases

Calling state accessors before `read` operates on empty state. File errors occur when reading/exporting. `ext` remains a compatibility alias for `export_file(..., suffix=...)`.

## AI usage guidance

Use the function API for stateless transformations and `Config` when several operations intentionally share mutable in-memory state.

## Example

See [`Config.py`](../../examples/config/Config.py).

## Tests

See [`test_config.py`](../../test/config/test_config.py).

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
