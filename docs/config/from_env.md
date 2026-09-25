# `config.from_env`

```python
config.from_env(config_data=None, prefix="KLYGO_", sep="_", parse_values=False)
```

Overlay matching environment variables onto a configuration.

## Contract

Names are stripped of the prefix, lower-cased, converted to nested dot paths, and deeply merged. Values remain strings unless JSON parsing is enabled.

## Parameters

- `config_data`: optional base mapping or `Box`.
- `prefix`: environment variable prefix to select and remove.
- `sep`: hierarchy separator used in variable names.
- `parse_values`: decode JSON scalars, arrays, objects, booleans, and null.

## Returns

A new nested plain `dict`.

## Errors and edge cases

Raises `TypeError` for invalid argument types and `ValueError` for an empty separator. Unparseable values remain strings.

## AI usage guidance

Keep `parse_values=False` for conventional environment semantics; enable it only when typed values are expected and trusted.

## Example

See [`from_env.py`](../../examples/config/from_env.py).

## Tests

See [`test_environment.py`](../../test/config/test_environment.py).

## Complete executable example

```python
import os
from klygo import config

os.environ["APP_MODEL_BATCH"] = "32"
settings = config.from_env(prefix="APP_", parse_values=True)
assert settings["model"]["batch"] == 32
del os.environ["APP_MODEL_BATCH"]
```
