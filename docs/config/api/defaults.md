# `config.defaults`

```python
config.defaults(default_data=None)
```

Build a fresh default Klygo configuration with optional deep overrides.

## Contract

Every call deep-copies built-in defaults. Overrides use the same recursive semantics as `config.update` and never modify global state.

## Parameters

- `default_data`: optional mapping or `Box` containing overrides.

## Returns

A mutable, independent `dict`.

## Errors and edge cases

Raises `TypeError` when overrides are not mapping-like. Nested mappings combine rather than replace by default.

## AI usage guidance

Use this as the canonical starting configuration; do not copy the built-in default literal into application code.

## Example

See [`defaults.py`](../../../examples/config/defaults.py).

## Tests

See [`test_creation.py`](../../../test/config/test_creation.py).

## Complete executable example

```python
from klygo import config

settings = config.defaults({"model": {"batch": 32}})
assert settings["model"]["batch"] == 32
```
