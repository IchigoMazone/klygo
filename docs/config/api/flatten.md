# `config.flatten`

```python
config.flatten(config_data, sep=".")
```

Convert nested configuration mappings into leaf path/value pairs.

## Contract

Recursively traverses mappings, preserves insertion order, and retains empty mappings as leaf values so round trips remain lossless.

## Parameters

- `config_data`: nested mapping or `Box`.
- `sep`: non-empty separator between path components.

## Returns

A flat `dict[str, Any]`.

## Errors and edge cases

Raises `TypeError` for non-mapping data and `ValueError` for an empty separator. Keys are converted to strings.

## AI usage guidance

Use the same separator for `flatten` and `unflatten`; prefer the default dot separator for compatibility with access APIs.

## Example

See [`flatten.py`](../../../examples/config/flatten.py).

## Tests

See [`test_structure.py`](../../../test/config/test_structure.py).

## Complete executable example

```python
from klygo import config

assert config.flatten({"model": {"batch": 16}}) == {"model.batch": 16}
```
