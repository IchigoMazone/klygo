# `config.diff`

```python
config.diff(config1, config2)
```

Compare two configuration objects or files by flattened leaf path.

## Contract

Reports keys added to the second configuration, removed from it, and present with changed values. Modified entries contain `from` and `to`.

## Parameters

- `config1`: first mapping, `Box`, or file path.
- `config2`: second mapping, `Box`, or file path.

## Returns

`dict` with `added`, `removed`, and `modified` dictionaries.

## Errors and edge cases

File errors propagate from `load`; non-mapping object inputs raise `TypeError`. Nested mappings are compared at leaf granularity.

## AI usage guidance

Use the structured report rather than manually comparing serialized text, which is sensitive to ordering and formatting.

## Example

See [`diff.py`](../../../examples/config/diff.py).

## Tests

See [`test_inspection.py`](../../../test/config/test_inspection.py).

## Complete executable example

```python
from klygo import config

changes = config.diff({"batch": 16}, {"batch": 32, "seed": 7})
assert changes["added"] == {"seed": 7}
assert changes["modified"]["batch"] == {"from": 16, "to": 32}
```
