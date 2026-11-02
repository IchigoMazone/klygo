# `utils.exclude_keys`

```python
utils.exclude_keys(
    values: Mapping[str, Any],
    keys: str | list[str],
) -> dict[str, Any]
```

Return a shallow dictionary copy without the selected key or keys.

## Contract

Accepts any `collections.abc.Mapping` (`dict`, `Box`, read-only mapping proxies, etc.) and constructs a new plain `dict` containing every `(key, value)` pair from `values` whose key is not listed in `keys`. The input mapping `values` is never modified in place, and keys specified in `keys` that do not exist in `values` are ignored without error.

`keys` must be either a single `str` (excluding one key) or a `list[str]` (excluding zero or more keys). Other iterable containers such as tuples or sets, or lists containing non-string elements, are rejected with a `TypeError`.

## Parameters

- `values`: `Mapping[str, Any]` — source mapping to copy.
- `keys`: `str | list[str]` — single key name or list of key names to omit from the returned dictionary.

## Returns

`dict[str, Any]` — a new shallow dictionary copy containing all key-value pairs from `values` except those matching `keys`.

## Errors and edge cases

- Raises `TypeError("values must be a mapping")` if `values` is not an instance of `collections.abc.Mapping`.
- Raises `TypeError("keys must be a string or a list of strings")` if `keys` is neither a `str` nor a `list` whose elements are all `str` instances.
- Passing an empty list `[]` returns a shallow copy of all items in `values`.

## AI usage guidance

Prefer `from klygo import utils` followed by `utils.exclude_keys(...)` when stripping runtime-only keys from configuration or metadata dictionaries before forwarding keyword arguments.

## Example

See [`exclude_keys.py`](../../../examples/utils/exclude_keys.py) for an executable example.

## Tests

See [`test_exclude_keys.py`](../../../test/utils/test_exclude_keys.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.utils.exclude_keys."""

from klygo import utils

settings = {"threshold": 0.4, "dtype": "float32", "device": "cpu"}
model_settings = utils.exclude_keys(settings, ["dtype", "device"])
single_excluded = utils.exclude_keys(settings, "device")

print("Filtered:", model_settings)
print("Original:", settings)

assert model_settings == {"threshold": 0.4}
assert single_excluded == {"threshold": 0.4, "dtype": "float32"}
assert settings == {"threshold": 0.4, "dtype": "float32", "device": "cpu"}
```
