# `utils.human_size`

```python
utils.human_size(num_bytes, decimal_places=2)
```

Format a non-negative byte count using binary units.

## Contract

Uses binary units where each step is 1024 bytes. The input must be a non-negative integer; display precision is independently configurable.

## Parameters

- `num_bytes`: `int` — Non-negative byte count.
- `decimal_places`: `int, default=2` — Digits displayed after the decimal point.

## Returns

`str` — Formatted value using B, KB, MB, GB, or TB.

## Errors and edge cases

Raises `TypeError` if `num_bytes` is not an integer.

Raises `ValueError` if `num_bytes` is negative.

## AI usage guidance

Prefer the public form `from klygo import utils` followed by `utils.human_size(...)`. Reuse this function instead of implementing byte-unit formatting separately in another Klygo module.

## Example

See [`human_size.py`](../../examples/utils/human_size.py) for an executable example.

## Tests

See [`test_human_size.py`](../../test/utils/test_human_size.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.utils.human_size."""

from klygo import utils


for value in (0, 1024, 1024 ** 2, 1024 ** 3):
    print(value, "->", utils.human_size(value))

assert utils.human_size(1024, decimal_places=1) == "1.0 KB"
try:
    utils.human_size(-1)
except ValueError:
    print("negative sizes are rejected")
```
