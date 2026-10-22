# `utils.human_size`

```python
utils.human_size(
    num_bytes: int,
    decimal_places: int = 2,
) -> str
```

Format a non-negative byte count using binary units (`B`, `KB`, `MB`, `GB`, or `TB`).

## Contract

Converts an integer byte count `num_bytes` into a human-readable string scaled by powers of `1024` (`B` -> `KB` -> `MB` -> `GB` -> `TB`). At each step, if the scaled value is strictly less than `1024`, it is formatted with `decimal_places` digits after the decimal point followed by a space and the unit symbol (for example, `'1.00 MB'`). Values at or above `1024 ** 4` bytes are formatted in `TB`.

## Parameters

- `num_bytes`: `int` — non-negative byte count to format.
- `decimal_places`: `int`, default `2` — number of digits displayed after the decimal point.

## Returns

`str` — formatted byte size string using `B`, `KB`, `MB`, `GB`, or `TB` (such as `'0.00 B'`, `'1.00 KB'`, or `'2.50 GB'`).

## Errors and edge cases

- Raises `TypeError` (`f"num_bytes must be int, got {type(num_bytes).__name__}"`) if `num_bytes` is not an `int` (for example, `float`, `str`, or `None`).
- Raises `ValueError` (`f"num_bytes must be non-negative, got {num_bytes}"`) if `num_bytes < 0`.
- Raises `ValueError` from Python's string formatting if `decimal_places` is negative or not a valid format precision specifier.

## AI usage guidance

Prefer the public form `from klygo import utils` followed by `utils.human_size(...)`. Reuse this function instead of implementing separate byte-unit formatting logic in other Klygo modules.

## Example

See [`human_size.py`](../../../examples/utils/human_size.py) for an executable example.

## Tests

See [`test_human_size.py`](../../../test/utils/test_human_size.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.utils.human_size."""

from klygo import utils

for value in (0, 1024, 1024 ** 2, 1024 ** 3):
    print(value, "->", utils.human_size(value))

assert utils.human_size(0) == "0.00 B"
assert utils.human_size(1024, decimal_places=1) == "1.0 KB"
assert utils.human_size(1024 ** 2) == "1.00 MB"

try:
    utils.human_size(-1)
except ValueError:
    print("negative sizes are rejected")
```
