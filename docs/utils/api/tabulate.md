# `utils.tabulate`

```python
utils.tabulate(
    data: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    title: str | None = None,
    key_header: str = "Property",
    value_header: str = "Value",
    show: bool = True,
    show_header: bool = True,
    show_lines: bool = True,
    padding: tuple[int, int] | int = (0, 2),
    box_style: rich.box.Box | None = None,
    border_style: str = "none",
    header_style: str = "none",
    title_style: str = "none",
    **table_kwargs: Any,
) -> rich.table.Table | None
```

Render a mapping or sequence of row mappings as a Rich table.

## Contract

Accepts either a single `Mapping[str, Any]` (rendered as two `Property` / `Value` columns, automatically flattening nested mappings into dotted keys such as `config.model.layers`) or a non-empty `Sequence[Mapping[str, Any]]` (rendered as a multi-column table with one column per unique key, left-aligning text columns and right-aligning numeric columns). By default it uses `box.HORIZONTALS`, `show_lines=True`, `padding=(0, 2)`, and theme-adaptive uniform styling (`"none"`), while allowing callers to customize `show_header`, `show_lines`, `padding`, `box_style`, `border_style`, `header_style`, `title_style`, and any extra `rich.table.Table` keyword arguments via `**table_kwargs`. `None`, empty mappings, and empty sequences/sets are displayed as `_`.

## Parameters

- `data`: `Mapping[str, Any] | Sequence[Mapping[str, Any]]` — key-value mapping or non-empty sequence of row mappings.
- `title`: `str | None`, default `None` — optional title displayed above the table.
- `key_header`: `str`, default `'Property'` — header label for the key column when `data` is a single mapping.
- `value_header`: `str`, default `'Value'` — header label for the value column when `data` is a single mapping.
- `show`: `bool`, default `True` — when `True`, print the table to the Rich console and return `None`; when `False`, return the constructed `rich.table.Table` instance without printing.
- `show_header`: `bool`, default `True` — whether to render the column header row.
- `show_lines`: `bool`, default `True` — whether to draw horizontal divider lines between rows.
- `padding`: `tuple[int, int] | int`, default `(0, 2)` — cell padding `(vertical, horizontal)` or a single integer.
- `box_style`: `rich.box.Box | None`, default `None` — Rich box drawing style; defaults to `rich.box.HORIZONTALS`.
- `border_style`: `str`, default `'none'` — Rich style applied to borders and dividers.
- `header_style`: `str`, default `'none'` — Rich style applied to column headers.
- `title_style`: `str`, default `'none'` — Rich style applied to the table title.
- `**table_kwargs`: `Any` — additional keyword arguments forwarded to `rich.table.Table` (for example `width`, `expand`, or `caption`).

## Returns

`rich.table.Table | None` — `None` when `show=True`, or the constructed `rich.table.Table` instance when `show=False`.

## Errors and edge cases

- Raises `TypeError` if `data` is neither a `Mapping` nor a `Sequence` of `Mapping` instances.
- Raises `ValueError` if `data` is an empty sequence (`[]`).
- Raises `TypeError` if `title`, `key_header`, `value_header`, `border_style`, `header_style`, `title_style`, `show`, `show_header`, `show_lines`, `padding`, or `box_style` have invalid types.

## AI usage guidance

Prefer the public form `from klygo import utils` followed by `utils.tabulate(...)`. Use `**table_kwargs` or the explicit style/layout parameters when a caller needs custom borders, widths, or captions.

## Example

See [`tabulate.py`](../../../examples/utils/tabulate.py) for an executable example.

## Tests

See [`test_tabulate.py`](../../../test/utils/test_tabulate.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable examples for klygo.utils.tabulate."""

from rich import box

from klygo import utils

summary = {
    "dataset": "coco15",
    "classes": ["person", "bicycle", "car"],
    "splits": {"train": 10, "val": 5},
    "notes": None,
}
utils.tabulate(summary, title="Dataset Summary")

rows = [
    {"Dessert (100g serving)": "Frozen yogurt", "Calories": 159, "Fat (g)": 6.0, "Carbs (g)": 24},
    {"Dessert (100g serving)": "Ice cream sandwich", "Calories": 237, "Fat (g)": 9.0, "Carbs (g)": 37},
    {"Dessert (100g serving)": "Eclair", "Calories": 262, "Fat (g)": 16.0, "Carbs (g)": 24},
]
table = utils.tabulate(
    rows,
    box_style=box.ROUNDED,
    show_lines=False,
    padding=(0, 1),
    caption="Nutrition facts",
    show=False,
)
assert len(table.columns) == 4
assert table.caption == "Nutrition facts"

try:
    utils.tabulate("not-a-mapping")  # type: ignore[arg-type]
except TypeError:
    print("non-mapping inputs are rejected")
```
