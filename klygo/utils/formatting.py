"""Formatting and mapping utilities shared across Klygo modules."""

from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from rich.table import Table


def exclude_keys(
    values: Mapping[str, Any],
    keys: str | list[str],
) -> dict[str, Any]:
    """Return a shallow copy without the selected keys.

    ``keys`` accepts one string or a list of strings. The input mapping is
    never modified.

    Raises
    ------
    TypeError
        If ``values`` is not a mapping or ``keys`` is not a string or list of strings.
    """
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping")
    if isinstance(keys, str):
        excluded = {keys}
    elif isinstance(keys, list) and all(isinstance(key, str) for key in keys):
        excluded = set(keys)
    else:
        raise TypeError("keys must be a string or a list of strings")
    return {key: value for key, value in values.items() if key not in excluded}


def invert(values: Mapping[Any, Any]) -> dict[Any, Any]:
    """Return a new dictionary with every key and value exchanged.

    Values must be unique and hashable so the operation remains reversible.
    The input mapping is never modified.

    Raises
    ------
    TypeError
        If ``values`` is not a mapping or contains unhashable values.
    ValueError
        If any two keys map to the same value.
    """
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping")

    swapped = {}
    for key, value in values.items():
        try:
            duplicate = value in swapped
        except TypeError as exc:
            raise TypeError("mapping values must be hashable") from exc
        if duplicate:
            raise ValueError(f"mapping values must be unique; duplicate value: {value!r}")
        swapped[value] = key
    return swapped


def human_size(num_bytes: int, decimal_places: int = 2) -> str:
    """Format a non-negative byte count using binary units.

    Parameters
    ----------
    num_bytes : int
        Non-negative byte count.
    decimal_places : int, default=2
        Digits displayed after the decimal point.

    Returns
    -------
    str
        Formatted value using B, KB, MB, GB, or TB.

    Raises
    ------
    TypeError
        If ``num_bytes`` is not an integer.
    ValueError
        If ``num_bytes`` is negative.

    Examples
    --------
    >>> from klygo import utils
    >>> utils.human_size(1048576)
    '1.00 MB'
    """
    if not isinstance(num_bytes, int):
        raise TypeError(f"num_bytes must be int, got {type(num_bytes).__name__}")
    if num_bytes < 0:
        raise ValueError(f"num_bytes must be non-negative, got {num_bytes}")

    size = float(num_bytes)
    fmt = f"{{:.{decimal_places}f}}"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{fmt.format(size)} {unit}"
        size /= 1024
    return f"{fmt.format(size)} TB"


def _flatten_mapping(
    mapping: Mapping[str, Any],
    prefix: str = "",
) -> list[tuple[str, Any]]:
    """Flatten nested mappings into ``(dotted_key, value)`` pairs."""
    items: list[tuple[str, Any]] = []
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            if not value:
                items.append((full_key, None))
            else:
                items.extend(_flatten_mapping(value, full_key))
        else:
            items.append((full_key, value))
    return items


def _format_table_value(value: Any) -> Any:
    """Format a scalar, sequence, or mapping value into a clean Rich Text cell."""
    from rich.text import Text

    if value is None:
        return Text("_")
    if isinstance(value, Mapping):
        if not value:
            return Text("_")
        flat = _flatten_mapping(value)
        return Text(", ".join(f"{k}: {_format_table_value(v).plain}" for k, v in flat))
    if isinstance(value, (list, tuple, set)):
        if not value:
            return Text("_")
        return Text(", ".join(str(item) for item in value))
    return Text(str(value))


def _is_numeric_column(rows: Sequence[Mapping[str, Any]], column: str) -> bool:
    """Return True when all non-None values in ``column`` are numbers (excluding bool)."""
    seen = False
    for row in rows:
        val = row.get(column)
        if val is None:
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return False
        seen = True
    return seen


def tabulate(
    data: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    title: str | None = None,
    key_header: str = "Property",
    value_header: str = "Value",
    show: bool = True,
    show_header: bool = True,
    show_lines: bool = True,
    padding: tuple[int, int] | int = (0, 2),
    box_style: Any = None,
    border_style: str = "none",
    header_style: str = "none",
    title_style: str = "none",
    **table_kwargs: Any,
) -> "Table | None":
    """Render a mapping or sequence of row mappings as a Rich table.

    Parameters
    ----------
    data : Mapping[str, Any] or Sequence[Mapping[str, Any]]
        Either a single key-value mapping (rendered as two columns) or a sequence
        of row mappings (rendered as a multi-column table with one column per key).
    title : str or None, default=None
        Optional table title displayed above the header.
    key_header : str, default='Property'
        Header label for the key column when ``data`` is a single mapping.
    value_header : str, default='Value'
        Header label for the value column when ``data`` is a single mapping.
    show : bool, default=True
        When ``True``, print the table to the Rich console and return ``None``.
        When ``False``, return the constructed ``rich.table.Table`` instance.
    show_header : bool, default=True
        Whether to render the column header row.
    show_lines : bool, default=True
        Whether to draw horizontal divider lines between rows.
    padding : tuple[int, int] or int, default=(0, 2)
        Cell padding ``(vertical, horizontal)`` or a single integer.
    box_style : rich.box.Box or None, default=None
        Rich box drawing style. Defaults to ``rich.box.HORIZONTALS``.
    border_style : str, default='none'
        Rich style applied to table borders and dividers.
    header_style : str, default='none'
        Rich style applied to column headers.
    title_style : str, default='none'
        Rich style applied to the table title.
    **table_kwargs : Any
        Additional keyword arguments forwarded to ``rich.table.Table``.

    Returns
    -------
    rich.table.Table or None
        ``None`` when ``show=True``, otherwise the constructed Rich ``Table``.

    Raises
    ------
    TypeError
        If ``data`` is neither a mapping nor a sequence of mappings, or if
        formatting arguments have invalid types.
    ValueError
        If ``data`` is an empty sequence.

    Examples
    --------
    >>> from klygo import utils
    >>> table = utils.tabulate({"classes": ["apple", "banana"], "nc": 2}, title="Dataset", show=False)
    """
    is_mapping = isinstance(data, Mapping)
    is_row_seq = (
        not is_mapping
        and isinstance(data, Sequence)
        and not isinstance(data, (str, bytes, bytearray))
        and all(isinstance(row, Mapping) for row in data)
    )
    if not is_mapping and not is_row_seq:
        raise TypeError(
            f"data must be a mapping or sequence of mappings, got {type(data).__name__}"
        )
    if is_row_seq and len(data) == 0:
        raise ValueError("data sequence must not be empty")
    if title is not None and not isinstance(title, str):
        raise TypeError(f"title must be str or None, got {type(title).__name__}")
    for name, val in (
        ("key_header", key_header),
        ("value_header", value_header),
        ("border_style", border_style),
        ("header_style", header_style),
        ("title_style", title_style),
    ):
        if not isinstance(val, str):
            raise TypeError(f"{name} must be str, got {type(val).__name__}")
    for name, val in (
        ("show", show),
        ("show_header", show_header),
        ("show_lines", show_lines),
    ):
        if not isinstance(val, bool):
            raise TypeError(f"{name} must be bool, got {type(val).__name__}")
    if not (
        isinstance(padding, int)
        and not isinstance(padding, bool)
        or (
            isinstance(padding, tuple)
            and len(padding) == 2
            and all(isinstance(p, int) and not isinstance(p, bool) for p in padding)
        )
    ):
        raise TypeError("padding must be an int or a 2-tuple of ints")

    from rich import box
    from rich.console import Console
    from rich.table import Table

    active_box = box.HORIZONTALS if box_style is None else box_style
    if not isinstance(active_box, box.Box):
        raise TypeError(f"box_style must be a rich.box.Box or None, got {type(box_style).__name__}")

    table = Table(
        title=title,
        box=active_box,
        border_style=border_style,
        header_style=header_style,
        title_style=title_style,
        padding=padding,
        show_header=show_header,
        show_lines=show_lines,
        **table_kwargs,
    )

    if is_mapping:
        table.add_column(key_header, no_wrap=True, justify="left")
        table.add_column(value_header, justify="left")
        for key, value in _flatten_mapping(data):
            table.add_row(key, _format_table_value(value))
    else:
        columns: list[str] = []
        for row in data:
            for col in row.keys():
                col_str = str(col)
                if col_str not in columns:
                    columns.append(col_str)
        for idx, col in enumerate(columns):
            is_num = _is_numeric_column(data, col)
            justify = "right" if is_num else "left"
            table.add_column(
                col,
                justify=justify,
                no_wrap=is_num or (idx == 0),
                overflow="fold",
            )
        for row in data:
            table.add_row(*(_format_table_value(row.get(col)) for col in columns))

    if is_mapping:
        needed_w = max(
            len(key_header) + len(value_header) + 20,
            max((len(str(k)) + len(str(v)) + 12 for k, v in _flatten_mapping(data)), default=60),
        )
    else:
        col_widths = {}
        for col in columns:
            max_col_w = len(str(col))
            for row in data:
                val = row.get(col)
                val_len = len(str(val)) if val is not None else 1
                if val_len > max_col_w:
                    max_col_w = val_len
            col_widths[col] = max_col_w
        pad_h = padding if isinstance(padding, int) else padding[1]
        needed_w = sum(col_widths.values()) + len(columns) * (pad_h * 2 + 1) + 6

    if show:
        active_console = Console()
        console_width = max(active_console.width or 80, needed_w)
        Console(width=console_width).print(table)
        return None
    return table


__all__ = ["exclude_keys", "invert", "human_size", "tabulate"]
