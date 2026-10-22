"""Formatting and mapping utilities shared across Klygo modules."""

from typing import Any, Mapping


def exclude_keys(
    values: Mapping[str, Any],
    keys: str | list[str],
) -> dict[str, Any]:
    """Return a shallow copy without the selected keys.

    ``keys`` accepts one string or a list of strings. The input mapping is
    never modified.
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


__all__ = ["exclude_keys", "invert", "human_size"]
