"""Formatting utilities shared across Klygo modules."""


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
