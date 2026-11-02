
def validate_type(
    value: object,
    expected: type | tuple[type, ...],
    name: str,
) -> None:
    """Raise ``TypeError`` when an argument has an unexpected type.

    Parameters
    ----------
    value : object
        Argument to check.
    expected : type or tuple[type, ...]
        Accepted type or types, evaluated with ``isinstance``.
    name : str
        Argument name included in the error message.

    Returns
    -------
    None
        Successful validation has no return value.

    Raises
    ------
    TypeError
        If ``value`` is not an instance of any accepted type.

    Examples
    --------
    >>> from klygo import validators
    >>> validators.validate_type("image.jpg", (str, bytes), "source") is None
    True
    """
    if not isinstance(value, expected):
        if isinstance(expected, tuple):
            expected_name = ", ".join(
                t.__name__ for t in expected
            )
        else:
            expected_name = expected.__name__

        raise TypeError(
            f"{name} must be {expected_name}, "
            f"got {type(value).__name__}"
        )


def validate_count(
    value: object,
    name: str,
    minimum: int = 1,
) -> None:
    """Raise ``TypeError`` or ``ValueError`` when an argument is not a valid count.

    A valid count must be an integer (excluding boolean values) greater than or
    equal to ``minimum``.

    Parameters
    ----------
    value : object
        Argument to check.
    name : str
        Argument name included in the error message.
    minimum : int, default 1
        Minimum acceptable integer value.

    Returns
    -------
    None
        Successful validation has no return value.

    Raises
    ------
    TypeError
        If ``value`` is not an integer or is a boolean.
    ValueError
        If ``value`` is less than ``minimum``.

    Examples
    --------
    >>> from klygo import validators
    >>> validators.validate_count(10, "batch") is None
    True
    >>> validators.validate_count(0, "warmup", minimum=0) is None
    True
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


__all__ = ["validate_type", "validate_count"]
