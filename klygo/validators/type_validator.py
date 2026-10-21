
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


__all__ = ["validate_type"]
