
def validate_type(
    value: object,
    expected: type | tuple[type, ...],
    name: str,
) -> None:
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