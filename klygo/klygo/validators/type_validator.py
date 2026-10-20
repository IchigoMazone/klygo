
def validate_type(
    value: object,
    expected: type | tuple[type, ...],
    name: str,
) -> None:
    """
    Tác dụng:
    - Kiểm tra kiểu dữ liệu của tham số

    Đầu vào:
    - value: Tham số value của hàm
    - expected: Tham số expected của hàm
    - name: Tham số name của hàm

    Đầu ra:
    - Không trả về dữ liệu

    Ngoại lệ:
    - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
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
