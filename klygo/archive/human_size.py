def human_size(num_bytes: int, decimal_places: int = 2) -> str:
    """
    Tác dụng:
    - Chuyển số byte thành chuỗi kích thước dễ đọc.

    Đầu vào:
    - num_bytes [int]: Kích thước dữ liệu theo byte.
    - decimal_places [int]: Số chữ số thập phân hiển thị. Mặc định: 2.

    Đầu ra:
    - [str] Chuỗi kích thước theo đơn vị B, KB, MB, GB hoặc TB.

    Ngoại lệ:
    - TypeError: num_bytes không phải số nguyên.
    - ValueError: num_bytes là số âm.

    Ví dụ:
    >>> from klygo.archive import human_size

    # Ví dụ 1: Đổi số byte sang KB
    >>> print(human_size(1024))
    1.00 KB

    # Ví dụ 2: Đổi số byte sang MB
    >>> print(human_size(1048576000))
    999.88 MB

    # Ví dụ 3: Đổi số byte với 4 chữ số thập phân
    >>> print(human_size(1048576000, decimal_places=4))
    999.8779 MB

    Nguồn: TrinhNhuNhat_28072026.
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
