def human_size(num_bytes: int) -> str:
    """
    Tác dụng:
    - Chuyển số byte thành chuỗi kích thước dễ đọc

    Đầu vào:
    - num_bytes: Kích thước dữ liệu theo byte

    Đầu ra:
    - Chuỗi kích thước theo đơn vị B, KB, MB, GB hoặc TB

    Ngoại lệ:
    - TypeError: num_bytes không phải số nguyên
    - ValueError: num_bytes là số âm

    Nguồn: TrinhNhuNhat_12072026.
    """
    if not isinstance(num_bytes, int):
        raise TypeError(f"num_bytes must be int, got {type(num_bytes).__name__}")
    if num_bytes < 0:
        raise ValueError(f"num_bytes must be non-negative, got {num_bytes}")

    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"
