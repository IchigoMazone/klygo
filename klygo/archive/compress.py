from pathlib import Path
from typing import Optional, Union

from klygo.archive.backend import get_backend
from klygo.validators.archive import Compress


def compress(
    source: Union[str, Path],
    output_path: Union[str, Path],
    format: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = True,
    compresslevel: int = 6,
    method: Optional[str] = None,
    preserve_timestamp: bool = True,
    preserve_permissions: bool = True,
    follow_symlinks: bool = False,
) -> None:
    """
    Tác dụng:
    - Nén file hoặc thư mục thành file lưu trữ (ZIP, TAR, TAR.GZ, TAR.XZ, 7Z, GZ).
    - Tự động nhận diện định dạng từ đuôi file đầu ra.

    Đầu vào:
    - source [str | Path]: Đường dẫn file hoặc thư mục nguồn cần nén.
    - output_path [str | Path]: Đường dẫn file nén đầu ra (ví dụ: 'data.zip', 'archive.tar.gz').
    - format [str | None]: Định dạng nén tùy chọn (nếu None sẽ tự động nhận diện từ đuôi output_path).
    - overwrite [bool]: Cho phép ghi đè nếu file đầu ra đã tồn tại. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan trong console. Mặc định: True.
    - compresslevel [int]: Mức độ nén từ 1 (nhanh nhất) đến 9 (nén cao nhất). Mặc định: 6.
    - method [str | None]: Thuật toán nén ('deflated', 'stored', 'bzip2', 'lzma'). Mặc định: None.
    - preserve_timestamp [bool]: Giữ nguyên thời gian chỉnh sửa (mtime) của file gốc. Mặc định: True.
    - preserve_permissions [bool]: Giữ nguyên quyền POSIX của file gốc. Mặc định: True.
    - follow_symlinks [bool]: Xử lý theo liên kết mềm symlink. Mặc định: False.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi source không tồn tại.
    - FileExistsError: Phát sinh khi output_path đã tồn tại và overwrite=False.
    - ValueError: Phát sinh khi định dạng hoặc tham số không hợp lệ.

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Nén thư mục dataset thành file ZIP chuẩn
    >>> ar.compress("dataset/", "dataset.zip", overwrite=True)
    # Kết quả hiển thị thanh tiến trình:
    # dataset.zip: compressing: 100%|##############################| 120/120 [00:00<00:00, 450file/s]

    # Ví dụ 2: Nén thư mục thành file TAR.GZ với mức nén tối đa (level 9)
    >>> ar.compress("models/", "models.tar.gz", compresslevel=9, overwrite=True)
    # Kết quả hiển thị thanh tiến trình:
    # models.tar.gz: compressing: 100%|##############################| 15/15 [00:00<00:00, 200file/s]

    # Ví dụ 3: Nén 1 file đơn lẻ thành file ZIP mà không hiển thị thanh tiến trình (verbose=False)
    >>> ar.compress("config.yaml", "config.zip", overwrite=True, verbose=False)

    Nguồn: TrinhNhuNhat_28072026.
    """
    output_path = Path(output_path)
    source_path = Path(source)

    if not source_path.exists():
        raise FileNotFoundError(f"source does not exist: {source_path}")

    backend = get_backend(output_path, format_hint=format)
    backend.compress(
        source=source_path,
        output_path=output_path,
        compresslevel=compresslevel,
        method=method,
        preserve_timestamp=preserve_timestamp,
        preserve_permissions=preserve_permissions,
        follow_symlinks=follow_symlinks,
        overwrite=overwrite,
        verbose=verbose,
    )
