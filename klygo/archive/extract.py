from pathlib import Path
from typing import Optional, Union, List

from klygo.archive.backend import get_backend
from klygo.validators.archive import Extract, ExtractFile


def extract(
    archive_path: Union[str, Path],
    output_dir: Union[str, Path] = ".",
    overwrite: bool = False,
    verbose: bool = True,
    password: Optional[str] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
    preserve_timestamp: bool = True,
    preserve_permissions: bool = True,
) -> None:
    """
    Tác dụng:
    - Giải nén toàn bộ hoặc các file chỉ định từ file lưu trữ vào thư mục đích.
    - Hỗ trợ chống lỗ hổng Zip-Slip (Path Traversal), giải nén mật khẩu và lọc theo wildcard.

    Đầu vào:
    - archive_path [str | Path]: Đường dẫn file lưu trữ cần giải nén.
    - output_dir [str | Path]: Thư mục đích nhận các file giải nén. Mặc định: ".".
    - overwrite [bool]: Trạng thái cho phép ghi đè nếu file trong output_dir đã tồn tại. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan trong console. Mặc định: True.
    - password [str | None]: Mật khẩu giải nén đối với archive mã hóa. Mặc định: None.
    - include [str | list[str] | None]: Mẫu wildcard chỉ định giải nén các file phù hợp (ví dụ: '*.png').
    - exclude [str | list[str] | None]: Mẫu wildcard loại trừ các file không muốn giải nén.
    - preserve_timestamp [bool]: Giữ nguyên thời gian lưu trữ gốc của file. Mặc định: True.
    - preserve_permissions [bool]: Giữ nguyên quyền hạn POSIX của file. Mặc định: True.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi archive_path không tồn tại.
    - FileExistsError: Phát sinh khi các file đích đã tồn tại và overwrite=False.
    - ValueError: Phát sinh khi phát hiện đường dẫn Zip-Slip độc hại.

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Giải nén toàn bộ file ZIP vào thư mục ./extracted
    >>> ar.extract("dataset.zip", output_dir="./extracted", overwrite=True)
    # Kết quả hiển thị thanh tiến trình:
    # dataset.zip: extracting: 100%|##############################| 120/120 [00:00<00:00, 520file/s]

    # Ví dụ 2: Giải nén chỉ các file ảnh PNG với mật khẩu mở khóa
    >>> ar.extract("data.tar.gz", output_dir="./imgs", include="*.png", password="secret_pass")
    # Kết quả hiển thị thanh tiến trình:
    # data.tar.gz: extracting: 100%|##############################| 45/45 [00:00<00:00, 310file/s]

    # Ví dụ 3: Giải nén ngoại trừ các file tạm .log
    >>> ar.extract("backup.zip", output_dir="./restore", exclude="*.log", overwrite=True)

    Nguồn: TrinhNhuNhat_28072026.
    """
    path = Path(archive_path)
    out = Path(output_dir)

    backend = get_backend(path)
    backend.extract(
        archive_path=path,
        output_dir=out,
        password=password,
        include=include,
        exclude=exclude,
        preserve_timestamp=preserve_timestamp,
        preserve_permissions=preserve_permissions,
        overwrite=overwrite,
        verbose=verbose,
    )


def extract_file(
    archive_path: Union[str, Path],
    filename: str,
    output_dir: Union[str, Path] = ".",
    overwrite: bool = False,
    password: Optional[str] = None,
) -> None:
    """
    Tác dụng:
    - Giải nén một file đơn lẻ từ file lưu trữ bằng cơ chế Streaming I/O tiết kiệm RAM.

    Đầu vào:
    - archive_path [str | Path]: Đường dẫn file lưu trữ.
    - filename [str]: Tên đường dẫn file cụ thể bên trong archive (ví dụ: 'images/001.jpg').
    - output_dir [str | Path]: Thư mục đích lưu trữ file giải nén. Mặc định: ".".
    - overwrite [bool]: Cho phép ghi đè nếu file đã tồn tại tại đích. Mặc định: False.
    - password [str | None]: Mật khẩu giải nén nếu file bị mã hóa.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ngoại lệ:
    - KeyError: Phát sinh khi filename không tồn tại trong archive.
    - FileExistsError: Phát sinh khi file đã tồn tại tại output_dir và overwrite=False.

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Giải nén 1 file data.yaml từ dataset.zip ra thư mục hiện tại
    >>> ar.extract_file("dataset.zip", "data.yaml", output_dir=".", overwrite=True)

    # Ví dụ 2: Giải nén 1 file nhãn cụ thể ra thư mục ./labels
    >>> ar.extract_file("data.zip", "labels/train.csv", output_dir="./labels", overwrite=True)

    Nguồn: TrinhNhuNhat_28072026.
    """
    path = Path(archive_path)
    out = Path(output_dir)

    backend = get_backend(path)
    backend.extract_file(
        archive_path=path,
        filename=filename,
        output_dir=out,
        password=password,
        overwrite=overwrite,
    )


def extract_matching(
    archive_path: Union[str, Path],
    pattern: str,
    output_dir: Union[str, Path] = ".",
    overwrite: bool = False,
    password: Optional[str] = None,
) -> None:
    """
    Tác dụng:
    - Giải nén trực tiếp các file khớp mẫu pattern wildcard từ file lưu trữ.

    Đầu vào:
    - archive_path [str | Path]: Đường dẫn file lưu trữ.
    - pattern [str]: Mẫu wildcard cần khớp (ví dụ: '*.png', 'labels/*.txt').
    - output_dir [str | Path]: Thư mục đích nhận file giải nén. Mặc định: ".".
    - overwrite [bool]: Cho phép ghi đè file đã tồn tại. Mặc định: False.
    - password [str | None]: Mật khẩu nếu archive được bảo vệ.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Giải nén tất cả các file ảnh PNG ra thư mục ./images
    >>> ar.extract_matching("dataset.zip", "*.png", output_dir="./images", overwrite=True)

    # Ví dụ 2: Giải nén các file nhãn định dạng TXT
    >>> ar.extract_matching("dataset.zip", "labels/*.txt", output_dir="./labels", overwrite=True)

    Nguồn: TrinhNhuNhat_28072026.
    """
    extract(
        archive_path=archive_path,
        output_dir=output_dir,
        include=pattern,
        overwrite=overwrite,
        password=password,
        verbose=False,
    )
