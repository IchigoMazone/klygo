from zipfile import ZipFile
from pathlib import Path
import fnmatch

from klygo.validators.archive import ListFiles, Search


def list_files(
    archive_path: str | Path,
) -> list[str]:
    """
    Tác dụng:
    - Lấy danh sách file bên trong file lưu trữ

    Đầu vào:
    - archive_path: Đường dẫn file lưu trữ

    Đầu ra:
    - Kết quả xử lý của hàm

    Ngoại lệ:
    - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
    """

    params = ListFiles(archive_path=archive_path)

    with ZipFile(params.archive_path, mode="r") as zf:
        names = zf.namelist()

    return names


def search(
    archive_path: str | Path,
    pattern: str,
) -> list[str]:
    """
    Tác dụng:
    - Tìm các file khớp mẫu bên trong file lưu trữ

    Đầu vào:
    - archive_path: Đường dẫn file lưu trữ
    - pattern: Tham số pattern của hàm

    Đầu ra:
    - Kết quả xử lý của hàm

    Ngoại lệ:
    - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
    """

    params = Search(archive_path=archive_path, pattern=pattern)

    with ZipFile(params.archive_path, mode="r") as zf:
        names = zf.namelist()

    return [name for name in names if fnmatch.fnmatch(name, params.pattern)]
