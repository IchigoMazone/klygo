from zipfile import ZipFile
from pathlib import Path

from klygo.validators.archive import GetInfo, Test


def get_info(
    archive_path: str | Path,
) -> dict:
    """
    Tác dụng:
    - Lấy thông tin thống kê của dữ liệu

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

    params = GetInfo(archive_path=archive_path)

    with ZipFile(params.archive_path, mode="r") as zf:
        members = zf.infolist()
        total_uncompressed = sum(m.file_size for m in members)
        total_compressed = sum(m.compress_size for m in members)

    ratio = (
        round((1 - total_compressed / total_uncompressed) * 100, 2)
        if total_uncompressed > 0
        else 0.0
    )

    return {
        "path": str(params.archive_path),
        "format": params.archive_path.suffix.lstrip("."),
        "file_count": len(members),
        "uncompressed_size": total_uncompressed,
        "compressed_size": total_compressed,
        "compress_ratio": ratio,
        "archive_size": params.archive_path.stat().st_size,
    }


def test(
    archive_path: str | Path,
) -> bool:
    """
    Tác dụng:
    - Kiểm tra tính toàn vẹn của file lưu trữ

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

    params = Test(archive_path=archive_path)

    with ZipFile(params.archive_path, mode="r") as zf:
        bad_file = zf.testzip()

    if bad_file is not None:
        raise ValueError(
            f"archive is corrupted. First bad file: '{bad_file}'"
        )

    return True


# Ngăn pytest thu thập nhầm API này khi được import vào module test.
test.__test__ = False
