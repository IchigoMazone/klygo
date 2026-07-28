from pathlib import Path
from typing import Union, Dict, Any

from klygo.archive.backend import get_backend
from klygo.validators.archive import GetInfo, Test


def get_info(archive_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Tác dụng:
    - Lấy thông tin thống kê chi tiết và metadata của file lưu trữ (định dạng, thuật toán nén, mã hóa, số lượng file/thư mục, tỷ lệ nén, kích thước dễ đọc).

    Đầu vào:
    - archive_path [str | Path]: Đường dẫn file lưu trữ.

    Đầu ra:
    - [Dict[str, Any]] Từ điển chứa các trường thông tin chi tiết:
      + 'path': Đường dẫn file archive
      + 'format': Định dạng nén ('zip', 'tar.gz', '7z', v.v.)
      + 'compression_algorithm': Thuật toán nén sử dụng
      + 'encrypted': Trạng thái mã hóa mật khẩu (True/False)
      + 'file_count': Số lượng file bên trong
      + 'directory_count': Số lượng thư mục bên trong
      + 'uncompressed_size': Dung lượng khi giải nén (bytes)
      + 'human_uncompressed_size': Dung lượng nén đọc dễ nhìn (KB/MB/GB)
      + 'compressed_size': Dung lượng đã nén (bytes)
      + 'human_compressed_size': Dung lượng đã nén đọc dễ nhìn
      + 'compress_ratio': Tỷ lệ nén (%)
      + 'archive_size': Dung lượng file archive trên ổ đĩa (bytes)
      + 'human_archive_size': Dung lượng archive đọc dễ nhìn
      + 'largest_file': Tên file có kích thước lớn nhất
      + 'smallest_file': Tên file có kích thước nhỏ nhất

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Lấy từ điển metadata chi tiết của dataset.zip
    >>> info = ar.get_info("dataset.zip")
    >>> print(info['format'])
    'zip'
    >>> print(info['human_uncompressed_size'])
    '105.42 MB'
    >>> print(info['file_count'])
    723

    Nguồn: TrinhNhuNhat_28072026.
    """
    path = Path(archive_path)
    backend = get_backend(path)
    return backend.get_info(path)


def test(archive_path: Union[str, Path], raise_exception: bool = False) -> bool:
    """
    Tác dụng:
    - Kiểm tra tính toàn vẹn dữ liệu và mã kiểm tra lỗi (CRC/Header) của file lưu trữ.

    Đầu vào:
    - archive_path [str | Path]: Đường dẫn file lưu trữ.
    - raise_exception [bool]: Ném ra ngoại lệ ValueError khi phát hiện file hỏng thay vì trả về False. Mặc định: False.

    Đầu ra:
    - [bool] Trả về True nếu archive hợp lệ hoàn toàn; trả về False nếu archive bị hỏng.

    Ngoại lệ:
    - ValueError: Phát sinh khi archive bị hỏng và raise_exception=True.

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Kiểm tra tính toàn vẹn của file nén
    >>> is_valid = ar.test("data.zip")
    >>> print(is_valid)
    True

    # Ví dụ 2: Kiểm tra với tùy chọn ném ra ngoại lệ khi file bị hỏng
    >>> is_valid = ar.test("corrupted.zip", raise_exception=False)
    >>> print(is_valid)
    False

    Nguồn: TrinhNhuNhat_28072026.
    """
    path = Path(archive_path)
    backend = get_backend(path)
    return backend.test(path, raise_exception=raise_exception)


def verify(archive_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Tác dụng:
    - Kiểm tra và xác minh toàn diện trạng thái file lưu trữ (Header, CRC, Metadata, trạng thái mã hóa).

    Đầu vào:
    - archive_path [str | Path]: Đường dẫn file lưu trữ.

    Đầu ra:
    - [Dict[str, Any]] Từ điển chứa thông tin xác minh ('valid', 'format', 'file_count', 'archive_size', 'human_archive_size', 'encrypted').

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Xác minh file backup.tar.gz
    >>> report = ar.verify("backup.tar.gz")
    >>> print(report)
    {
        'valid': True,
        'format': 'tar.gz',
        'file_count': 45,
        'archive_size': 10485760,
        'human_archive_size': '10.00 MB',
        'encrypted': False
    }

    Nguồn: TrinhNhuNhat_28072026.
    """
    path = Path(archive_path)
    backend = get_backend(path)
    is_valid = backend.test(path, raise_exception=False)
    info = backend.get_info(path)

    return {
        "valid": is_valid,
        "format": info.get("format"),
        "file_count": info.get("file_count", 0),
        "archive_size": info.get("archive_size", 0),
        "human_archive_size": info.get("human_archive_size", ""),
        "encrypted": info.get("encrypted", False),
    }


# Ngăn pytest thu thập nhầm API này khi được import vào module test.
test.__test__ = False
