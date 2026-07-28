from pathlib import Path
from typing import Union, List, Iterator

from klygo.archive.backend import get_backend
from klygo.validators.archive import ListFiles, Search


def list_files(archive_path: Union[str, Path]) -> List[str]:
    """
    Tác dụng:
    - Lấy danh sách tất cả các đường dẫn file bên trong file lưu trữ.

    Đầu vào:
    - archive_path [str | Path]: Đường dẫn file lưu trữ (ZIP, TAR, 7Z, v.v.).

    Đầu ra:
    - [List[str]] Danh sách tên/đường dẫn của tất cả các file trong archive.

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi archive_path không tồn tại.

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Lấy danh sách tất cả các file trong dataset.zip
    >>> files = ar.list_files("dataset.zip")
    >>> print(files[:3])
    ['data.yaml', 'images/001.jpg', 'images/002.jpg']

    # Ví dụ 2: Lấy số lượng file trong file nén TAR.GZ
    >>> files_tar = ar.list_files("backup.tar.gz")
    >>> print(len(files_tar))
    450

    Nguồn: TrinhNhuNhat_28072026.
    """
    path = Path(archive_path)
    backend = get_backend(path)
    return backend.list_files(path)


def iter_files(archive_path: Union[str, Path]) -> Iterator[str]:
    """
    Tác dụng:
    - Duyệt danh sách đường dẫn file dưới dạng Generator để tiết kiệm bộ nhớ RAM với các archive chứa hàng triệu file.

    Đầu vào:
    - archive_path [str | Path]: Đường dẫn file lưu trữ.

    Đầu ra:
    - [Iterator[str]] Generator phát ra từng đường dẫn file một.

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Duyệt từng file trong archive lớn mà không tạo list lớn trong RAM
    >>> for file_name in ar.iter_files("large_dataset.tar.gz"):
    ...     if file_name.endswith(".png"):
    ...         print("Found PNG:", file_name)
    Found PNG: images/001.png
    Found PNG: images/002.png

    Nguồn: TrinhNhuNhat_28072026.
    """
    path = Path(archive_path)
    backend = get_backend(path)
    yield from backend.iter_files(path)


def search(
    archive_path: Union[str, Path],
    pattern: str,
    regex: bool = False,
    case_sensitive: bool = True,
) -> List[str]:
    """
    Tác dụng:
    - Tìm kiếm các đường dẫn file phù hợp với mẫu wildcard hoặc biểu thức chính quy (Regex) trong file lưu trữ.

    Đầu vào:
    - archive_path [str | Path]: Đường dẫn file lưu trữ.
    - pattern [str]: Mẫu tìm kiếm (wildcard glob như 'images/*.jpg' hoặc chuỗi Regex).
    - regex [bool]: Bật chế độ tìm kiếm bằng Regular Expression. Mặc định: False.
    - case_sensitive [bool]: Phân biệt chữ hoa/chữ thường khi tìm kiếm. Mặc định: True.

    Đầu ra:
    - [List[str]] Danh sách các đường dẫn file khớp với mẫu tìm kiếm.

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Tìm tất cả các file ảnh PNG trong ZIP
    >>> pngs = ar.search("dataset.zip", "*.png")
    >>> print(pngs)
    ['images/001.png', 'images/002.png']

    # Ví dụ 2: Tìm file theo biểu thức chính quy Regex
    >>> matches = ar.search("data.zip", r"frame_\\d+\\.jpg", regex=True)
    >>> print(matches)
    ['images/frame_001.jpg', 'images/frame_002.jpg']

    Nguồn: TrinhNhuNhat_28072026.
    """
    path = Path(archive_path)
    backend = get_backend(path)
    return backend.search(
        archive_path=path,
        pattern=pattern,
        regex=regex,
        case_sensitive=case_sensitive,
    )
