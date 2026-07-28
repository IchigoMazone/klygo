from pathlib import Path
from typing import Union, List, Iterator, Dict, Any, Optional

from klygo.archive.backend import get_backend, detect_format


class ArchiveFile:
    """
    Tác dụng:
    - Đối tượng Wrapper hướng đối tượng (OOP) và Context Manager làm việc trực tiếp với file archive mà không phải mở/đóng lại file nhiều lần.

    Nguồn: TrinhNhuNhat_28072026.
    """

    def __init__(self, archive_path: Union[str, Path]):
        """
        Khởi tạo đối tượng ArchiveFile từ đường dẫn file archive.
        """
        self.archive_path = Path(archive_path)
        self.format = detect_format(self.archive_path)
        self.backend = get_backend(self.archive_path, format_hint=self.format)

    def list_files(self) -> List[str]:
        """Lấy danh sách tất cả các đường dẫn file trong archive."""
        return self.backend.list_files(self.archive_path)

    def iter_files(self) -> Iterator[str]:
        """Duyệt danh sách file dạng Generator tiết kiệm RAM."""
        yield from self.backend.iter_files(self.archive_path)

    def search(
        self,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
    ) -> List[str]:
        """Tìm kiếm file theo pattern wildcard hoặc Regex trong archive."""
        return self.backend.search(
            self.archive_path, pattern, regex=regex, case_sensitive=case_sensitive
        )

    def extract(
        self,
        output_dir: Union[str, Path] = ".",
        password: Optional[str] = None,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        """Giải nén toàn bộ archive vào thư mục đích."""
        self.backend.extract(
            self.archive_path,
            Path(output_dir),
            password=password,
            overwrite=overwrite,
            verbose=verbose,
        )

    def extract_file(
        self,
        filename: str,
        output_dir: Union[str, Path] = ".",
        password: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """Giải nén một file cụ thể từ archive bằng Streaming I/O."""
        self.backend.extract_file(
            self.archive_path,
            filename,
            Path(output_dir),
            password=password,
            overwrite=overwrite,
        )

    def get_info(self) -> Dict[str, Any]:
        """Lấy thông tin chi tiết metadata và thống kê archive."""
        return self.backend.get_info(self.archive_path)

    def test(self, raise_exception: bool = False) -> bool:
        """Kiểm tra tính toàn vẹn dữ liệu (CRC) của archive."""
        return self.backend.test(self.archive_path, raise_exception=raise_exception)

    def __enter__(self) -> "ArchiveFile":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


def open_archive(archive_path: Union[str, Path]) -> ArchiveFile:
    """
    Tác dụng:
    - Mở một file archive và trả về đối tượng ArchiveFile (OOP Context Manager) để gọi liên tiếp các phương thức mà không cần reopen file nhiều lần.

    Đầu vào:
    - archive_path [str | Path]: Đường dẫn file lưu trữ.

    Đầu ra:
    - [ArchiveFile] Đối tượng quản lý file archive.

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Sử dụng với cú pháp câu lệnh with
    >>> with ar.open("dataset.zip") as archive:
    ...     print("Format:", archive.format)
    ...     print("File count:", len(archive.list_files()))
    ...     archive.extract(output_dir="./out", overwrite=True)
    Format: zip
    File count: 723

    Nguồn: TrinhNhuNhat_28072026.
    """
    return ArchiveFile(archive_path)
