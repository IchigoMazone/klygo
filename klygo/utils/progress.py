import sys
from typing import Optional, Any
from tqdm import tqdm


class ProgressBar:
    """
    Tác dụng:
    - Lớp quản lý Progress Bar dùng chung toàn bộ thư viện klygo với giao diện đồng nhất màu xanh da trời (Cyan), hiển thị tốc độ, ETA đếm ngược và đơn vị tự động.

    Đầu vào:
    - total [int]: Tổng số bước hoặc số lượng phần tử cần xử lý.
    - desc [str]: Mô tả hiển thị đầu thanh tiến trình (ví dụ: 'dataset.zip: compressing').
    - unit [str]: Đơn vị đếm ('file', 'byte', 'it', v.v.). Mặc định: 'file'.
    - verbose [bool]: Trạng thái bật/tắt hiển thị tiến trình. Mặc định: True.
    - colour [str]: Màu sắc thanh tiến trình trong console. Mặc định: 'cyan'.
    - unit_scale [bool]: Tự động quy đổi đơn vị (k, M, G). Mặc định: False.
    - unit_divisor [int]: Cơ số chia quy đổi đơn vị (1024 cho byte, 1000 cho số lượng). Mặc định: 1024.

    Đầu ra:
    - [ProgressBar] Đối tượng quản lý tiến trình.

    Nguồn: TrinhNhuNhat_28072026.
    """

    def __init__(
        self,
        total: int,
        desc: str,
        unit: str = "file",
        verbose: bool = True,
        colour: str = "cyan",
        unit_scale: bool = False,
        unit_divisor: int = 1024,
    ) -> None:
        self.verbose = verbose
        self.bar: Optional[tqdm] = None

        if verbose:
            self.bar = tqdm(
                total=total or 1,
                desc=desc,
                unit=unit,
                unit_scale=unit_scale,
                unit_divisor=unit_divisor,
                colour=colour,
                bar_format="{l_bar}{bar:30}{r_bar}",
                file=sys.stdout,
                leave=True,
            )

    def update(self, n: int = 1) -> None:
        """Cập nhật tiến trình thêm n bước."""
        if self.bar is not None:
            self.bar.update(n)

    def close(self) -> None:
        """Đóng thanh tiến trình."""
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# Alias dùng chung cho archive
ArchiveProgress = ProgressBar


def create_progress_bar(
    total: int,
    desc: str,
    unit: str = "file",
    verbose: bool = True,
    colour: str = "cyan",
    unit_scale: bool = False,
    unit_divisor: int = 1024,
) -> ProgressBar:
    """
    Tác dụng:
    - Hàm khởi tạo nhanh một ProgressBar dùng chung cho toàn dự án klygo.

    Đầu vào:
    - total [int]: Tổng số phần tử.
    - desc [str]: Chuỗi mô tả tiến trình.
    - unit [str]: Đơn vị hiển thị. Mặc định: 'file'.
    - verbose [bool]: Trạng thái hiển thị. Mặc định: True.
    - colour [str]: Màu sắc console. Mặc định: 'cyan'.
    - unit_scale [bool]: Tùy chọn tự động đổi scale đơn vị. Mặc định: False.
    - unit_divisor [int]: Cơ số chia (1024 hoặc 1000). Mặc định: 1024.

    Đầu ra:
    - [ProgressBar] Đối tượng progress bar đã tạo.

    Nguồn: TrinhNhuNhat_28072026.
    """
    return ProgressBar(
        total=total,
        desc=desc,
        unit=unit,
        verbose=verbose,
        colour=colour,
        unit_scale=unit_scale,
        unit_divisor=unit_divisor,
    )
