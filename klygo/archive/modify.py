from pathlib import Path
from typing import Union, List, Literal

from klygo.archive.backend import get_backend
from klygo.validators.archive import Add, Remove


def add(
    archive_path: Union[str, Path],
    files: Union[str, Path, List[Union[str, Path]]],
    verbose: bool = True,
    on_conflict: Literal["rename", "overwrite", "skip"] = "rename",
) -> None:
    """
    Tác dụng:
    - Thêm một hoặc nhiều file, thư mục mới vào file lưu trữ đã tồn tại.

    Định dạng tương thích:
    - Hỗ trợ: ZIP (thêm nhanh trực tiếp), TAR, TAR.GZ, TAR.XZ, 7Z (tự động nén rebuild archive). Không hỗ trợ: GZ file lẻ, RAR.

    Đầu vào:
    - archive_path [str | Path]: Đường dẫn file lưu trữ đích.
    - files [str | Path | list]: Một hoặc danh sách các đường dẫn file/thư mục cần thêm vào.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan trong console. Mặc định: True.
    - on_conflict [str]: Chiến lược xử lý khi trùng tên file trong archive ('rename': đổi tên với hậu tố _dup, 'overwrite': ghi đè, 'skip': bỏ qua). Mặc định: 'rename'.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Thêm 1 file mới vào ZIP đã có
    >>> ar.add("dataset.zip", "new_image.jpg", verbose=True)
    # Kết quả hiển thị thanh tiến trình:
    # dataset.zip: adding: 100%|##############################| 1/1 [00:00<00:00, 1200file/s]

    # Ví dụ 2: Thêm nhiều file và thư mục với chiến lược bỏ qua file trùng tên (skip)
    >>> ar.add("data.zip", ["extra1.png", "extra_dir/"], on_conflict="skip")

    Nguồn: TrinhNhuNhat_28072026.
    """
    path = Path(archive_path)

    if isinstance(files, (str, Path)):
        files_list = [Path(files)]
    else:
        files_list = [Path(f) for f in files]

    backend = get_backend(path)
    backend.add(
        archive_path=path,
        files=files_list,
        on_conflict=on_conflict,
        verbose=verbose,
    )


def remove(
    archive_path: Union[str, Path],
    files: Union[str, List[str]],
) -> None:
    """
    Tác dụng:
    - Xóa một hoặc nhiều file được chỉ định khỏi file lưu trữ bằng Streaming I/O tiết kiệm RAM.

    Định dạng tương thích:
    - Hỗ trợ: ZIP, TAR, TAR.GZ, TAR.XZ, 7Z (sử dụng rebuild archive qua stream). Không hỗ trợ: GZ file lẻ, RAR.

    Đầu vào:
    - archive_path [str | Path]: Đường dẫn file lưu trữ.
    - files [str | list[str]]: Tên file hoặc danh sách tên file bên trong archive cần xóa.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ngoại lệ:
    - KeyError: Phát sinh khi một hoặc nhiều file chỉ định xóa không tồn tại trong archive.

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Xóa 1 file temp.log khỏi dataset.zip
    >>> ar.remove("dataset.zip", "temp.log")

    # Ví dụ 2: Xóa nhiều file rác khỏi file nén TAR.GZ
    >>> ar.remove("data.tar.gz", ["old_label.txt", "debug/cache.tmp"])

    Nguồn: TrinhNhuNhat_28072026.
    """
    path = Path(archive_path)

    if isinstance(files, str):
        files_list = [files]
    else:
        files_list = files

    backend = get_backend(path)
    backend.remove(archive_path=path, files=files_list)
