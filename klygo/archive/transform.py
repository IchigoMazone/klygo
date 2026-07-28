import shutil
import tempfile
from pathlib import Path
from typing import Union, List

from klygo.archive.backend import get_backend
from klygo.archive.human_size import human_size
from klygo.validators.archive import Merge, Split


def merge(
    archive_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Tác dụng:
    - Gộp nhiều file archive nguồn thành một file archive kết quả bằng Streaming I/O tiết kiệm RAM.

    Đầu vào:
    - archive_paths [list[str | Path]]: Danh sách các file archive nguồn cần gộp (tối thiểu 2 file).
    - output_path [str | Path]: Đường dẫn file archive đầu ra sau khi gộp.
    - overwrite [bool]: Cho phép ghi đè nếu output_path đã tồn tại. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan trong console. Mặc định: True.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Gộp 2 file ZIP thành merged.zip
    >>> ar.merge(["part1.zip", "part2.zip"], "merged.zip", overwrite=True)
    # Kết quả hiển thị thanh tiến trình:
    # merged.zip: merging: 100%|##############################| 78/78 [00:00<00:00, 2100file/s]

    Nguồn: TrinhNhuNhat_28072026.
    """
    out_path = Path(output_path)
    sources = [Path(p) for p in archive_paths]

    backend = get_backend(out_path)
    backend.merge(
        archive_paths=sources,
        output_path=out_path,
        overwrite=overwrite,
        verbose=verbose,
    )


def split_by_size(
    archive_path: Union[str, Path],
    size: Union[int, float],
    output_dir: Union[str, Path] = ".",
    overwrite: bool = False,
    verbose: bool = True,
) -> List[str]:
    """
    Tác dụng:
    - Chia một file archive lớn thành nhiều file archive nhỏ hơn (part archive) theo dung lượng tối đa quy định (MB).

    Đầu vào:
    - archive_path [str | Path]: Đường dẫn file archive nguồn cần chia nhỏ.
    - size [int | float]: Dung lượng tối đa của mỗi file nén part đầu ra tính bằng Megabytes (MB).
    - output_dir [str | Path]: Thư mục chứa các file part nén đầu ra. Mặc định: ".".
    - overwrite [bool]: Cho phép ghi đè file part nếu đã tồn tại. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan trong console. Mặc định: True.

    Đầu ra:
    - [List[str]] Danh sách đường dẫn của tất cả các file part đã được tạo ra.

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Chia file data.zip thành các phần có dung lượng tối đa 5 MB
    >>> parts = ar.split_by_size("data.zip", size=5, output_dir="./parts", overwrite=True)
    >>> print(parts)
    ['./parts/data_part_001.zip', './parts/data_part_002.zip', './parts/data_part_003.zip']

    Nguồn: TrinhNhuNhat_28072026.
    """
    path = Path(archive_path)
    out_dir = Path(output_dir)

    backend = get_backend(path)
    return backend.split_by_size(
        archive_path=path,
        size=float(size),
        output_dir=out_dir,
        overwrite=overwrite,
        verbose=verbose,
    )


def convert(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Tác dụng:
    - Chuyển đổi định dạng của file archive sang định dạng khác (ví dụ: chuyển từ .zip sang .tar.gz hoặc .7z).

    Đầu vào:
    - source_path [str | Path]: Đường dẫn file archive nguồn.
    - target_path [str | Path]: Đường dẫn file archive mục tiêu (định dạng được nhận diện tự động từ đuôi file).
    - overwrite [bool]: Cho phép ghi đè nếu target_path đã tồn tại. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan trong console. Mặc định: True.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Chuyển đổi file dataset.zip sang định dạng dataset.tar.gz
    >>> ar.convert("dataset.zip", "dataset.tar.gz", overwrite=True)
    # Kết quả hiển thị thanh tiến trình:
    # dataset.zip: extracting: 100%|##############################| 48/48 [00:00<00:00, 520file/s]
    # dataset.tar.gz: compressing: 100%|##############################| 48/48 [00:00<00:00, 180file/s]

    Nguồn: TrinhNhuNhat_28072026.
    """
    src = Path(source_path)
    dst = Path(target_path)

    if dst.exists() and not overwrite:
        raise FileExistsError(f"Target file already exists: {dst}. Use overwrite=True.")

    src_backend = get_backend(src)
    dst_backend = get_backend(dst)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_backend.extract(src, tmp_path, verbose=verbose)
        dst_backend.compress(tmp_path, dst, overwrite=overwrite, verbose=verbose)


def recompress(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    compresslevel: int = 6,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Tác dụng:
    - Nén lại file archive với mức độ nén hoặc thuật toán nén khác để tối ưu dung lượng đĩa.

    Đầu vào:
    - source_path [str | Path]: Đường dẫn file archive nguồn.
    - target_path [str | Path]: Đường dẫn file archive đầu ra nén lại.
    - compresslevel [int]: Mức độ nén từ 1 (nhanh) đến 9 (nén tối đa). Mặc định: 6.
    - overwrite [bool]: Cho phép ghi đè file mục tiêu. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình Cyan trong console. Mặc định: True.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Nén lại backup.zip với mức nén cao nhất (compresslevel=9)
    >>> ar.recompress("backup.zip", "backup_max.zip", compresslevel=9, overwrite=True)

    Nguồn: TrinhNhuNhat_28072026.
    """
    src = Path(source_path)
    dst = Path(target_path)

    if dst.exists() and not overwrite:
        raise FileExistsError(f"Target file already exists: {dst}. Use overwrite=True.")

    src_backend = get_backend(src)
    dst_backend = get_backend(dst)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_backend.extract(src, tmp_path, verbose=verbose)
        dst_backend.compress(tmp_path, dst, compresslevel=compresslevel, overwrite=overwrite, verbose=verbose)
