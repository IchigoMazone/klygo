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
    - Gộp nhiều file archive nguồn thành một file archive kết quả.
    - Tự động hỗ trợ gộp khác định dạng (cross-format) bằng cơ chế extract → recompress.

    Định dạng tương thích:
    - Nguồn hỗ trợ: ZIP, TAR, TAR.GZ, TAR.XZ, GZ, 7Z, RAR.
    - Đích hỗ trợ: ZIP, TAR, TAR.GZ, TAR.XZ, 7Z.

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
    # merged.zip: merging: 100%|##############################| 78/78 [00:00<00:00, 2100file/s]

    # Ví dụ 2: Gộp khác định dạng — zip + tar.gz → merged.zip
    >>> ar.merge(["data.zip", "extra.tar.gz"], "merged.zip", overwrite=True)

    Nguồn: TrinhNhuNhat_28072026.
    """
    from klygo.archive.backend import detect_format

    out_path = Path(output_path)
    sources = [Path(p) for p in archive_paths]

    out_fmt = detect_format(out_path)
    src_fmts = [detect_format(p) for p in sources]
    all_same_format = all(f == out_fmt for f in src_fmts)

    if all_same_format:
        # Fast path — same format, merge directly
        backend = get_backend(out_path)
        backend.merge(
            archive_paths=sources,
            output_path=out_path,
            overwrite=overwrite,
            verbose=verbose,
        )
    else:
        # Cross-format path — extract all to temp dir, then compress
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"output_path already exists: {out_path}. Use overwrite=True.")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for src in sources:
                src_backend = get_backend(src)
                src_backend.extract(src, tmp_path, verbose=verbose)
            dst_backend = get_backend(out_path)
            dst_backend.compress(tmp_path, out_path, overwrite=overwrite, verbose=verbose)



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

    Định dạng tương thích:
    - Hỗ trợ: ZIP, TAR, TAR.GZ, TAR.XZ, 7Z.

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

    Định dạng tương thích:
    - Nguồn hỗ trợ: ZIP, TAR, TAR.GZ, TAR.XZ, GZ, 7Z, RAR.
    - Đích hỗ trợ: ZIP, TAR, TAR.GZ, TAR.XZ, 7Z.

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

    Định dạng tương thích:
    - Hỗ trợ: ZIP, TAR, TAR.GZ, TAR.XZ, 7Z.

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


def copy(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    overwrite: bool = False,
) -> None:
    """
    Tác dụng:
    - Sao chép file archive sang đường dẫn đích mà không giải nén hay thay đổi nội dung.

    Định dạng tương thích:
    - Hỗ trợ tất cả mọi định dạng: ZIP, TAR, TAR.GZ, TAR.XZ, GZ, 7Z, RAR.

    Đầu vào:
    - source_path [str | Path]: Đường dẫn file archive nguồn.
    - target_path [str | Path]: Đường dẫn file archive đích.
    - overwrite [bool]: Cho phép ghi đè nếu target_path đã tồn tại. Mặc định: False.

    Đầu ra:
    - [None] Không trả về dữ liệu.

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi source_path không tồn tại.
    - FileExistsError: Phát sinh khi target_path đã tồn tại và overwrite=False.

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: Sao chép file ZIP sang thư mục backup
    >>> ar.copy("dataset.zip", "backup/dataset.zip", overwrite=True)

    # Ví dụ 2: Tạo bản sao với tên khác
    >>> ar.copy("folder.tar.gz", "folder_backup.tar.gz")

    Nguồn: TrinhNhuNhat_28072026.
    """
    src = Path(source_path)
    dst = Path(target_path)

    if not src.exists():
        raise FileNotFoundError(f"source_path does not exist: {src}")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"target_path already exists: {dst}. Use overwrite=True.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
