from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path

from tqdm import tqdm

from klygo.validators.archive import Compress
from klygo.archive.human_size import human_size


def compress(
    source: str | Path,
    output_path: str | Path,
    format: str = "zip",
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Tác dụng:
    - Nén file hoặc thư mục thành file lưu trữ

    Đầu vào:
    - source: File hoặc thư mục đầu vào
    - output_path: Đường dẫn file đầu ra
    - format: Tham số format của hàm
    - overwrite: Trạng thái cho phép ghi đè
    - verbose: Trạng thái hiển thị tiến trình

    Đầu ra:
    - Không trả về dữ liệu

    Ngoại lệ:
    - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - FileExistsError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
    """

    params = Compress(
        source=source,
        output_path=output_path,
        format=format,
        overwrite=overwrite,
        verbose=verbose,
    )

    source_item: Path = params.source
    output_path = params.output_path

    # Collect all files to compress
    if source_item.is_dir():
        all_files: list[Path] = sorted(
            f for f in source_item.rglob("*") if f.is_file()
        )
    else:
        all_files = [source_item]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output_path, mode="w", compression=ZIP_DEFLATED) as zf:
        bar = (
            tqdm(
                total=len(all_files) or 1,
                desc="Compressing",
                unit="file",
                colour="green",
                bar_format="{l_bar}{bar:30}{r_bar}",
            )
            if verbose
            else None
        )
        for file in all_files:
            arcname = (
                file.relative_to(source_item.parent)
                if source_item.is_dir()
                else file.name
            )
            zf.write(file, arcname=arcname)
            if bar is not None:
                bar.update(1)
        if bar is not None:
            if not all_files:
                bar.update(1)
            bar.close()

    if verbose:
        total_size = output_path.stat().st_size
        print(f"Done. Archive size: {human_size(total_size)}")
