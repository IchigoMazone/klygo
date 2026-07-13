from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path

from tqdm import tqdm

from klygo.validators.archive import Merge, Split
from klygo.archive.human_size import human_size


def merge(
    archive_paths: list,
    output_path: str | Path,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Tác dụng:
    - Gộp nhiều nguồn dữ liệu thành một kết quả

    Đầu vào:
    - archive_paths: Danh sách đường dẫn file lưu trữ
    - output_path: Đường dẫn file đầu ra
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

    params = Merge(
        archive_paths=archive_paths,
        output_path=output_path,
        overwrite=overwrite,
    )

    # Pre-collect members per source for accurate total count
    src_members: dict[Path, list] = {}
    total = 0
    for src in params.archive_paths:
        with ZipFile(src, mode="r") as zf:
            src_members[src] = zf.infolist()
            total += len(src_members[src])

    bar = (
        tqdm(
            total=total or 1,
            desc="Merging",
            unit="file",
            colour="magenta",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        if verbose
        else None
    )

    with ZipFile(params.output_path, mode="w", compression=ZIP_DEFLATED) as out_zf:
        for src in params.archive_paths:
            with ZipFile(src, mode="r") as src_zf:
                for item in src_members[src]:
                    out_zf.writestr(item, src_zf.read(item.filename))
                    if bar is not None:
                        bar.update(1)

    if bar is not None:
        if total == 0:
            bar.update(1)
        bar.close()

    if verbose:
        size = params.output_path.stat().st_size
        print(
            f"Done. Merged {len(params.archive_paths)} archives -> "
            f"{human_size(size)}"
        )


def split_by_size(
    archive_path: str | Path,
    size: int | float,
    output_dir: str | Path = ".",
    overwrite: bool = False,
    verbose: bool = True,
) -> list[str]:
    """
    Tác dụng:
    - Chia file ZIP thành nhiều file ZIP theo dung lượng tối đa

    Đầu vào:
    - archive_path: Đường dẫn file lưu trữ
    - size: Dung lượng nén tối đa của mỗi file ZIP đầu ra, tính bằng MB
    - output_dir: Đường dẫn thư mục đầu ra
    - overwrite: Trạng thái cho phép ghi đè
    - verbose: Trạng thái hiển thị tiến trình

    Đầu ra:
    - Danh sách đường dẫn các file ZIP đã tạo

    Ngoại lệ:
    - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - FileExistsError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_13072026.
    """

    params = Split(
        archive_path=archive_path,
        size=size,
        output_dir=output_dir,
        overwrite=overwrite,
    )

    params.output_dir.mkdir(parents=True, exist_ok=True)

    stem = params.archive_path.stem
    suffix = params.archive_path.suffix
    parts: list[str] = []
    part_num = 1

    with ZipFile(params.archive_path, mode="r") as src_zf:
        members = src_zf.infolist()

        bar = (
            tqdm(
                total=len(members) or 1,
                desc="Splitting",
                unit="file",
                colour="blue",
                bar_format="{l_bar}{bar:30}{r_bar}",
            )
            if verbose
            else None
        )

        current_members: list = []
        current_size = 0

        def _flush() -> None:
            """
            Tác dụng:
            - Thực hiện chức năng _flush

            Đầu vào:
            - Không có tham số đầu vào

            Đầu ra:
            - Không trả về dữ liệu

            Ngoại lệ:
            - FileExistsError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

            Nguồn: TrinhNhuNhat_12072026.
            """
            nonlocal part_num
            part_path = params.output_dir / f"{stem}_part_{part_num:03d}{suffix}"
            if part_path.exists() and not params.overwrite:
                raise FileExistsError(
                    f"output already exists: {part_path}. "
                    "Use overwrite=True."
                )
            with ZipFile(part_path, mode="w", compression=ZIP_DEFLATED) as pzf:
                for m in current_members:
                    pzf.writestr(m, src_zf.read(m.filename))
            parts.append(str(part_path))
            part_num += 1

        for member in members:
            if current_members and (current_size + member.compress_size) > params.size:
                _flush()
                current_members = []
                current_size = 0
            current_members.append(member)
            current_size += member.compress_size
            if bar is not None:
                bar.update(1)

        if current_members:
            _flush()

        if bar is not None:
            if not members:
                bar.update(1)
            bar.close()

    if verbose:
        print(f"Done. Split into {len(parts)} part(s).")

    return parts
