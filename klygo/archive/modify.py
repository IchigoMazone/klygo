from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path

from tqdm import tqdm

from klygo.validators.archive import Add, Remove


def add(
    archive_path: str | Path,
    files: "str | Path | list",
    verbose: bool = True,
) -> None:
    """
    Tác dụng:
    - Thêm file hoặc thư mục vào file lưu trữ

    Đầu vào:
    - archive_path: Đường dẫn file lưu trữ
    - files: Tham số files của hàm
    - verbose: Trạng thái hiển thị tiến trình

    Đầu ra:
    - Không trả về dữ liệu

    Ngoại lệ:
    - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
    """

    params = Add(archive_path=archive_path, files=files, verbose=verbose)

    # Expand directories to individual files
    all_files: list[tuple[Path, str]] = []  # (absolute_path, arcname)
    for fp in params.files:
        if fp.is_dir():
            for child in sorted(fp.rglob("*")):
                if child.is_file():
                    all_files.append((child, str(child.relative_to(fp.parent))))
        else:
            all_files.append((fp, fp.name))

    with ZipFile(params.archive_path, mode="a", compression=ZIP_DEFLATED) as zf:
        existing = set(zf.namelist())
        bar = (
            tqdm(
                total=len(all_files) or 1,
                desc="Adding",
                unit="file",
                colour="yellow",
                bar_format="{l_bar}{bar:30}{r_bar}",
            )
            if verbose
            else None
        )
        for abs_path, arcname in all_files:
            # Avoid silently overwriting; suffix with _dup if name conflicts
            if arcname in existing:
                stem = Path(arcname).stem
                suffix = Path(arcname).suffix
                arcname = f"{stem}_dup{suffix}"
            zf.write(abs_path, arcname=arcname)
            if bar is not None:
                bar.update(1)
        if bar is not None:
            if not all_files:
                bar.update(1)
            bar.close()

    if verbose:
        print(f"Done. Added {len(all_files)} file(s) to '{params.archive_path}'")


def remove(
    archive_path: str | Path,
    files: "str | list[str]",
) -> None:
    """
    Tác dụng:
    - Xóa các file được chỉ định khỏi file lưu trữ

    Đầu vào:
    - archive_path: Đường dẫn file lưu trữ
    - files: Tham số files của hàm

    Đầu ra:
    - Không trả về dữ liệu

    Ngoại lệ:
    - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - KeyError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
    """

    params = Remove(archive_path=archive_path, files=files)
    to_remove = set(params.files)

    with ZipFile(params.archive_path, mode="r") as zf:
        names = set(zf.namelist())
        missing = to_remove - names
        if missing:
            raise KeyError(
                f"Files not found in archive: {sorted(missing)}. "
                f"Use list_files() to see available files."
            )

        tmp_path = params.archive_path.with_suffix(".tmp.zip")
        with ZipFile(tmp_path, mode="w", compression=ZIP_DEFLATED) as tmp_zf:
            for item in zf.infolist():
                if item.filename not in to_remove:
                    tmp_zf.writestr(item, zf.read(item.filename))

    tmp_path.replace(params.archive_path)
