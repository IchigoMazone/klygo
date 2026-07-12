from zipfile import ZipFile
from pathlib import Path

from tqdm import tqdm

from klygo.validators.archive import Extract, ExtractFile


def extract(
    archive_path: str | Path,
    output_dir: str | Path = ".",
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Tác dụng:
    - Giải nén toàn bộ file lưu trữ vào thư mục đích

    Đầu vào:
    - archive_path: Đường dẫn file lưu trữ
    - output_dir: Đường dẫn thư mục đầu ra
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

    params = Extract(
        archive_path=archive_path,
        output_dir=output_dir,
        overwrite=overwrite,
        verbose=verbose,
    )

    source_path: Path = params.archive_path
    output_path: Path = params.output_dir

    output_path.mkdir(parents=True, exist_ok=True)

    with ZipFile(source_path, mode="r") as zf:
        members = zf.infolist()

        if not overwrite:
            existing = [
                m for m in members
                if (output_path / m.filename).exists()
            ]
            if existing:
                names = ", ".join(m.filename for m in existing[:5])
                suffix = f"… (+{len(existing) - 5} more)" if len(existing) > 5 else ""
                raise FileExistsError(
                    f"Files already exist in output directory: {names}{suffix}. "
                    "Use overwrite=True to replace them."
                )

        iterator = (
            tqdm(
                members,
                desc="Extracting",
                unit="file",
                colour="cyan",
                bar_format="{l_bar}{bar:30}{r_bar}",
            )
            if verbose
            else iter(members)
        )

        for member in iterator:
            zf.extract(member, path=output_path)

    if verbose:
        print(f"Done. Extracted {len(members)} file(s) to '{output_path}'")


def extract_file(
    archive_path: str | Path,
    filename: str,
    output_dir: str | Path = ".",
    overwrite: bool = False,
) -> None:
    """
    Tác dụng:
    - Giải nén một file cụ thể từ file lưu trữ

    Đầu vào:
    - archive_path: Đường dẫn file lưu trữ
    - filename: Tham số filename của hàm
    - output_dir: Đường dẫn thư mục đầu ra
    - overwrite: Trạng thái cho phép ghi đè

    Đầu ra:
    - Không trả về dữ liệu

    Ngoại lệ:
    - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - FileExistsError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - KeyError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
    """

    params = ExtractFile(
        archive_path=archive_path,
        filename=filename,
        output_dir=output_dir,
        overwrite=overwrite,
    )

    output_path: Path = params.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    with ZipFile(params.archive_path, mode="r") as zf:
        names = zf.namelist()
        if params.filename not in names:
            raise KeyError(
                f"'{params.filename}' not found in archive. "
                f"Use list_files() to see available files."
            )

        target = output_path / Path(params.filename).name
        if target.exists() and not params.overwrite:
            raise FileExistsError(
                f"file already exists: {target}. Use overwrite=True."
            )

        with open(target, "wb") as file:
            file.write(zf.read(params.filename))
