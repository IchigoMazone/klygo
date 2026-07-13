import random
import shutil
import time
from pathlib import Path

from klygo.validators.datasets import RemapClasses as DatasetRemapClasses
from klygo.archive import extract
from klygo.io import read_yaml, write_yaml
from klygo.utils.dataset import (
    _find_dataset_root,
    _read_class_names,
    _remap_label_file,
    _safe_copy,
    _scan_dataset_files,
)


def remap_classes(
    source: str | Path,
    target: str | Path,
    class_map: dict[int | str, int | str],
    overwrite: bool = False,
    verbose: bool = True,
) -> None:

    """
    Tác dụng:
    - Ánh xạ lại class của dataset

    Đầu vào:
    - source: File hoặc thư mục đầu vào
    - target: File hoặc thư mục đầu ra
    - class_map: Tham số class_map của hàm
    - overwrite: Trạng thái cho phép ghi đè
    - verbose: Trạng thái hiển thị tiến trình

    Đầu ra:
    - Không trả về dữ liệu

    Ngoại lệ:
    - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
    - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
    """
    params = DatasetRemapClasses(
        source=source,
        target=target,
        class_map=class_map,
        overwrite=overwrite,
        verbose=verbose
    )

    is_zip = params.source.is_file()
    temp_extract_dir = None

    # 1. Source handling (extract if ZIP)
    if is_zip:
        temp_extract_dir = params.target.parent / f".temp_remap_extract_{random.randint(1000, 9999)}"
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        extract(
            archive_path=params.source,
            output_dir=temp_extract_dir,
            overwrite=True,
            verbose=False
        )
        src_base = _find_dataset_root(temp_extract_dir)
    else:
        src_base = params.source

    images_src = src_base / "images"
    labels_src = src_base / "labels"

    if not images_src.exists():
        if temp_extract_dir and temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        raise FileNotFoundError(f"images directory not found under {src_base}")

    source_names = _read_class_names(src_base, temp_extract_dir)

    # 2. Resolve class mapping
    id_to_id = {i: i for i in range(len(source_names))}
    id_to_name = {i: source_names[i] for i in range(len(source_names))}

    for k, v in params.class_map.items():
        if isinstance(k, str):
            if k in source_names:
                old_id = source_names.index(k)
            else:
                if temp_extract_dir and temp_extract_dir.exists():
                    shutil.rmtree(temp_extract_dir)
                raise ValueError(f"Class name '{k}' not found in source dataset.")
        elif isinstance(k, int):
            old_id = k
            if old_id >= len(source_names):
                if temp_extract_dir and temp_extract_dir.exists():
                    shutil.rmtree(temp_extract_dir)
                raise ValueError(f"Class ID {k} out of range of source dataset classes.")
        else:
            if temp_extract_dir and temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir)
            raise TypeError("class_map keys must be int or str.")

        if isinstance(v, str):
            id_to_name[old_id] = v
        elif isinstance(v, int):
            id_to_id[old_id] = v
        else:
            if temp_extract_dir and temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir)
            raise TypeError("class_map values must be int or str.")

    new_names = {new_id: id_to_name[old_id] for old_id, new_id in id_to_id.items()}
    needs_remap = any(old_id != new_id for old_id, new_id in id_to_id.items())

    # 3. Setup temporary output folder
    is_out_zip = str(params.target).lower().endswith(".zip")
    temp_output_dir = params.target.parent / f".temp_remap_out_{random.randint(1000, 9999)}"
    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir)
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    img_dest_dir = temp_output_dir / "images"
    lbl_dest_dir = temp_output_dir / "labels"
    img_dest_dir.mkdir(parents=True, exist_ok=True)
    lbl_dest_dir.mkdir(parents=True, exist_ok=True)

    # 4. Scan and process all image-label pairs
    pairs = _scan_dataset_files(images_src, labels_src)
    for img_path, lbl_path, rel_img, rel_lbl in pairs:
        dest_img = img_dest_dir / rel_img
        dest_lbl = lbl_dest_dir / rel_lbl
        dest_img.parent.mkdir(parents=True, exist_ok=True)
        dest_lbl.parent.mkdir(parents=True, exist_ok=True)

        _safe_copy(img_path, dest_img)

        if lbl_path and lbl_path.exists() and lbl_path.is_file():
            if needs_remap:
                try:
                    _remap_label_file(lbl_path, dest_lbl, id_to_id)
                except Exception:
                    _safe_copy(lbl_path, dest_lbl)
            else:
                _safe_copy(lbl_path, dest_lbl)

    # 5. Write new data.yaml
    yaml_data = read_yaml(src_base / "data.yaml", verbose=False)
    new_yaml = dict(yaml_data)
    new_yaml["path"] = str(params.target.resolve().as_posix()) if not is_out_zip else "/content/data"

    if new_names and set(new_names.keys()) == set(range(len(new_names))):
        yaml_names = [new_names[i] for i in range(len(new_names))]
    else:
        yaml_names = new_names

    new_yaml["nc"] = max(new_names.keys()) + 1 if new_names else 0
    new_yaml["names"] = yaml_names
    write_yaml(temp_output_dir / "data.yaml", new_yaml, overwrite=True, verbose=False)

    if temp_extract_dir and temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)

    # 6. Output handling
    if is_out_zip:
        from zipfile import ZipFile, ZIP_DEFLATED
        with ZipFile(params.target, mode="w", compression=ZIP_DEFLATED) as zf:
            all_files = sorted(f for f in temp_output_dir.rglob("*") if f.is_file())
            for file in all_files:
                arcname = file.relative_to(temp_output_dir)
                zf.write(file, arcname=arcname)
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)
    else:
        if params.target.exists():
            for _ in range(10):
                try:
                    if params.target.is_dir():
                        shutil.rmtree(params.target)
                    else:
                        params.target.unlink()
                    break
                except PermissionError:
                    time.sleep(0.05)
            else:
                if params.target.is_dir():
                    shutil.rmtree(params.target)
                else:
                    params.target.unlink()

        for _ in range(10):
            try:
                temp_output_dir.rename(params.target)
                break
            except PermissionError:
                time.sleep(0.05)
        else:
            temp_output_dir.rename(params.target)
