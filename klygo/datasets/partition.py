import random
import shutil
import time
from pathlib import Path

from klygo.validators.datasets import Partition, Repartition
from klygo.archive import extract
from klygo.io import read_yaml, write_yaml
from klygo.utils.dataset import (
    _find_dataset_root,
    _safe_copy,
    _safe_move,
    _scan_dataset_files,
)


def _execute_partition(params: Partition) -> None:
    # 1. Ratios extraction
    """
    Tác dụng:
    - Thực hiện chức năng _execute_partition

    Đầu vào:
    - params: Tham số params của hàm

    Đầu ra:
    - Không trả về dữ liệu

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
    """
    if len(params.ratios) == 1:
        train = params.ratios[0]
        val = test = (1 - train) / 2
    elif len(params.ratios) == 2:
        train, val = params.ratios
        test = 1 - (train + val)
    else:
        train, val, test = params.ratios

    train, val, test = round(train, 4), round(val, 4), round(test, 4)

    is_zip = params.source.is_file()

    # 2. Source handling (temp extract if ZIP)
    temp_extract_dir = None
    if is_zip:
        temp_extract_dir = params.target.parent / f".temp_partition_extract_{random.randint(1000, 9999)}"
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        extract(
            archive_path=params.source,
            output_dir=temp_extract_dir,
            overwrite=True,
            verbose=params.verbose
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

    # 3. Scan images and labels
    pairs_info = _scan_dataset_files(images_src, labels_src)
    pairs = [(p[0], p[1]) for p in pairs_info]

    # 4. Split datasets reproducibly
    rng = random.Random(42)
    rng.shuffle(pairs)

    total = len(pairs)
    train_count = int(total * train)
    if test == 0.0:
        train_pairs = pairs[:train_count]
        val_pairs = pairs[train_count:]
        test_pairs = []
    else:
        val_count = int(total * val)
        train_pairs = pairs[:train_count]
        val_pairs = pairs[train_count:train_count + val_count]
        test_pairs = pairs[train_count + val_count:]

    splits = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs
    }

    # 5. Setup temporary destination directory
    temp_output_dir = params.target.parent / f".temp_partition_out_{random.randint(1000, 9999)}"
    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir)
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Copy/Move files to the temporary destination splits
    for split_name, pair_list in splits.items():
        if not pair_list:
            continue

        img_dest_dir = temp_output_dir / "images" / split_name
        lbl_dest_dir = temp_output_dir / "labels" / split_name

        for img, lbl in pair_list:
            img_dest = img_dest_dir / img.name
            if is_zip:
                _safe_move(img, img_dest)
                if lbl:
                    _safe_move(lbl, lbl_dest_dir / lbl.name)
            else:
                _safe_copy(img, img_dest)
                if lbl:
                    _safe_copy(lbl, lbl_dest_dir / lbl.name)

    # 7. Handle data.yaml
    yaml_src = src_base / "data.yaml"
    if yaml_src.exists():
        yaml_data = read_yaml(yaml_src, verbose=False)
        yaml_data["path"] = str(params.target.resolve().as_posix())
        yaml_data["train"] = "images/train"
        yaml_data["val"] = "images/val"
        if test_pairs:
            yaml_data["test"] = "images/test"
        elif "test" in yaml_data:
            del yaml_data["test"]

        yaml_dest = temp_output_dir / "data.yaml"
        write_yaml(yaml_dest, yaml_data, overwrite=True, verbose=params.verbose)

    # 8. Clear extraction temp dir if ZIP
    if temp_extract_dir and temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)

    # 9. Replace target directory atomically
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


def partition(
    source: str | Path,
    target: str | Path,
    ratios: tuple[float],
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Tác dụng:
    - Chia dataset thành các tập train, validation và test

    Đầu vào:
    - source: File hoặc thư mục đầu vào
    - target: File hoặc thư mục đầu ra
    - ratios: Tham số ratios của hàm
    - overwrite: Trạng thái cho phép ghi đè
    - verbose: Trạng thái hiển thị tiến trình

    Đầu ra:
    - Không trả về dữ liệu

    Nguồn: TrinhNhuNhat_12072026.
    """
    params = Partition(
        source=source,
        target=target,
        ratios=ratios,
        overwrite=overwrite,
        verbose=verbose
    )
    _execute_partition(params)


def repartition(
    source: str | Path,
    target: str | Path,
    ratios: tuple[float],
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Tác dụng:
    - Chia lại dataset theo các tỉ lệ mới

    Đầu vào:
    - source: File hoặc thư mục đầu vào
    - target: File hoặc thư mục đầu ra
    - ratios: Tham số ratios của hàm
    - overwrite: Trạng thái cho phép ghi đè
    - verbose: Trạng thái hiển thị tiến trình

    Đầu ra:
    - Không trả về dữ liệu

    Nguồn: TrinhNhuNhat_12072026.
    """
    params = Repartition(
        source=source,
        target=target,
        ratios=ratios,
        overwrite=overwrite,
        verbose=verbose
    )
    _execute_partition(params)


def unpartition(
    source: str | Path,
    target: str | Path,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Tác dụng:
    - Chuyển dataset đã chia tập về cấu trúc phẳng

    Đầu vào:
    - source: File hoặc thư mục đầu vào
    - target: File hoặc thư mục đầu ra
    - overwrite: Trạng thái cho phép ghi đè
    - verbose: Trạng thái hiển thị tiến trình

    Đầu ra:
    - Không trả về dữ liệu

    Ngoại lệ:
    - FileExistsError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
    """
    source = Path(source)
    target = Path(target)

    if target.exists() and not overwrite:
        raise FileExistsError(f"target already exists: {target}")

    is_zip = source.is_file()
    temp_extract_dir = None

    if is_zip:
        temp_extract_dir = target.parent / f".temp_unpartition_extract_{random.randint(1000, 9999)}"
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        extract(source, temp_extract_dir, overwrite=True, verbose=False)
        src_base = _find_dataset_root(temp_extract_dir)
    else:
        src_base = source

    images_src = src_base / "images"
    labels_src = src_base / "labels"

    # We need to find all sub-split files
    pairs = _scan_dataset_files(images_src, labels_src)

    # Prepare temp output dir
    temp_out = target.parent / f".temp_unpartition_out_{random.randint(1000, 9999)}"
    if temp_out.exists():
        shutil.rmtree(temp_out)

    images_dest = temp_out / "images"
    labels_dest = temp_out / "labels"
    images_dest.mkdir(parents=True, exist_ok=True)
    labels_dest.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path, rel_img, rel_lbl in pairs:
        # We flatten by using their rel_img name
        # e.g., images/train/img1.jpg -> img1.jpg
        img_name = rel_img.name
        lbl_name = rel_lbl.name if rel_lbl else None

        _safe_copy(img_path, images_dest / img_name)
        if lbl_path and lbl_path.exists() and lbl_name:
            _safe_copy(lbl_path, labels_dest / lbl_name)

    # Read original yaml names if exists
    yaml_src = src_base / "data.yaml"
    names = []
    if yaml_src.exists():
        yaml_data = read_yaml(yaml_src, verbose=False)
        names = yaml_data.get("names", [])

    # Write new data.yaml
    new_yaml = {
        "path": str(target.resolve()) if not target.name.endswith(".zip") else str(target.parent.resolve()),
        "train": "images",
        "val": "",
        "test": "",
        "nc": len(names),
        "names": names
    }
    write_yaml(temp_out / "data.yaml", new_yaml, overwrite=True, verbose=False)

    # Clean up temp extract
    if temp_extract_dir and temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)

    # Put to final output
    if target.name.endswith(".zip"):
        from klygo.archive import compress
        compress(temp_out, target, overwrite=overwrite, verbose=verbose)
        shutil.rmtree(temp_out)
    else:
        if target.exists() and overwrite:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        temp_out.rename(target)
