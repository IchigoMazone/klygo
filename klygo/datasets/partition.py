import random
import shutil
import time
from pathlib import Path

from klygo.validators.datasets import Partition, Repartition
from klygo.archive import extract
from klygo.io import read_yaml, write_yaml
from ._utils import _safe_copy, _safe_move, _scan_dataset_files


def _execute_partition(params: Partition) -> None:
    # 1. Ratios extraction
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
        temp_extract_dir = params.output.parent / f".temp_partition_extract_{random.randint(1000, 9999)}"
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        extract(
            source=params.source,
            output=temp_extract_dir, 
            overwrite=True,
            verbose=params.verbose
        )
        src_base = temp_extract_dir
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
    temp_output_dir = params.output.parent / f".temp_partition_out_{random.randint(1000, 9999)}"
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
        yaml_data["path"] = str(params.output.resolve().as_posix())
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
    if params.output.exists():
        for _ in range(10):
            try:
                if params.output.is_dir():
                    shutil.rmtree(params.output)
                else:
                    params.output.unlink()
                break
            except PermissionError:
                time.sleep(0.05)
        else:
            if params.output.is_dir():
                shutil.rmtree(params.output)
            else:
                params.output.unlink()

    for _ in range(10):
        try:
            temp_output_dir.rename(params.output)
            break
        except PermissionError:
            time.sleep(0.05)
    else:
        temp_output_dir.rename(params.output)


def partition(
    source: str | Path,
    output: str | Path,
    ratios: tuple[float],
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    params = Partition(
        source=source,
        output=output,
        ratios=ratios,
        overwrite=overwrite,
        verbose=verbose
    )
    _execute_partition(params)


def repartition(
    source: str | Path,
    output: str | Path,
    ratios: tuple[float],
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    params = Repartition(
        source=source,
        output=output,
        ratios=ratios,
        overwrite=overwrite,
        verbose=verbose
    )
    _execute_partition(params)
