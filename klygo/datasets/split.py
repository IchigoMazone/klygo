import random
import shutil
from pathlib import Path

from klygo.validators.datasets import Split as DatasetSplit
from klygo.archive import extract
from klygo.io import write_yaml
from ._utils import _safe_copy, _read_class_names, _scan_dataset_files


def split(
    source: str | Path,
    output_dir: str | Path,
    by_class: bool = False,
    class_groups: list[list[str]] | None = None,
    ratios: tuple[float] | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    
    params = DatasetSplit(
        source=source,
        output_dir=output_dir,
        by_class=by_class,
        class_groups=class_groups,
        ratios=ratios,
        overwrite=overwrite,
        verbose=verbose
    )

    params.output_dir.mkdir(parents=True, exist_ok=True)

    is_zip = params.source.is_file()
    temp_extract_dir = None

    # 1. Source handling (extract if ZIP)
    if is_zip:
        temp_extract_dir = params.output_dir / f".temp_split_extract_{random.randint(1000, 9999)}"
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        extract(
            source=params.source,
            output=temp_extract_dir,
            overwrite=True,
            verbose=False
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

    source_names = _read_class_names(src_base, temp_extract_dir)

    # 2. Scan all image-label pairs
    pairs_info = _scan_dataset_files(images_src, labels_src)
    pairs = [(p[0], p[1]) for p in pairs_info]

    # 3. Mode selection
    if params.class_groups is not None or params.by_class:
        # MODE 1: Split by Class
        if params.class_groups is not None:
            groups = params.class_groups
        else:
            groups = [[name] for name in source_names]

        # Overwrite check
        for idx, group in enumerate(groups):
            if len(group) == 1:
                zip_name = f"split_{group[0]}.zip"
            else:
                zip_name = f"split_group_{idx}.zip"
            zip_dest = params.output_dir / zip_name
            if zip_dest.exists() and not params.overwrite:
                if temp_extract_dir and temp_extract_dir.exists():
                    shutil.rmtree(temp_extract_dir)
                raise FileExistsError(f"split file already exists: {zip_dest}")

        # Process each class group
        for idx, group in enumerate(groups):
            if len(group) == 1:
                zip_name = f"split_{group[0]}.zip"
            else:
                zip_name = f"split_group_{idx}.zip"
            zip_dest = params.output_dir / zip_name

            workspace = params.output_dir / f".temp_split_workspace_{random.randint(1000, 9999)}"
            if workspace.exists():
                shutil.rmtree(workspace)
            img_dest_dir = workspace / "images"
            lbl_dest_dir = workspace / "labels"
            img_dest_dir.mkdir(parents=True, exist_ok=True)
            lbl_dest_dir.mkdir(parents=True, exist_ok=True)

            copied_count = 0

            for img, lbl in pairs:
                if lbl and lbl.exists() and lbl.is_file():
                    with open(lbl, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                    
                    matched_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            local_id = int(parts[0])
                            if local_id < len(source_names):
                                class_name = source_names[local_id]
                                if class_name in group:
                                    new_id = group.index(class_name)
                                    parts[0] = str(new_id)
                                    matched_lines.append(" ".join(parts) + "\n")

                    if matched_lines:
                        _safe_copy(img, img_dest_dir / img.name)
                        with open(lbl_dest_dir / f"{img.stem}.txt", "w", encoding="utf-8") as f_out:
                            f_out.writelines(matched_lines)
                        copied_count += 1

            if copied_count > 0:
                yaml_data = {
                    "path": "/content/data",
                    "train": "images/train",
                    "val": "images/val",
                    "nc": len(group),
                    "names": group
                }
                write_yaml(workspace / "data.yaml", yaml_data, overwrite=True, verbose=False)

                from zipfile import ZipFile, ZIP_DEFLATED
                with ZipFile(zip_dest, mode="w", compression=ZIP_DEFLATED) as zf:
                    all_files = sorted(f for f in workspace.rglob("*") if f.is_file())
                    for file in all_files:
                        arcname = file.relative_to(workspace)
                        zf.write(file, arcname=arcname)

            if workspace.exists():
                shutil.rmtree(workspace)

            if verbose:
                print(f"Created class split ZIP: {zip_dest.name} with {copied_count} image(s)")

    else:
        # MODE 2: Split by Ratio
        # Overwrite check
        for idx in range(len(params.ratios)):
            zip_name = f"split_part_{idx}.zip"
            zip_dest = params.output_dir / zip_name
            if zip_dest.exists() and not params.overwrite:
                if temp_extract_dir and temp_extract_dir.exists():
                    shutil.rmtree(temp_extract_dir)
                raise FileExistsError(f"split file already exists: {zip_dest}")

        rng = random.Random(42)
        rng.shuffle(pairs)

        total = len(pairs)
        start_idx = 0

        for idx, ratio in enumerate(params.ratios):
            zip_name = f"split_part_{idx}.zip"
            zip_dest = params.output_dir / zip_name

            if idx == len(params.ratios) - 1:
                end_idx = total
            else:
                end_idx = start_idx + int(total * ratio)

            split_pairs = pairs[start_idx:end_idx]
            start_idx = end_idx

            workspace = params.output_dir / f".temp_split_workspace_{random.randint(1000, 9999)}"
            if workspace.exists():
                shutil.rmtree(workspace)
            img_dest_dir = workspace / "images"
            lbl_dest_dir = workspace / "labels"
            img_dest_dir.mkdir(parents=True, exist_ok=True)
            lbl_dest_dir.mkdir(parents=True, exist_ok=True)

            for img, lbl in split_pairs:
                _safe_copy(img, img_dest_dir / img.name)
                if lbl and lbl.exists() and lbl.is_file():
                    _safe_copy(lbl, lbl_dest_dir / lbl.name)

            yaml_data = {
                "path": "/content/data",
                "train": "images/train",
                "val": "images/val",
                "nc": len(source_names),
                "names": source_names
            }
            write_yaml(workspace / "data.yaml", yaml_data, overwrite=True, verbose=False)

            from zipfile import ZipFile, ZIP_DEFLATED
            with ZipFile(zip_dest, mode="w", compression=ZIP_DEFLATED) as zf:
                all_files = sorted(f for f in workspace.rglob("*") if f.is_file())
                for file in all_files:
                    arcname = file.relative_to(workspace)
                    zf.write(file, arcname=arcname)

            if workspace.exists():
                shutil.rmtree(workspace)

            if verbose:
                print(f"Created ratio split ZIP: {zip_name} with {len(split_pairs)} image(s)")

    if temp_extract_dir and temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)
