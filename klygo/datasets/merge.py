import random
import shutil
from pathlib import Path

from klygo.validators.datasets import Merge as DatasetMerge
from klygo.archive import extract
from klygo.io import write_yaml
from klygo.utils.dataset import (
    _find_dataset_root,
    _read_class_names,
    _remap_label_file,
    _safe_copy,
    _safe_move,
    _scan_dataset_files,
)


def merge(
    sources: list[str | Path],
    output_path: str | Path,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:

    """
    Tác dụng:
    - Gộp nhiều nguồn dữ liệu thành một kết quả

    Đầu vào:
    - sources: Danh sách file hoặc thư mục đầu vào
    - output_path: Đường dẫn file đầu ra
    - overwrite: Trạng thái cho phép ghi đè
    - verbose: Trạng thái hiển thị tiến trình

    Đầu ra:
    - Không trả về dữ liệu

    Ngoại lệ:
    - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

    Nguồn: TrinhNhuNhat_12072026.
    """
    params = DatasetMerge(
        sources=sources,
        output_path=output_path,
        overwrite=overwrite,
        verbose=verbose
    )

    # 1. Setup temporary workspace directory
    merge_workspace = params.output_path.parent / f".temp_merge_workspace_{random.randint(1000, 9999)}"
    if merge_workspace.exists():
        shutil.rmtree(merge_workspace)

    images_merge_dir = merge_workspace / "images"
    labels_merge_dir = merge_workspace / "labels"
    images_merge_dir.mkdir(parents=True, exist_ok=True)
    labels_merge_dir.mkdir(parents=True, exist_ok=True)

    # First pass: collect classes from all sources to build global class list
    # and map local IDs to global IDs
    global_names = []
    source_classes_info = []

    for idx, src in enumerate(params.sources):
        is_zip = src.is_file()
        temp_extract_dir = None

        if is_zip:
            temp_extract_dir = params.output_path.parent / f".temp_merge_extract_{idx}_{random.randint(1000, 9999)}"
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir)
            extract(
                archive_path=src,
                output_dir=temp_extract_dir,
                overwrite=True,
                verbose=False
            )
            src_base = _find_dataset_root(temp_extract_dir)
        else:
            src_base = src

        curr_names = _read_class_names(src_base, temp_extract_dir, extra_cleanups=[merge_workspace])

        # Record mapping
        source_classes_info.append({
            "names": curr_names,
            "temp_extract_dir": temp_extract_dir,
            "src_base": src_base,
            "is_zip": is_zip
        })

        # Append unique class names to global list
        for name in curr_names:
            if name not in global_names:
                global_names.append(name)

    # 2. Build map and check if ID modifications are needed
    class_maps = []
    needs_remapping = []

    for info in source_classes_info:
        mapping = {}
        info_names = info["names"]
        for local_id, name in enumerate(info_names):
            global_id = global_names.index(name)
            mapping[local_id] = global_id

        class_maps.append(mapping)

        needs_remap = any(local_id != global_id for local_id, global_id in mapping.items())
        needs_remapping.append(needs_remap)

    # 3. Second pass: copy/move images and labels (remapping class IDs in label files if needed)
    for idx, info in enumerate(source_classes_info):
        src_base = info["src_base"]
        is_zip = info["is_zip"]
        temp_extract_dir = info["temp_extract_dir"]
        class_map = class_maps[idx]
        needs_remap = needs_remapping[idx]

        images_src = src_base / "images"
        labels_src = src_base / "labels"

        if not images_src.exists():
            if temp_extract_dir and temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir)
            if merge_workspace.exists():
                shutil.rmtree(merge_workspace)
            raise FileNotFoundError(f"images directory not found under {src_base}")

        pairs = _scan_dataset_files(images_src, labels_src)

        # Copy/Move files with prefix `src{idx}_`
        for img_path, lbl_path, rel_img, rel_lbl in pairs:
            # Flatten split path into unique name
            # e.g. train/img1.jpg -> src0_train_img1.jpg
            flat_img_name = "_".join(rel_img.parts) if len(rel_img.parts) > 1 else rel_img.name
            flat_lbl_name = "_".join(rel_lbl.parts) if len(rel_lbl.parts) > 1 else rel_lbl.name
            img_dest = images_merge_dir / f"src{idx}_{flat_img_name}"
            lbl_dest = labels_merge_dir / f"src{idx}_{flat_lbl_name}"

            # Copy/Move image
            if is_zip:
                _safe_move(img_path, img_dest)
            else:
                _safe_copy(img_path, img_dest)

            # Copy/Move label
            if lbl_path and lbl_path.exists() and lbl_path.is_file():
                if needs_remap:
                    try:
                        _remap_label_file(lbl_path, lbl_dest, class_map)
                        if is_zip:
                            lbl_path.unlink()
                    except Exception:
                        if is_zip:
                            _safe_move(lbl_path, lbl_dest)
                        else:
                            _safe_copy(lbl_path, lbl_dest)
                else:
                    if is_zip:
                        _safe_move(lbl_path, lbl_dest)
                    else:
                        _safe_copy(lbl_path, lbl_dest)

        if temp_extract_dir and temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)

    # 4. Write unified data.yaml
    yaml_data = {
        "path": "/content/data",
        "train": "images/train",
        "val": "images/val",
        "nc": len(global_names),
        "names": global_names
    }
    write_yaml(merge_workspace / "data.yaml", yaml_data, overwrite=True, verbose=verbose)

    # 5. Compress to the output ZIP file manually to ensure no parent folder in ZIP
    from zipfile import ZipFile, ZIP_DEFLATED
    with ZipFile(params.output_path, mode="w", compression=ZIP_DEFLATED) as zf:
        all_files = sorted(f for f in merge_workspace.rglob("*") if f.is_file())
        for file in all_files:
            arcname = file.relative_to(merge_workspace)
            zf.write(file, arcname=arcname)

    # 6. Cleanup
    if merge_workspace.exists():
        shutil.rmtree(merge_workspace)
