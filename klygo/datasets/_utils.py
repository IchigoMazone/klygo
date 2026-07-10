import shutil
import time
from pathlib import Path
from klygo.io import read_yaml


def _safe_move(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        try:
            shutil.move(str(src), str(dst))
            return
        except PermissionError:
            time.sleep(0.05)
    shutil.copy2(str(src), str(dst))
    for _ in range(10):
        try:
            src.unlink()
            return
        except PermissionError:
            time.sleep(0.05)
    src.unlink()


def _safe_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        try:
            shutil.copy2(str(src), str(dst))
            return
        except PermissionError:
            time.sleep(0.05)
    shutil.copy2(str(src), str(dst))


def _read_class_names(
    src_base: Path,
    temp_extract_dir: Path | None = None,
    extra_cleanups: list[Path] | None = None
) -> list[str]:
    yaml_src = src_base / "data.yaml"
    if not yaml_src.exists():
        if temp_extract_dir and temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        if extra_cleanups:
            for p in extra_cleanups:
                if p.exists():
                    shutil.rmtree(p)
        raise ValueError(f"data.yaml not found under {src_base}")

    yaml_data = read_yaml(yaml_src, verbose=False)
    if "names" not in yaml_data:
        if temp_extract_dir and temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        if extra_cleanups:
            for p in extra_cleanups:
                if p.exists():
                    shutil.rmtree(p)
        raise ValueError(f"names list not found in data.yaml under {src_base}")

    curr_names = yaml_data["names"]
    if isinstance(curr_names, dict):
        curr_names = [n for k, n in sorted(curr_names.items())]
    elif not isinstance(curr_names, list):
        curr_names = [curr_names]
    return curr_names


def _scan_dataset_files(images_src: Path, labels_src: Path) -> list[tuple[Path, Path | None, Path, Path]]:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    scan_dirs = [images_src]
    for split_sub in ["train", "val", "test"]:
        sub_dir = images_src / split_sub
        if sub_dir.exists() and sub_dir.is_dir():
            scan_dirs.append(sub_dir)

    image_files = []
    for d in scan_dirs:
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in image_extensions:
                image_files.append(f)
    
    image_files = sorted(list(set(image_files)))

    pairs = []
    for img_path in image_files:
        try:
            rel_img = img_path.relative_to(images_src)
            rel_lbl = rel_img.with_suffix(".txt")
            lbl_path = labels_src / rel_lbl
        except ValueError:
            rel_lbl = Path(img_path.name).with_suffix(".txt")
            lbl_path = labels_src / rel_lbl

        if not (lbl_path and lbl_path.exists() and lbl_path.is_file()):
            fallback_lbl = labels_src / (img_path.stem + ".txt")
            if fallback_lbl.exists() and fallback_lbl.is_file():
                lbl_path = fallback_lbl
                rel_lbl = Path(fallback_lbl.name)
            else:
                lbl_path = None

        pairs.append((img_path, lbl_path, rel_img, rel_lbl))
    return pairs


def _remap_label_file(src_lbl: Path, dest_lbl: Path, class_map: dict[int, int]) -> None:
    with open(src_lbl, "r", encoding="utf-8", errors="ignore") as f_in:
        lines = f_in.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            try:
                local_id = int(parts[0])
                global_id = class_map.get(local_id, local_id)
                parts[0] = str(global_id)
                new_lines.append(" ".join(parts) + "\n")
            except ValueError:
                new_lines.append(line)
        else:
            new_lines.append(line)

    dest_lbl.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_lbl, "w", encoding="utf-8") as f_out:
        f_out.writelines(new_lines)
