"""Convert Pascal VOC or COCO object-detection datasets to flat YOLO format."""

from __future__ import annotations

from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Any, Sequence
import warnings
import xml.etree.ElementTree as ET

from klygo import archive, config, files, validators
from ._structure import IMAGE_EXTENSIONS, copy_entry
from ._workspace import dataset_workspace, write_output


_SUPPORTED_SRC = {"voc", "coco"}
_SUPPORTED_DST = {"yolo"}


def _resolve_paths(
    source: str | PathLike[str] | None,
    images: str | PathLike[str] | None,
    annotations: str | PathLike[str] | None,
    src_format: str,
) -> tuple[Path, Path]:
    """Resolve images directory and annotations path/directory from inputs."""
    images_path: Path | None = files.path(images) if images is not None else None
    ann_path: Path | None = files.path(annotations) if annotations is not None else None

    if source is not None:
        src_root = files.path(source)
        if not files.exists(src_root):
            raise FileNotFoundError(f"source path does not exist: {src_root}")

        if src_format == "voc":
            if images_path is None:
                for candidate in ["JPEGImages", "images", "JPEGImages/val", "images/val"]:
                    cand_path = files.join(src_root, candidate)
                    if files.is_dir(cand_path):
                        images_path = cand_path
                        break
            if ann_path is None:
                for candidate in ["Annotations", "annotations"]:
                    cand_path = files.join(src_root, candidate)
                    if files.is_dir(cand_path):
                        ann_path = cand_path
                        break
        elif src_format == "coco":
            if images_path is None:
                for candidate in ["val2017", "train2017", "images", "images/val2017"]:
                    cand_path = files.join(src_root, candidate)
                    if files.is_dir(cand_path):
                        images_path = cand_path
                        break
            if ann_path is None:
                if files.is_file(src_root) and files.extension(src_root).lower() == ".json":
                    ann_path = src_root
                else:
                    candidates = files.find(src_root, "*.json", recursive=True)
                    if candidates:
                        ann_path = candidates[0]

    if images_path is None or not files.is_dir(images_path):
        raise ValueError(
            f"could not resolve valid images directory for {src_format}. "
            "Please specify images=... explicitly."
        )
    if ann_path is None or not files.exists(ann_path):
        raise ValueError(
            f"could not resolve valid annotations path for {src_format}. "
            "Please specify annotations=... explicitly."
        )

    return images_path, ann_path


def _parse_classes(
    classes: list[str] | dict[int, str] | None,
) -> dict[int, str] | None:
    """Normalize user-supplied classes mapping or sequence."""
    if classes is None:
        return None
    if isinstance(classes, (list, tuple)):
        class_map = {index: str(name).strip() for index, name in enumerate(classes)}
    elif isinstance(classes, dict):
        try:
            class_map = {int(index): str(name).strip() for index, name in classes.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("class IDs must be integers") from error
    else:
        raise TypeError("classes must be a mapping or sequence of class names")
    if set(class_map) != set(range(len(class_map))):
        raise ValueError("class IDs must be contiguous starting from 0")
    if any(not name for name in class_map.values()) or len(set(class_map.values())) != len(class_map):
        raise ValueError("class names must be non-empty and unique")
    return class_map


def _voc_to_yolo(
    xml_file: Path,
    img_width: float,
    img_height: float,
    name_to_id: dict[str, int],
    *,
    include_difficult: bool,
    voc_1_based: bool,
) -> list[str]:
    """Parse a single Pascal VOC XML annotation file and produce YOLO lines."""
    try:
        tree = ET.parse(xml_file)
    except ET.ParseError as error:
        raise ValueError(f"malformed XML in {xml_file}: {error}") from error
    root = tree.getroot()

    size_elem = root.find("size")
    if size_elem is not None:
        w_elem, h_elem = size_elem.find("width"), size_elem.find("height")
        if w_elem is not None and w_elem.text and h_elem is not None and h_elem.text:
            w_val = float(w_elem.text)
            h_val = float(h_elem.text)
            if w_val > 0 and h_val > 0:
                img_width, img_height = w_val, h_val

    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"invalid image dimensions ({img_width}x{img_height}) in {xml_file}")

    lines: list[str] = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            continue
        cname = name_elem.text.strip()
        if cname not in name_to_id:
            continue
        cid = name_to_id[cname]

        difficult_elem = obj.find("difficult")
        if difficult_elem is not None and difficult_elem.text:
            is_diff = int(difficult_elem.text.strip() or 0) == 1
            if is_diff and not include_difficult:
                continue

        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin_e, ymin_e = bnd.find("xmin"), bnd.find("ymin")
        xmax_e, ymax_e = bnd.find("xmax"), bnd.find("ymax")
        if None in (xmin_e, ymin_e, xmax_e, ymax_e):
            continue

        xmin = float(xmin_e.text)
        ymin = float(ymin_e.text)
        xmax = float(xmax_e.text)
        ymax = float(ymax_e.text)

        if voc_1_based:
            xmin -= 1.0
            ymin -= 1.0

        xmin = max(0.0, min(xmin, img_width))
        ymin = max(0.0, min(ymin, img_height))
        xmax = max(0.0, min(xmax, img_width))
        ymax = max(0.0, min(ymax, img_height))

        box_w = xmax - xmin
        box_h = ymax - ymin
        if box_w <= 0 or box_h <= 0:
            continue

        xc = ((xmin + xmax) / 2.0) / img_width
        yc = ((ymin + ymax) / 2.0) / img_height
        bw = box_w / img_width
        bh = box_h / img_height

        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))

        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _convert_voc(
    images_dir: Path,
    ann_dir: Path,
    workspace: Path,
    classes: dict[int, str] | None,
    image_ids: Sequence[str] | None,
    *,
    include_difficult: bool,
    voc_1_based: bool,
) -> dict[int, str]:
    """Execute Pascal VOC to YOLO conversion in the workspace."""
    # Discover available images
    all_images = [
        path for path in files.find(images_dir, "*", recursive=False)
        if files.extension(path).lower() in IMAGE_EXTENSIONS
    ]
    image_by_stem: dict[str, Path] = {}
    for img_path in all_images:
        stem = files.stem(img_path)
        if stem in image_by_stem:
            raise ValueError(f"duplicate image stem in source: {stem}")
        image_by_stem[stem] = img_path

    if image_ids is not None:
        target_stems = [str(item).strip() for item in image_ids if str(item).strip()]
    else:
        target_stems = sorted(image_by_stem.keys())

    # If classes not provided, discover all unique class names across XMLs
    if classes is None:
        discovered_names: set[str] = set()
        for stem in target_stems:
            xml_path = files.join(ann_dir, f"{stem}.xml")
            if not files.exists(xml_path):
                continue
            try:
                tree = ET.parse(xml_path)
                for obj in tree.getroot().findall("object"):
                    ne = obj.find("name")
                    if ne is not None and ne.text:
                        discovered_names.add(ne.text.strip())
            except Exception:
                pass
        sorted_names = sorted(discovered_names)
        classes = {idx: name for idx, name in enumerate(sorted_names)}

    name_to_id = {name: cid for cid, name in classes.items()}

    images_out = files.join(workspace, "images")
    labels_out = files.join(workspace, "labels")
    files.mkdir(images_out)
    files.mkdir(labels_out)

    processed_count = 0
    for stem in target_stems:
        img_path = image_by_stem.get(stem)
        if img_path is None:
            # Check with extension directly
            cand = files.join(images_dir, f"{stem}.jpg")
            if files.exists(cand):
                img_path = cand
            else:
                continue

        # Default dimensions from image file if needed
        img_w, img_h = 0.0, 0.0
        xml_path = files.join(ann_dir, f"{stem}.xml")

        lines: list[str] = []
        if files.exists(xml_path):
            lines = _voc_to_yolo(
                xml_path,
                img_w,
                img_h,
                name_to_id,
                include_difficult=include_difficult,
                voc_1_based=voc_1_based,
            )

        # Copy original image bytes without re-encoding
        dest_img = files.join(images_out, f"{stem}{files.extension(img_path).lower()}")
        copy_entry(img_path, dest_img)

        # Write label file (empty if no detections -> negative image)
        dest_lbl = files.join(labels_out, f"{stem}.txt")
        files.save(dest_lbl, lines if lines else "", overwrite=True, verbose=False)
        processed_count += 1

    if processed_count == 0:
        warnings.warn("No matching VOC samples were converted", UserWarning, stacklevel=3)
    return classes


def _convert_coco(
    images_dir: Path,
    ann_file: Path,
    workspace: Path,
    classes: dict[int, str] | None,
    image_ids: Sequence[str] | None,
    *,
    include_crowd: bool,
) -> dict[int, str]:
    """Execute COCO instances JSON to YOLO conversion in the workspace."""
    coco_data = files.load(ann_file, verbose=False)
    if not isinstance(coco_data, dict):
        raise ValueError(f"COCO annotations must be a JSON object: {ann_file}")

    raw_categories = coco_data.get("categories", [])
    sorted_categories = sorted(raw_categories, key=lambda cat: int(cat.get("id", 0)))

    if classes is None:
        classes = {idx: str(cat.get("name", "")).strip() for idx, cat in enumerate(sorted_categories)}
        cat_to_yolo = {cat["id"]: idx for idx, cat in enumerate(sorted_categories)}
    else:
        name_to_id = {name: cid for cid, name in classes.items()}
        cat_to_yolo = {}
        for cat in sorted_categories:
            cname = str(cat.get("name", "")).strip()
            if cname in name_to_id:
                cat_to_yolo[cat["id"]] = name_to_id[cname]

    # Map image IDs to image metadata
    raw_images = coco_data.get("images", [])
    img_map: dict[int, dict[str, Any]] = {im["id"]: im for im in raw_images}

    # Group annotations by image_id
    ann_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco_data.get("annotations", []):
        if ann.get("iscrowd") and not include_crowd:
            continue
        ann_by_img[ann["image_id"]].append(ann)

    if image_ids is not None:
        target_ids = {str(item).strip() for item in image_ids if str(item).strip()}
        selected_images = [
            im for im in raw_images
            if str(im["id"]) in target_ids
            or im.get("file_name") in target_ids
            or files.stem(im.get("file_name", "")) in target_ids
        ]
    else:
        selected_images = raw_images

    images_out = files.join(workspace, "images")
    labels_out = files.join(workspace, "labels")
    files.mkdir(images_out)
    files.mkdir(labels_out)

    processed_count = 0
    seen_stems: set[str] = set()
    for im_info in selected_images:
        fname = im_info.get("file_name", "")
        if not fname:
            continue
        src_path = files.join(images_dir, fname)
        if not files.exists(src_path):
            continue

        stem = files.stem(fname)
        if stem.lower() in seen_stems:
            raise ValueError(f"duplicate sample stem in COCO images: {stem}")
        seen_stems.add(stem.lower())

        w = float(im_info.get("width", 0))
        h = float(im_info.get("height", 0))
        if w <= 0 or h <= 0:
            raise ValueError(f"invalid dimensions ({w}x{h}) for image {fname}")

        lines: list[str] = []
        for ann in ann_by_img.get(im_info["id"], []):
            cat_id = ann.get("category_id")
            if cat_id not in cat_to_yolo:
                continue
            cid = cat_to_yolo[cat_id]

            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            x, y, bw_px, bh_px = [float(val) for val in bbox]

            xmin = max(0.0, min(x, w))
            ymin = max(0.0, min(y, h))
            xmax = max(0.0, min(x + bw_px, w))
            ymax = max(0.0, min(y + bh_px, h))

            box_w = xmax - xmin
            box_h = ymax - ymin
            if box_w <= 0 or box_h <= 0:
                continue

            xc = ((xmin + xmax) / 2.0) / w
            yc = ((ymin + ymax) / 2.0) / h
            norm_w = box_w / w
            norm_h = box_h / h

            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            lines.append(f"{cid} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}")

        # Copy original image bytes without re-encoding
        dest_img = files.join(images_out, f"{stem}{files.extension(fname).lower()}")
        copy_entry(src_path, dest_img)

        # Write label file (empty if no detections -> negative image)
        dest_lbl = files.join(labels_out, f"{stem}.txt")
        files.save(dest_lbl, lines if lines else "", overwrite=True, verbose=False)
        processed_count += 1

    if processed_count == 0:
        warnings.warn("No matching COCO samples were converted", UserWarning, stacklevel=3)
    return classes


def convert(
    source: str | PathLike[str] | None = None,
    target: str | PathLike[str] = "",
    *,
    src_format: str = "voc",
    dst_format: str = "yolo",
    images: str | PathLike[str] | None = None,
    annotations: str | PathLike[str] | None = None,
    classes: list[str] | dict[int, str] | None = None,
    image_ids: Sequence[str] | None = None,
    include_difficult: bool = True,
    include_crowd: bool = False,
    voc_1_based: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
    **kwargs,
) -> Path:
    """Convert Pascal VOC XML or COCO instances JSON to a flat YOLO dataset.

    Parameters
    ----------
    source : str or os.PathLike[str], optional
        Root directory or archive containing the source dataset.
    target : str or os.PathLike[str]
        Destination directory or ``.zip`` archive for the flat YOLO dataset.
    src_format : {'voc', 'coco'}, default='voc'
        Source annotation format.
    dst_format : {'yolo'}, default='yolo'
        Destination dataset format. Currently only ``'yolo'`` is supported.
    images : str or os.PathLike[str], optional
        Explicit path to the directory containing source image files.
    annotations : str or os.PathLike[str], optional
        Explicit path to the directory of VOC XML files or path to COCO JSON file.
    classes : list[str] or dict[int, str], optional
        Ordered sequence or mapping of target class names. If omitted, discovered
        automatically from the source annotations.
    image_ids : Sequence[str], optional
        Subset of image identifiers or file stems to convert.
    include_difficult : bool, default=True
        Whether to retain objects marked as difficult in Pascal VOC XML.
    include_crowd : bool, default=False
        Whether to retain annotations marked with ``iscrowd=1`` in COCO JSON.
    voc_1_based : bool, default=False
        Whether source Pascal VOC coordinates are 1-based (requiring 1-pixel offset).
    overwrite : bool, default=False
        Whether to overwrite an existing target directory or archive.
    verbose : bool, default=False
        Whether to display progress information.
    **kwargs
        Additional options. Passing ``tree`` raises an explicit migration error.

    Returns
    -------
    pathlib.Path
        Path to the published flat dataset directory or ``.zip`` archive.

    Raises
    ------
    TypeError
        If ``tree`` is passed, directing users to ``datasets.flatten()``.
    ValueError
        If format, paths, dimensions, or annotations violate the conversion contract.
    FileNotFoundError
        If source directories or annotations do not exist.
    FileExistsError
        If target exists and ``overwrite=False``.

    Examples
    --------
    >>> from klygo import datasets
    >>> datasets.convert(source="voc_dir", target="flat_yolo", src_format="voc")
    """
    if "tree" in kwargs:
        raise TypeError(
            "datasets.convert() is now dedicated to annotation format conversion "
            "(VOC/COCO -> YOLO). For dataset layout flattening with directory trees, "
            "use datasets.flatten()."
        )
    if not target:
        raise ValueError("target destination must be specified")

    src_fmt = str(src_format).strip().lower()
    if src_fmt not in _SUPPORTED_SRC:
        raise ValueError(f"unsupported source format: {src_format!r}; supported: {sorted(_SUPPORTED_SRC)}")
    dst_fmt = str(dst_format).strip().lower()
    if dst_fmt not in _SUPPORTED_DST:
        raise ValueError(f"unsupported destination format: {dst_format!r}; supported: {sorted(_SUPPORTED_DST)}")

    validators.validate_type(overwrite, bool, "overwrite")
    parsed_classes = _parse_classes(classes)

    images_dir, ann_path = _resolve_paths(source, images, annotations, src_fmt)

    with dataset_workspace("convert") as workspace:
        if src_fmt == "voc":
            final_classes = _convert_voc(
                images_dir,
                ann_path,
                workspace,
                parsed_classes,
                image_ids,
                include_difficult=include_difficult,
                voc_1_based=voc_1_based,
            )
        else:
            final_classes = _convert_coco(
                images_dir,
                ann_path,
                workspace,
                parsed_classes,
                image_ids,
                include_crowd=include_crowd,
            )

        data_yaml = {
            "path": ".",
            "train": "images",
            "val": "images",
            "nc": len(final_classes),
            "names": final_classes,
        }
        config.save(files.join(workspace, "data.yaml"), data_yaml, overwrite=True, verbose=False)
        published = write_output(workspace, target, overwrite=overwrite, verbose=verbose)

    return published


__all__ = ["convert"]
