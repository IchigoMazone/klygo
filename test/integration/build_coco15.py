"""Build a 15-class, evaluation-only YOLO dataset from the official COCO128 zip."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image


SOURCE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
SOURCE_SHA256 = "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e"
SOURCE = Path(__file__).resolve().parent / "generated" / "coco128.zip"
CLASSES = (
    (0, "person"),
    (2, "car"),
    (9, "traffic light"),
    (20, "elephant"),
    (25, "umbrella"),
    (26, "handbag"),
    (39, "bottle"),
    (40, "wine glass"),
    (41, "cup"),
    (43, "knife"),
    (44, "spoon"),
    (45, "bowl"),
    (56, "chair"),
    (73, "book"),
    (77, "teddy bear"),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(source: Path = SOURCE) -> Path:
    """Download the pinned official archive only when explicitly requested."""
    import requests

    source = Path(source)
    if source.exists():
        if _sha256(source.read_bytes()) != SOURCE_SHA256:
            raise ValueError(f"Existing COCO128 archive has an unexpected hash: {source}")
        return source
    response = requests.get(SOURCE_URL, timeout=60)
    response.raise_for_status()
    if _sha256(response.content) != SOURCE_SHA256:
        raise ValueError("Downloaded COCO128 archive differs from the pinned source")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(response.content)
    return source


def build(output: Path, source: Path = SOURCE) -> Path:
    """Copy all 128 real images and remap only the selected 15 label classes."""
    output, source = Path(output), Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"COCO128 archive missing: {source}; use --download")
    source_bytes = source.read_bytes()
    if _sha256(source_bytes) != SOURCE_SHA256:
        raise ValueError("COCO128 source hash differs from the pinned release")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Dataset output is not empty: {output}")
    image_dir, label_dir = output / "images", output / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    remap = {source_id: target_id for target_id, (source_id, _) in enumerate(CLASSES)}
    counts, positive_images = Counter(), Counter()
    items = []
    with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
        archive_names = set(archive.namelist())
        image_names = sorted(
            name for name in archive.namelist()
            if name.startswith("coco128/images/train2017/") and name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if len(image_names) != 128:
            raise ValueError(f"Expected 128 COCO128 images, found {len(image_names)}")
        for image_name in image_names:
            image_id = Path(image_name).stem
            label_name = f"coco128/labels/train2017/{image_id}.txt"
            image_bytes = archive.read(image_name)
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.verify()
            (image_dir / f"{image_id}.jpg").write_bytes(image_bytes)
            selected = []
            label_lines = archive.read(label_name).decode("utf-8").splitlines() if label_name in archive_names else []
            for line in label_lines:
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Malformed COCO128 label: {label_name}")
                original_id = int(parts[0])
                coordinates = list(map(float, parts[1:]))
                if not all(0 <= value <= 1 for value in coordinates) or coordinates[2] <= 0 or coordinates[3] <= 0:
                    raise ValueError(f"Invalid COCO128 coordinates: {label_name}")
                if original_id in remap:
                    class_id = remap[original_id]
                    selected.append(" ".join([str(class_id), *parts[1:]]))
                    counts[class_id] += 1
            for class_id in {int(row.split()[0]) for row in selected}:
                positive_images[class_id] += 1
            (label_dir / f"{image_id}.txt").write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
            items.append({"id": image_id, "image_sha256": _sha256(image_bytes), "boxes": len(selected)})
        (output / "LICENSE").write_bytes(archive.read("coco128/LICENSE"))
    if len(counts) != len(CLASSES):
        raise ValueError("At least one selected class has no boxes")
    import yaml

    (output / "data.yaml").write_text(
        yaml.safe_dump({"names": [name for _, name in CLASSES], "nc": len(CLASSES)}, sort_keys=False),
        encoding="utf-8",
    )
    manifest = {
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "source_archive": str(source.resolve()),
        "source_license_file": "LICENSE",
        "purpose": "integration and evaluation only; COCO128 is not an independent train/validation benchmark",
        "image_count": len(items),
        "negative_image_count": sum(item["boxes"] == 0 for item in items),
        "classes": [
            {"id": index, "name": name, "source_coco_id": source_id,
             "boxes": counts[index], "positive_images": positive_images[index]}
            for index, (source_id, name) in enumerate(CLASSES)
        ],
        "items": items,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download(args.source)
    result = build(args.output, args.source)
    print(result)
