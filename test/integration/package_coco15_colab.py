"""Bundle the current Klygo checkout and COCO15 dataset for Colab/Drive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
_PRIMARY_DATASET = REPO / "artifacts" / "integration" / "coco15"
_FALLBACK_DATASET = Path(__file__).resolve().parent / "generated" / "coco15"
DATASET = _PRIMARY_DATASET if _PRIMARY_DATASET.is_dir() else _FALLBACK_DATASET

_PRIMARY_BUNDLE = REPO / "artifacts" / "integration" / "bundles" / "klygo_coco15_bundle.zip"
_FALLBACK_BUNDLE = Path(__file__).resolve().parent / "generated" / "klygo_coco15_bundle.zip"
DEFAULT_OUTPUT = _PRIMARY_BUNDLE if _PRIMARY_BUNDLE.parent.is_dir() else _FALLBACK_BUNDLE


def package(output: Path = DEFAULT_OUTPUT, dataset: Path = DATASET) -> Path:
    """Package source and images without changing or uploading the workspace."""
    output, dataset = Path(output), Path(dataset)
    if not (dataset / "manifest.json").is_file():
        raise FileNotFoundError("Build the COCO15 dataset before packaging")
    package_files = [REPO / "pyproject.toml", REPO / "README.md"]
    package_files.extend(
        path for path in (REPO / "klygo").rglob("*")
        if path.is_file() and (path.suffix == ".py" or path.name == "models.json")
    )
    dataset_files = [path for path in dataset.rglob("*") if path.is_file()]
    manifest = {
        "dataset_source_sha256": json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))["source_sha256"],
        "dataset_image_count": len(list((dataset / "images").glob("*.jpg"))),
        "package_files": len(package_files),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(package_files):
            bundle.write(path, "project/" + path.relative_to(REPO).as_posix())
        for path in sorted(dataset_files):
            bundle.write(path, "dataset/" + path.relative_to(dataset).as_posix())
        bundle.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output) as bundle:
        if bundle.testzip() is not None:
            raise OSError("Bundle failed ZIP integrity check")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = package(args.output)
    print(path)
    print("sha256", hashlib.sha256(path.read_bytes()).hexdigest())
