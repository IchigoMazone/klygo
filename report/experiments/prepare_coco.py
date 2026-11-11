"""Freeze a COCO validation sample before inference; download pilot images only by default."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "klygo/artifacts/report/data"
ANNOTATION_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def download(url, destination):
    """Resume an interrupted public download; never accept a partial file as complete."""
    destination = Path(destination)
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"User-Agent": "Klygo-Academic-Evaluation/1.0"})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                resumed = response.status == 206 and offset > 0
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an inconsistent byte range")
                expected = int(response.headers.get("Content-Length", 0))
                received = 0
                with partial.open("ab" if resumed else "wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        received += len(chunk)
                if expected and received != expected:
                    raise OSError(f"Incomplete download: {received} of {expected} bytes")
            partial.replace(destination)
            print(f"Downloaded {destination.name}: {destination.stat().st_size} bytes", flush=True)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2)


def prepare(sample_size=500, image_limit=16, seed=20261105, annotation_json_url=None,
            image_base_url="https://s3.amazonaws.com/images.cocodataset.org/val2017"):
    DATA.mkdir(parents=True, exist_ok=True)
    annotations = DATA / "instances_val2017.json"
    if annotation_json_url and not annotations.exists():
        download(annotation_json_url, annotations)
    if not annotations.exists():
        archive = DATA / "annotations_trainval2017.zip"
        download(ANNOTATION_URL, archive)
        # Read one explicitly named member, without extracting arbitrary paths.
        with zipfile.ZipFile(archive) as bundle, bundle.open("annotations/instances_val2017.json") as source:
            with annotations.with_suffix(".json.part").open("wb") as target:
                shutil.copyfileobj(source, target)
        annotations.with_suffix(".json.part").replace(annotations)
    source = json.loads(annotations.read_text(encoding="utf-8"))
    assert len(source["images"]) == 5000 and len(source["categories"]) == 80
    selected = random.Random(seed).sample(sorted(item["id"] for item in source["images"]), sample_size)
    ids = set(selected)
    subset = {key: value for key, value in source.items() if key not in ("images", "annotations")}
    subset["images"] = [item for item in source["images"] if item["id"] in ids]
    subset["annotations"] = [item for item in source["annotations"] if item["image_id"] in ids]
    manifest_path = DATA / "manifest.json"
    manifest = {
        "dataset": "COCO val2017", "sample_size": sample_size, "seed": seed,
        "selection": "random.Random(seed).sample(sorted(image_ids), sample_size)",
        "python": platform.python_version(), "image_ids_in_sample_order": selected,
        "annotation_url": annotation_json_url or ANNOTATION_URL, "annotation_sha256": sha256(annotations),
        "image_base_url": image_base_url,
        "annotation_count": len(subset["annotations"]), "image_sha256": {},
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("sample_size", "seed", "image_ids_in_sample_order", "annotation_sha256"):
            if existing[key] != manifest[key]:
                raise ValueError(f"Refusing to change frozen manifest field: {key}")
        manifest["image_sha256"] = existing.get("image_sha256", {})
        manifest["annotation_url"] = existing["annotation_url"]
        manifest["image_base_url"] = existing["image_base_url"]
    write_json(manifest_path, manifest)
    write_json(DATA / "instances_sample.json", subset)
    images = {item["id"]: item for item in source["images"]}
    def prepare_image(image_id):
        item = images[image_id]
        filename = Path(item["file_name"]).name
        path = DATA / "images" / filename
        download(f"{manifest['image_base_url']}/{filename}", path)
        digest = sha256(path)
        if str(image_id) in manifest["image_sha256"] and manifest["image_sha256"][str(image_id)] != digest:
            raise ValueError(f"Image changed after manifest was frozen: {image_id}")
        return str(image_id), digest
    with ThreadPoolExecutor(max_workers=6) as pool:
      for image_id, digest in pool.map(prepare_image, selected[:image_limit or sample_size]):
        manifest["image_sha256"][image_id] = digest
        write_json(manifest_path, manifest)
    print(f"Frozen {sample_size} images; downloaded {len(manifest['image_sha256'])} images", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--image-limit", type=int, default=16, help="0 downloads the whole frozen sample")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--annotation-json-url", help="Optional documented mirror for the original JSON")
    parser.add_argument("--image-base-url", default="https://s3.amazonaws.com/images.cocodataset.org/val2017")
    args = parser.parse_args()
    prepare(args.sample_size, args.image_limit, args.seed, args.annotation_json_url, args.image_base_url)
