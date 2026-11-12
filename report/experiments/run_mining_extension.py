"""Benchmark actual mining APIs, export YOLO data and freeze train/val/test splits."""
from extension_common import *
import time
import hashlib
import shutil
import numpy as np
from klygo import mining, outputs, datasets

def export_trainable(data, export):
    if not (export / "manifest.json").exists():
        data.export(str(export), format="yolo")
    (export / "images").mkdir(exist_ok=True)
    manifest = load(export / "manifest.json")
    assert [str(item["source_id"]) for item in manifest] == [str(frame.source_id) for frame in data]
    for item in manifest:
        source = Path(item["source_path"])
        target = export / "images" / (Path(item["label_file"]).stem + source.suffix)
        if not target.exists():
            shutil.copyfile(source, target)

def frames_from_coco(ids, source):
    categories = sorted(source["categories"], key=lambda item: item["id"])
    names = dict(enumerate(item["name"] for item in categories))
    mapping = {item["id"]: index for index, item in enumerate(categories)}
    records = {item["id"]: item for item in source["images"]}
    annotations = {key: [] for key in ids}
    for item in source["annotations"]:
        if item["image_id"] in annotations and not item.get("iscrowd", 0):
            annotations[item["image_id"]].append(item)
    frames = []
    for image_id in ids:
        record = records[image_id]
        path = DATA / "images" / record["file_name"]
        boxes = []
        for item in annotations[image_id]:
            x, y, w, h = item["bbox"]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(record["width"], x+w), min(record["height"], y+h)
            if x2 > x1 and y2 > y1:
                cid = mapping[item["category_id"]]
                boxes.append(outputs.Box([x1, y1, x2, y2], label=names[cid], class_id=cid))
        frames.append(outputs.Detection(boxes, source_id=str(image_id), source_path=str(path),
                      image_size=(record["width"], record["height"]), names=names))
    return outputs.Detections(frames, names=names)

def run():
    folder = OUT / "mining"
    if (folder / "summary.json").exists():
        raise FileExistsError("Completed mining results already exist")
    ids = load(DATA / "manifest.json")["image_ids_in_sample_order"]
    split = {"train": ids[:300], "val": ids[300:400], "test": ids[400:500]}
    assert not set(split["train"]) & set(split["test"])
    save(folder / "protocol.json", {"revision": REVISION, "splits": split, "limit": 150,
         "selection_seed": 42, "timing_repeats": 3, "warmup": 1,
         "note": "Exploratory split of COCO val2017, not the official COCO training benchmark; human annotations, non-crowd only."})
    source = load(DATA / "instances_sample.json")
    data = frames_from_coco(split["train"], source)
    before = mining.stats(data, quality=False).to_dict()
    signatures = [f.to_dict() for f in data]
    operations = {
        "full": lambda: data,
        "random": lambda: mining.select(data, method="random", limit=150, seed=42),
        "weighted": lambda: mining.select(data, method="weighted", limit=150),
        "diversity": lambda: mining.select(data, method="diversity", limit=150,
                                  cluster_result=mining.cluster(data, clusters=150, seed=42)),
        "rare_first": lambda: mining.select(data, method="rare_first", limit=150),
        "coverage": lambda: mining.select(data, method="coverage", limit=150, per_class=3),
        "quota": lambda: mining.select(data, method="quota", limit=150, per_class=3),
        "quality": lambda: mining.select(data, method="quality", limit=150, threshold=0.5),
        "boxes": lambda: mining.select(data, method="boxes", limit=150, min_boxes=2),
        "class": lambda: mining.select(data, method="class", limit=150, classes=["person"]),
        "balance": lambda: mining.balance(data, target="median", count_by="images"),
    }
    results = {}
    for name, function in operations.items():
        print("Mining", name, flush=True)
        function()
        times = []
        for _ in range(3):
            start = time.perf_counter()
            selected = function()
            times.append((time.perf_counter()-start)*1000)
        selected_ids = [str(frame.source_id) for frame in selected]
        assert len(selected_ids) == len(set(selected_ids))
        assert set(selected_ids) <= {str(key) for key in split["train"]}
        export = folder / "datasets" / name
        start = time.perf_counter()
        export_trainable(selected, export)
        export_ms = (time.perf_counter()-start)*1000
        reloaded = datasets.load(str(export), format="yolo", verbose=False)
        after = mining.stats(selected, quality=False).to_dict()
        check = mining.stats(reloaded, quality=False).to_dict()
        assert (check["images"], check["boxes"], check["boxes_per_class"]) == (after["images"], after["boxes"], after["boxes_per_class"])
        results[name] = {"selection_ms": times, "median_ms": float(np.median(times)),
                         "export_ms": export_ms, "ids": selected_ids, "stats": after,
                         "metadata": selected.metadata, "roundtrip_pass": True}
        mining.compare(data, selected).save(str(folder / f"comparison_{name}.json"))
    assert signatures == [f.to_dict() for f in data], "Mining mutated source"
    for name in ("val", "test"):
        export_trainable(frames_from_coco(split[name], source), folder / "datasets" / name)
    clustered = mining.cluster(data, clusters=12, seed=42)
    clustered.save(str(folder / "clusters12.json"))
    mining.score(data).save(str(folder / "scores.json"))
    save(folder / "summary.json", {"revision": REVISION, "before": before, "methods": results,
                                   "input_unchanged": True})
    print("Mining complete", flush=True)

if __name__ == "__main__":
    run()
