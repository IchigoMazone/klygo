"""Fresh-process raw/native/managed benchmark of a pinned Klygo revision."""
from extension_common import *
import argparse
import gc
import subprocess
import time
import threading
import shutil
import hashlib
from datetime import datetime, timezone
import numpy as np
import torch
import psutil
from ultralytics import YOLO, YOLOWorld
from klygo import models

MODES = ["raw", "raw_stream", "native", "native_backend_stream", "managed", "managed_stream"]
EXEC = OUT / "execution_conf025"

def prepare(destination=EXEC):
    folder = destination / "input"
    folder.mkdir(parents=True, exist_ok=True)
    source = load(DATA / "instances_sample.json")
    records = {item["id"]: item for item in source["images"]}
    ids = load(DATA / "manifest.json")["image_ids_in_sample_order"][:100]
    hashes = load(DATA / "manifest.json")["image_sha256"]
    for index, image_id in enumerate(ids):
        target = folder / f"{index:04d}_{image_id:012d}.jpg"
        if not target.exists():
            shutil.copyfile(DATA / "images" / records[image_id]["file_name"], target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != hashes[str(image_id)]:
            raise ValueError(f"Input checksum mismatch: {image_id}")
    return folder

def worker(name, mode, workers=1, prefetch=1, destination=EXEC):
    folder = prepare(destination)
    weights = WEIGHTS / ("yolov8n.pt" if name == "yolo" else "yolov8s-worldv2.pt")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    kwargs = dict(device="cuda:0", imgsz=640, half=False, max_det=100, conf=0.25, iou=0.7,
                  batch=1, verbose=False, save=False)
    if mode.startswith("raw"):
        model = (YOLO if name == "yolo" else YOLOWorld)(str(weights))
        call = lambda: model.predict(str(folder), stream=mode == "raw_stream", **kwargs)
    else:
        registry = load(SOURCE / "klygo/models/models.json")
        schema = registry["yolov8?.pt" if name == "yolo" else "yolov8*-world*.pt"].copy()
        schema.pop("option", None)
        schema["model_id"] = str(weights)
        schema["model"] = {"device": "cuda:0", "imgsz": 640, "half": False, "max_det": 100}
        schema["post"] = {"threshold": 0.25, "iou": 0.7}
        model = models.load(schema)
        if mode in {"native", "native_backend_stream"}:
            call = lambda: model.predict(str(folder), native=True,
                         model_stream=mode == "native_backend_stream", verbose=False)
        else:
            call = lambda: model.predict(str(folder), native=False, stream=mode == "managed_stream",
                                         workers=workers, prefetch=prefetch, verbose=False)
    def consume(record=False):
        start = time.perf_counter()
        result = call()
        first = None
        count = 0
        signatures = []
        for item in result:
            count += 1
            if first is None:
                torch.cuda.synchronize()
                first = time.perf_counter()-start
            if record:
                if mode.startswith("raw"):
                    arr = item.boxes.data.cpu().numpy()
                    signatures.append(arr.tolist())
                else:
                    signatures.append([[*box.box, box.score, box.class_id] for box in item])
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        close = getattr(result, "close", None)
        if callable(close): close()
        assert count == 100, count
        return elapsed, first, signatures
    with torch.inference_mode():
        consume()
        rows = []
        proc = psutil.Process()
        for repeat in range(3):
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            rss = [proc.memory_info().rss]
            stop = threading.Event()
            def sample():
                while not stop.wait(0.01): rss.append(proc.memory_info().rss)
            thread = threading.Thread(target=sample, daemon=True)
            thread.start()
            start_alloc = torch.cuda.memory_allocated()
            elapsed, first, _ = consume()
            stop.set(); thread.join()
            rows.append({"total_s": elapsed, "first_s": first, "fps": 100/elapsed,
                         "peak_rss_mib": max(rss)/2**20, "delta_rss_mib": (max(rss)-rss[0])/2**20,
                         "peak_vram_mib": torch.cuda.max_memory_allocated()/2**20,
                         "delta_vram_mib": (torch.cuda.max_memory_allocated()-start_alloc)/2**20})
            print(name, mode, repeat, elapsed, flush=True)
        _, _, predictions = consume(record=True)
    import importlib.metadata
    suffix = f"_w{workers}_p{prefetch}" if (workers,prefetch)!=(1,1) else ""
    save(destination / f"{name}_{mode}{suffix}.json", {"revision": REVISION, "mode": mode, "confidence": 0.25,
        "workers": workers, "prefetch": prefetch, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": name, "rows": rows, "predictions": predictions, "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "ultralytics": importlib.metadata.version("ultralytics"),
        "input_files": [p.name for p in sorted(folder.glob("*.jpg"))],
        "scope": "100 JPEG files, batch 1, prepared model, complete source warmup, decode through public output; parity serialized in separate untimed pass"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["yolo", "world"])
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--prefetch", type=int, default=1)
    parser.add_argument("--output-tag", default="execution_conf025")
    args = parser.parse_args()
    if Path(args.output_tag).name != args.output_tag:
        raise ValueError("output-tag must be a directory name")
    destination = OUT / args.output_tag
    if args.model:
        worker(args.model, args.mode, args.workers, args.prefetch, destination)
    else:
        for name in ["yolo", "world"]:
            for mode in MODES:
                target = destination / f"{name}_{mode}.json"
                if not target.exists():
                    subprocess.run([sys.executable, "-X", "utf8", "-u", __file__, "--model", name, "--mode", mode,
                                    "--output-tag",args.output_tag], check=True)
