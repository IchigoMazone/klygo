"""Execute the preregistered 500-image evaluation and paired lifecycle timing."""
import argparse
import gc
import json
import time

import numpy as np
import torch
from PIL import Image

from prepare_coco import DATA, sha256, write_json
from run_pilot import ARTIFACTS, environment, load_detector, run


def synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def signature(frame):
    return [(box.label, *map(float, box.box), float(box.score)) for box in frame.objects]


def equivalent(first, second):
    a, b = signature(first), signature(second)
    return len(a) == len(b) and all(x[0] == y[0] and np.allclose(x[1:], y[1:], rtol=1e-5, atol=1e-4)
                                  for x, y in zip(a, b))


def benchmark(name, device):
    folder = ARTIFACTS / "main" / name / "timing_controlled"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "timing_summary.json").exists():
        raise FileExistsError(f"Timing evidence exists: {folder}")
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    ids = manifest["image_ids_in_sample_order"][:100]
    source = json.loads((DATA / "instances_sample.json").read_text(encoding="utf-8"))
    records = {item["id"]: item for item in source["images"]}
    prepared = []
    for image_id in ids:
        path = DATA / "images" / records[image_id]["file_name"]
        if sha256(path) != manifest["image_sha256"][str(image_id)]:
            raise ValueError(f"Checksum mismatch: {image_id}")
        with Image.open(path) as image:
            prepared.append((image_id, image.convert("RGB")))
    detector = load_detector(name, device)
    # Both paths include backend preprocessing, inference, postprocessing and Detection packaging.
    # forward bypasses only Detector.predict lifecycle; it is not an unmodified native backend baseline.
    paths = {"forward": lambda image: detector.forward([image], prompt=None)[0],
             "predict": lambda image: detector.predict(image, verbose=False)[0]}
    rows, mismatch_ids = [], []
    try:
        for _ in range(5):
            for function in paths.values():
                with torch.inference_mode():
                    function(prepared[0][1])
        for repeat in range(3):
            for position, (image_id, image) in enumerate(prepared):
                order = ("forward", "predict") if (position + repeat) % 2 == 0 else ("predict", "forward")
                frames, measured = {}, {}
                for path in order:
                    synchronize(device)
                    begin = time.perf_counter()
                    with torch.inference_mode():
                        frames[path] = paths[path](image)
                    synchronize(device)
                    measured[path] = (time.perf_counter() - begin) * 1000
                equal = equivalent(frames["forward"], frames["predict"])
                if not equal:
                    mismatch_ids.append({"repeat": repeat, "image_id": image_id})
                rows.append({"repeat": repeat, "image_id": image_id,
                             "forward_ms": measured["forward"], "predict_ms": measured["predict"],
                             "equivalent": equal, "first_path": order[0]})
            print(f"Timing {name}: repeat {repeat + 1}/3 complete", flush=True)
        summary = {"image_count": 100, "repeats": 3, "paired_calls": len(rows),
            "mismatches": mismatch_ids,
            "scope": "Same backend/model/input, FP32, torch.inference_mode and Detection outputs; predict lifecycle versus direct forward, not pure native backend overhead",
            "environment": environment()}
        for path in paths:
            values = np.array([row[path + "_ms"] for row in rows])
            summary[path] = {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
                "p90_ms": float(np.percentile(values, 90)), "sequential_fps": float(1000 / values.mean()),
                "repeat_mean_ms": [float(np.mean([row[path + "_ms"] for row in rows if row["repeat"] == repeat])) for repeat in range(3)]}
        differences = np.array([row["predict_ms"] - row["forward_ms"] for row in rows])
        summary["paired_difference"] = {"mean_ms": float(differences.mean()),
            "median_ms": float(np.median(differences)),
            "relative_mean_percent": float(100 * differences.mean() / summary["forward"]["mean_ms"])}
        write_json(folder / "paired_timing.json", rows)
        write_json(folder / "timing_summary.json", summary)
        if mismatch_ids:
            raise RuntimeError("Output mismatch: do not interpret this timing as equivalent-path overhead")
    finally:
        detector.unload()
        del detector
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["yolo", "world", "faster"], choices=["yolo", "world", "faster"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if len(manifest["image_sha256"]) != 500:
        raise RuntimeError("Prepare and verify all 500 images before main evaluation")
    for name in args.models:
        if not args.timing_only and not (ARTIFACTS / "main" / name / "metrics.json").exists():
            run(name, 500, args.device, phase="main")
        benchmark(name, args.device)
