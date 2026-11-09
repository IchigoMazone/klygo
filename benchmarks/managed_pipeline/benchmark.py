"""Measure input preparation and optional real YOLO throughput on one clip.

Run from the repository root with PYTHONPATH=. so the selected checkout is used.
Runtime reports belong under artifacts/, not in the source tree.
"""

import argparse
import gc
import hashlib
import statistics
import time

import cv2
import numpy as np

from klygo import files, media, models
from klygo.models.detection.base import _prepared_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--output", default="artifacts/managed-pipeline/report.json")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--scope", choices=("all", "decode", "yolo"), default="all")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()
    if args.iterations < 1 or args.batch < 1:
        parser.error("iterations and batch must be positive")
    cv2.setNumThreads(1)
    reports = {}

    def measure(name, run):
        run()  # Warm the filesystem, codec, and model independently of timing.
        timings = []
        signature = None
        for _ in range(args.iterations):
            gc.collect()
            started = time.perf_counter()
            count, digest = run()
            timings.append(time.perf_counter() - started)
            if signature is not None and signature != digest:
                raise RuntimeError(f"Non-deterministic output in {name}")
            signature = digest
        median = statistics.median(timings)
        reports[name] = {"frames": count, "median_seconds": median,
                         "fps": count / median, "signature": signature,
                         "seconds": timings}
        print(f"{name}: {count / median:.2f} FPS", flush=True)
        files.save(args.output, reports, overwrite=True, verbose=False)

    def decode(workers, prefetch, stride):
        source = media.load(args.source, indexed=True, backend="opencv", verbose=False).sample(stride)
        count = 0
        digest = hashlib.sha256()
        try:
            for part, prepared in _prepared_batches(source, args.batch, workers, prefetch, "Ultralytics"):
                for original, pixels in zip(part, prepared):
                    digest.update(str(original.frame_index).encode())
                    digest.update(pixels.tobytes())
                    count += 1
        finally:
            source.close()
        return count, digest.hexdigest()

    if args.scope in ("all", "decode"):
        for stride in (1, 3):
            for workers, prefetch in ((0, 0), (1, 1), (4, 2)):
                measure(f"decode_stride{stride}_workers{workers}_prefetch{prefetch}",
                        lambda w=workers, p=prefetch, s=stride: decode(w, p, s))

    if args.weights and args.scope in ("all", "yolo"):
        import torch
        import ultralytics

        torch.set_num_threads(2)
        options = {"device": args.device, "imgsz": args.imgsz, "threshold": .25, "iou": .7}
        managed = models.load("yolov8n.pt", model_weights=args.weights, **options)

        raw_model = ultralytics.YOLO(args.weights)

        def raw_predict():
            if args.device != "cpu":
                torch.cuda.synchronize(args.device)
            results = raw_model.predict(args.source, device=args.device, imgsz=args.imgsz,
                                        batch=args.batch, conf=.25, iou=.7, verbose=False)
            digest = hashlib.sha256()
            for frame in results:
                digest.update(np.round(frame.boxes.xyxy.cpu().numpy().astype(np.float64), 3).tobytes())
                digest.update(np.round(frame.boxes.conf.cpu().numpy().astype(np.float64), 5).tobytes())
                digest.update(frame.boxes.cls.cpu().numpy().astype(np.int32).tobytes())
            if args.device != "cpu":
                torch.cuda.synchronize(args.device)
            return len(results), digest.hexdigest()

        measure("yolo_raw", raw_predict)

        def predict(detector, **execution):
            if args.device != "cpu":
                torch.cuda.synchronize(args.device)
            results = detector.predict(args.source, verbose=False, **execution)
            digest = hashlib.sha256()
            count = 0
            for frame in results:
                digest.update(np.round(frame._xyxy, 3).tobytes())
                digest.update(np.round(frame._scores, 5).tobytes())
                digest.update(frame._class_ids.tobytes())
                count += 1
            if args.device != "cpu":
                torch.cuda.synchronize(args.device)
            return count, digest.hexdigest()

        if args.native:
            native = models.load("yolov8n.pt", model_weights=args.weights, **options)
            measure("yolo_native_batch1", lambda: predict(native, native=True))
        for workers, prefetch in ((1, 1), (4, 2)):
            measure(f"yolo_managed_workers{workers}_prefetch{prefetch}",
                    lambda w=workers, p=prefetch: predict(managed, batch=args.batch, workers=w, prefetch=p))
        reports["environment"] = {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                                   "device": args.device, "torch_threads": torch.get_num_threads()}
    reports["settings"] = vars(args)
    files.save(args.output, reports, overwrite=True, verbose=False)


if __name__ == "__main__":
    main()
