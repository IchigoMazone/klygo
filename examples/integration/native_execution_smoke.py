"""Run real pretrained detection through all four framework families.

This is a correctness smoke run, not a labeled accuracy or GPU FPS benchmark.
Run from the repository root with PYTHONPATH=. Runtime files stay in artifacts/.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image

from klygo import files, media, models, outputs, visual


class HuggingFaceDetector(models.Detector):
    """Own the native DETR processor, forward call and postprocessing."""

    def __init__(self, device):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        super().__init__(backend="Hugging Face", model_id="facebook/detr-resnet-50")
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.model_id, use_pretrained_backbone=False,
        ).to(device).eval()
        self.names = dict(self.model.config.id2label)

    def forward(self, images, prompt=None):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        raw = self.model(**inputs)
        formatted = self.processor.post_process_object_detection(
            raw, threshold=0.25, target_sizes=[image.size[::-1] for image in images],
        )
        for item in formatted:
            ids = item["labels"].tolist()
            item["class_ids"] = ids
            item["labels"] = [self.names[index] for index in ids]
        return outputs.build_detections(images, formatted, names=self.names)


class TorchvisionDetector(models.Detector):
    """Use a small pretrained native Torchvision detector."""

    def __init__(self, device):
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )

        super().__init__(backend="Torchvision", model_id="ssdlite320_mobilenet_v3_large")
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=weights).to(device).eval()
        self.names = dict(enumerate(weights.meta["categories"]))

    def forward(self, images, prompt=None):
        inputs = [media.to_tensor(image).to(self.device) for image in images]
        raw = self.model(inputs)
        formatted = []
        for item in raw:
            keep = item["scores"] >= 0.25
            ids = item["labels"][keep].tolist()
            formatted.append({
                "boxes": item["boxes"][keep], "scores": item["scores"][keep],
                "class_ids": ids, "labels": [self.names[index] for index in ids],
            })
        return outputs.build_detections(images, formatted, names=self.names)


class KerasDetector(models.Detector):
    """Exercise native KerasHub and the retained optional Detector helpers."""

    def __init__(self, device):
        import keras_hub

        super().__init__(backend="Keras", model_id="dfine_nano_coco")
        self.model = keras_hub.models.ObjectDetector.from_preset(self.model_id)
        if hasattr(self.model, "to"):
            self.model.to(device)
        self.names = {}

    def forward(self, images, prompt=None):
        inputs = self.cast_inputs(images)
        raw = self.run_inference(inputs)
        formatted = self.format_results(raw)
        for item in formatted:
            keep = [i for i, score in enumerate(item["scores"]) if score >= 0.25]
            for key in ("boxes", "scores", "labels", "class_ids"):
                item[key] = [item[key][i] for i in keep]
        return outputs.build_detections(images, formatted, names=self.names)


def snapshot(result):
    """Check and retain outputs for reproducible managed/stream comparisons."""
    frames = []
    for frame in result:
        boxes = np.asarray(frame.boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(frame.scores, dtype=np.float64)
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise AssertionError("Non-finite detection output")
        frames.append({"boxes": boxes.tolist(), "scores": scores.tolist(),
                       "labels": list(frame.labels)})
    return frames


def run_one(backend, source, root):
    import torch

    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if backend == "ultralytics":
        weights = files.join(root, "cache", "yolov8n.pt")
        if not files.is_file(weights):
            files.download(
                "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
                files.parent(weights), verbose=False,
            )
        detector = models.load("yolov8n.pt", model_weights=str(weights), device=device, imgsz=320)
    elif backend == "huggingface":
        detector = HuggingFaceDetector(device)
    elif backend == "torchvision":
        detector = TorchvisionDetector(device)
    else:
        detector = KerasDetector(device)

    positive = media.to_pil(media.load(source, verbose=False)[0])
    negative = Image.new("RGB", positive.size, "black")
    images = [positive, negative]
    started = time.perf_counter()
    result = detector.predict(images, workers=0, verbose=False)
    managed = snapshot(result)
    if len(managed) != 2 or not managed[0]["boxes"]:
        raise AssertionError("Expected two frames and detections on the positive image")
    streamed = snapshot(detector.predict(images, stream=True, workers=0, verbose=False))
    for left, right in zip(managed, streamed):
        if left["labels"] != right["labels"]:
            raise AssertionError("Managed and stream labels differ")
        np.testing.assert_allclose(left["boxes"], right["boxes"], atol=0.01, rtol=0)
        np.testing.assert_allclose(left["scores"], right["scores"], atol=1e-5, rtol=0)
    native = snapshot(detector.predict(positive, native=True, verbose=False))
    if len(native) != 1 or not native[0]["boxes"]:
        raise AssertionError("Native inference did not return positive detections")
    early = detector.predict(images, stream=True, workers=0, verbose=False)
    next(iter(early))
    early.close()
    if "klygo.models.backend" in sys.modules:
        raise AssertionError("Removed backend package was imported")

    preview = visual.draw_bboxes(positive, result[0])
    media.save(files.join(root, f"{backend}.jpg"), preview, overwrite=True, verbose=False)
    report = {
        "backend": backend, "model": detector.model_id, "device": device,
        "source_sha256": files.hash(source), "managed": managed, "stream": streamed,
        "native": native, "early_close": True, "seconds_all_checks": time.perf_counter()-started,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "transformers", "ultralytics", "keras", "keras-hub")},
    }
    files.save(files.join(root, f"{backend}.json"), report, overwrite=True, verbose=False)
    print(f"PASS {backend}: device={device}, detections={[len(x['boxes']) for x in managed]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["all", "ultralytics", "huggingface", "torchvision", "keras"], default="all")
    parser.add_argument("--source", default=None)
    parser.add_argument("--artifacts", default="artifacts/integration/native-execution")
    args = parser.parse_args()
    files.mkdir(args.artifacts)
    source = args.source or str(files.download(
        "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg",
        args.artifacts, overwrite=True, verbose=False,
    ))
    if args.backend != "all":
        run_one(args.backend, source, args.artifacts)
        return
    statuses = {}
    for backend in ("ultralytics", "huggingface", "torchvision", "keras"):
        completed = subprocess.run([sys.executable, __file__, "--backend", backend,
                                    "--source", source, "--artifacts", args.artifacts], check=False)
        statuses[backend] = completed.returncode
    files.save(files.join(args.artifacts, "status.json"), statuses, overwrite=True, verbose=False)
    if any(statuses.values()):
        raise SystemExit(f"Backend failures: {statuses}")


if __name__ == "__main__":
    main()
