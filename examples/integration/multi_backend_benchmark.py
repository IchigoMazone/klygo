"""Unified Multi-Backend and Multi-GPU Benchmark Tool for Klygo.

Evaluates throughput, first-frame latency, memory footprint, and output integrity
across 5 integrated detector model families and 5 execution modes:
- Architectures: YOLOv8, YOLO-World v2, Grounding DINO tiny, Faster R-CNN, RT-DETR / D-FINE.
- Modes: Raw Framework, Klygo Native Stream, Klygo Native Non-stream,
         Klygo Managed Stream, Klygo Managed Non-stream.
- Hardware: Multi-GPU matrix (CUDA:0, CUDA:1, etc.), Apple Silicon (MPS), CPU.

Usage:
    python examples/integration/multi_backend_benchmark.py --device auto --models all --modes all
    python examples/integration/multi_backend_benchmark.py --device cuda:0 --models yolo,faster-rcnn --repeats 3
    python examples/integration/multi_backend_benchmark.py --device all-gpus --output artifacts/benchmark_results.json
"""

import argparse
import os
import statistics
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil
import numpy as np
import torch

from klygo import files, media, models, outputs, utils, validators

# Ensure PyTorch backend for Keras 3 if loaded
os.environ["KERAS_BACKEND"] = "torch"


# =============================================================================
# 1. HARDWARE DISCOVERY & DEVICE MANAGEMENT
# =============================================================================

def get_cpu_brand_name() -> str:
    """Return accurate CPU brand/model name cross-platform."""
    import platform
    sys_name = platform.system()
    if sys_name == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return val.strip()
        except Exception:
            pass
    elif sys_name == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    elif sys_name == "Darwin":
        try:
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor() or "Host CPU"


def discover_available_devices() -> Dict[str, Any]:
    """Inspect and catalog host hardware accelerators."""
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0

    gpus = []
    for i in range(cuda_count):
        prop = torch.cuda.get_device_properties(i)
        gpus.append({
            "index": i,
            "device_id": f"cuda:{i}",
            "name": prop.name,
            "total_vram_gb": round(prop.total_memory / (1024 ** 3), 2),
            "major": prop.major,
            "minor": prop.minor,
            "multi_processor_count": prop.multi_processor_count,
        })

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cpu_name = get_cpu_brand_name()

    return {
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_count,
        "gpus": gpus,
        "mps_available": mps_available,
        "cpu_name": cpu_name,
        "cpu_threads": os.cpu_count() or 1,
        "torch_version": torch.__version__,
    }


def resolve_target_devices(device_arg: str, discovery: Dict[str, Any]) -> List[str]:
    """Parse device CLI argument into a concrete list of target device strings."""
    raw = device_arg.strip().lower()

    # Handle comma-separated list like "cuda:0,cpu" or "all"
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    resolved = []

    for token in tokens:
        if token in ("all", "all-devices"):
            if discovery["cuda_device_count"] > 0:
                resolved.extend([f"cuda:{i}" for i in range(discovery["cuda_device_count"])])
            if discovery.get("mps_available", False):
                resolved.append("mps")
            resolved.append("cpu")
        elif token == "all-gpus":
            if discovery["cuda_device_count"] == 0:
                resolved.append("cpu")
            else:
                resolved.extend([f"cuda:{i}" for i in range(discovery["cuda_device_count"])])
        elif token in ("gpu", "cuda"):
            if discovery["cuda_device_count"] > 0:
                resolved.append("cuda:0")
            elif discovery.get("mps_available", False):
                resolved.append("mps")
            else:
                resolved.append("cpu")
        elif token in ("auto", "best"):
            if discovery["cuda_available"] and discovery["cuda_device_count"] > 0:
                resolved.append("cuda:0")
            elif discovery.get("mps_available", False):
                resolved.append("mps")
            else:
                resolved.append("cpu")
        elif token.isdigit():
            resolved.append(f"cuda:{token}")
        else:
            resolved.append(token)

    # Deduplicate while preserving order
    seen = set()
    final_devices = []
    for d in resolved:
        if d not in seen:
            seen.add(d)
            final_devices.append(d)
    return final_devices or ["cpu"]


def synchronize_device(device_str: str) -> None:
    """Synchronize device pipeline for accurate wall-clock timing."""
    if device_str.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device_str)
    elif device_str == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


# =============================================================================
# 2. MEASUREMENT ENGINE & PROFILING
# =============================================================================

def measure_run(runner_fn: Callable[[], Tuple[float, int, int]], device_str: str) -> Dict[str, Any]:
    """Measure total duration, first-frame latency, FPS, peak heap, and RSS delta."""
    tracemalloc.start()
    proc = psutil.Process(os.getpid())
    rss_start = proc.memory_info().rss / (1024 * 1024)

    synchronize_device(device_str)
    t0 = time.perf_counter()

    first_frame_ms, total_boxes, frame_count = runner_fn()

    synchronize_device(device_str)
    total_sec = time.perf_counter() - t0

    rss_end = proc.memory_info().rss / (1024 * 1024)
    peak_heap = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    fps = frame_count / total_sec if total_sec > 0 else 0.0
    ms_per_frame = (total_sec / frame_count) * 1000.0 if frame_count > 0 else 0.0

    return {
        "fps": fps,
        "total_sec": total_sec,
        "first_frame_ms": first_frame_ms,
        "ms_per_frame": ms_per_frame,
        "frames": frame_count,
        "total_boxes": total_boxes,
        "peak_heap_mb": peak_heap,
        "rss_delta_mb": rss_end - rss_start,
    }


def aggregate_metrics(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics (median, mean, std) across repeated runs."""
    fps_values = [r["fps"] for r in runs]
    first_ms_values = [r["first_frame_ms"] for r in runs]
    ms_frame_values = [r["ms_per_frame"] for r in runs]
    heap_values = [r["peak_heap_mb"] for r in runs]
    rss_values = [r["rss_delta_mb"] for r in runs]

    return {
        "fps_median": round(statistics.median(fps_values), 2),
        "fps_mean": round(statistics.mean(fps_values), 2),
        "fps_std": round(statistics.stdev(fps_values), 2) if len(fps_values) > 1 else 0.0,
        "first_frame_ms_median": round(statistics.median(first_ms_values), 1),
        "ms_per_frame_median": round(statistics.median(ms_frame_values), 2),
        "peak_heap_mb_median": round(statistics.median(heap_values), 2),
        "rss_delta_mb_median": round(statistics.median(rss_values), 2),
        "total_boxes": runs[-1]["total_boxes"],
        "frames": runs[-1]["frames"],
    }


# =============================================================================
# 3. DETECTOR FACTORIES FOR 5 ARCHITECTURES
# =============================================================================

def build_yolo_detector(device_str: str, imgsz: int = 320) -> Tuple[models.Detector, None]:
    """Instantiate Ultralytics YOLOv8n detector."""
    det = models.load("yolov8n.pt", device=device_str, imgsz=imgsz, threshold=0.25, iou=0.7)
    return det, None


def build_yolo11_detector(device_str: str, imgsz: int = 320) -> Tuple[models.Detector, None]:
    """Instantiate an Ultralytics YOLO11n detector."""
    return models.load(
        "yolo11n.pt", device=device_str, imgsz=imgsz, threshold=0.25, iou=0.7
    ), None


def build_yolo26_detector(device_str: str, imgsz: int = 320) -> Tuple[models.Detector, None]:
    """Instantiate an Ultralytics YOLO26n detector."""
    return models.load(
        "yolo26n.pt", device=device_str, imgsz=imgsz, threshold=0.25, iou=0.7
    ), None


def build_yolo_world_detector(device_str: str, imgsz: int = 320) -> Tuple[models.Detector, List[str]]:
    """Instantiate Ultralytics YOLO-World v2 detector with standard COCO prompt."""
    det = models.load("yolov8s-worldv2.pt", device=device_str, imgsz=imgsz, threshold=0.2, iou=0.7)
    prompt = ["person", "car", "chair", "bottle", "bowl"]
    return det, prompt


def build_grounding_dino_detector(device_str: str, imgsz: int = 320) -> Tuple[models.Detector, str]:
    """Instantiate Hugging Face Grounding DINO tiny detector."""
    det = models.load("grounding-dino-tiny")
    if device_str.startswith("cuda") and torch.cuda.is_available():
        det.model.to(device_str)
    prompt = "person . car . chair . bottle . bowl"
    return det, prompt


def build_torchvision_detector(device_str: str, imgsz: int = 320) -> Tuple[models.Detector, None]:
    """Instantiate Torchvision Faster R-CNN MobileNetV3 FPN detector."""
    import torchvision.models.detection as tv_det

    class TVFasterRCNN(models.Detector):
        def __init__(self):
            metadata = {
                "model_id": "torchvision-fasterrcnn",
                "details": models.details(
                    name="Faster R-CNN",
                    task="Object-Detection",
                    backend="Torchvision",
                    library="torchvision",
                ),
                "flags": models.flags(model=False, post=True),
                "priority": models.priority(models.flags(model=False, post=True), post=["threshold"]),
                "post": {"threshold": 0.25},
            }
            super().__init__(metadata=metadata, backend="Torchvision")
            weights = tv_det.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
            self.model = tv_det.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
            if device_str.startswith("cuda") and torch.cuda.is_available():
                self.model.to(device_str)
            self.model.eval()
            self.names = {i: str(i) for i in range(91)}

        def forward(self, images, prompt=None):
            tensors = self.cast_inputs([media.to_tensor(img) for img in images])
            raw = self.run_inference(tensors)
            formatted = self.format_results(raw)
            return outputs.build_detections(images, formatted, names=self.names)

    return TVFasterRCNN(), None


def build_kerashub_detector(device_str: str, imgsz: int = 320) -> Tuple[models.Detector, None]:
    """Instantiate KerasHub RT-DETR / D-FINE Nano detector."""
    import keras_hub

    class KerasDFine(models.Detector):
        def __init__(self):
            metadata = {
                "model_id": "kerashub-dfine-nano",
                "details": models.details(
                    name="D-FINE",
                    task="Object-Detection",
                    backend="Keras",
                    library="keras_hub",
                ),
                "flags": models.flags(model=False, post=True),
                "priority": models.priority(models.flags(model=False, post=True), post=["threshold"]),
                "post": {"threshold": 0.25},
            }
            super().__init__(metadata=metadata, backend="Keras")
            self.model = keras_hub.models.ObjectDetector.from_preset("dfine_nano_coco")
            self.names = {i: str(i) for i in range(80)}

        def forward(self, images, prompt=None):
            inputs = self.cast_inputs(images)
            raw = self.run_inference(inputs)
            formatted = self.format_results(raw)
            return outputs.build_detections(images, formatted, names=self.names)

    return KerasDFine(), None


MODEL_REGISTRY = {
    "yolo": ("YOLOv8n", "Ultralytics", build_yolo_detector, True),
    "yolo11": ("YOLO11n", "Ultralytics", build_yolo11_detector, True),
    "yolo26": ("YOLO26n", "Ultralytics", build_yolo26_detector, True),
    "yolo-world": ("YOLO-World v2", "Ultralytics", build_yolo_world_detector, True),
    "grounding-dino": ("Grounding DINO tiny", "Hugging Face", build_grounding_dino_detector, True),
    "faster-rcnn": ("Faster R-CNN MobileNetV3", "Torchvision", build_torchvision_detector, True),
    "d-fine": ("RT-DETR / D-FINE Nano", "KerasHub", build_kerashub_detector, True),
}


# =============================================================================
# 4. RUNNER BUILDERS PER MODE
# =============================================================================

def make_stream_runner(detector: models.Detector, paths: List[str], prompt: Any, native: bool, device_str: str):
    def runner():
        synchronize_device(device_str)
        t0 = time.perf_counter()
        if native:
            stream_res = detector.predict(
                paths,
                prompt=prompt,
                native=True,
                model_stream=True,
                verbose=False,
            )
        else:
            stream_res = detector.predict(
                paths,
                prompt=prompt,
                native=False,
                stream=True,
                verbose=False,
            )
        it = iter(stream_res)
        first = next(it)
        synchronize_device(device_str)
        first_ms = (time.perf_counter() - t0) * 1000.0
        total_boxes = len(first)
        cnt = 1
        for frame in it:
            total_boxes += len(frame)
            cnt += 1
        return first_ms, total_boxes, cnt
    return runner


def make_non_stream_runner(detector: models.Detector, paths: List[str], prompt: Any, native: bool, device_str: str):
    def runner():
        synchronize_device(device_str)
        t0 = time.perf_counter()
        res = detector.predict(paths, prompt=prompt, native=native, stream=False, verbose=False)
        synchronize_device(device_str)
        total_ms = (time.perf_counter() - t0) * 1000.0
        total_boxes = sum(len(f) for f in res)
        cnt = len(res)
        first_ms = total_ms / max(cnt, 1)
        return first_ms, total_boxes, cnt
    return runner


def make_raw_ultralytics_runner(
    detector: models.Detector,
    paths: List[str],
    prompt: Any,
    device_str: str,
    imgsz: int,
    stream: bool = False,
):
    config = detector.predict_config
    runtime = {
        "device": config.get("device", device_str),
        "imgsz": config.get("imgsz", imgsz),
        "conf": config.get("threshold", 0.25),
        "iou": config.get("iou", 0.7),
    }
    if stream:
        def runner():
            synchronize_device(device_str)
            t0 = time.perf_counter()
            raw_stream = detector.model.predict(
                source=paths,
                stream=True,
                verbose=False,
                **runtime,
            )
            it = iter(raw_stream)
            first = next(it)
            synchronize_device(device_str)
            first_ms = (time.perf_counter() - t0) * 1000.0
            total_boxes = len(first.boxes)
            cnt = 1
            for f in it:
                total_boxes += len(f.boxes)
                cnt += 1
            return first_ms, total_boxes, cnt
        return runner
    else:
        def runner():
            synchronize_device(device_str)
            t0 = time.perf_counter()
            raw_res = detector.model.predict(
                source=paths,
                stream=False,
                verbose=False,
                **runtime,
            )
            synchronize_device(device_str)
            total_ms = (time.perf_counter() - t0) * 1000.0
            total_boxes = sum(len(f.boxes) for f in raw_res)
            cnt = len(raw_res)
            first_ms = total_ms / max(cnt, 1)
            return first_ms, total_boxes, cnt
        return runner


def _load_pil(path: str):
    """Decode one benchmark image through the public media facade."""
    loaded = media.load(path, verbose=False)
    image = loaded[0] if isinstance(loaded, media.MediaFrames) else loaded
    return image.to_pil(cache=False) if isinstance(image, media.LazyImage) else media.to_pil(image)


def make_raw_grounding_dino_runner(
    detector: models.Detector,
    paths: List[str],
    prompt: Any,
    device_str: str,
    stream: bool = False,
):
    """Run the Hugging Face processor and model without Klygo output containers."""
    prompt_list = [prompt] if isinstance(prompt, str) else list(prompt)
    text = ". ".join(prompt_list) + "."

    if stream:
        def runner():
            first_ms = 0.0
            total_boxes = 0
            started = time.perf_counter()
            for index, path in enumerate(paths):
                image = _load_pil(path)
                inputs = detector.processor(
                    images=[image], text=[text], return_tensors="pt"
                ).to(device_str)
                with torch.inference_mode():
                    raw = detector.model(**inputs)
                processed = detector.processor.post_process_grounded_object_detection(
                    outputs=raw,
                    input_ids=inputs.get("input_ids"),
                    threshold=0.25,
                    text_threshold=0.3,
                    target_sizes=[image.size[::-1]],
                )
                item = processed[0]
                labels = item.get("labels", [])
                total_boxes += sum(
                    label is not None and bool(str(label).strip())
                    for label in labels
                ) if len(labels) else len(item.get("boxes", []))
                if index == 0:
                    synchronize_device(device_str)
                    first_ms = (time.perf_counter() - started) * 1000.0
            return first_ms, total_boxes, len(paths)
        return runner
    else:
        def runner():
            synchronize_device(device_str)
            started = time.perf_counter()
            images = [_load_pil(path) for path in paths]
            texts = [text] * len(images)
            inputs = detector.processor(
                images=images, text=texts, return_tensors="pt"
            ).to(device_str)
            with torch.inference_mode():
                raw = detector.model(**inputs)
            processed = detector.processor.post_process_grounded_object_detection(
                outputs=raw,
                input_ids=inputs.get("input_ids"),
                threshold=0.25,
                text_threshold=0.3,
                target_sizes=[img.size[::-1] for img in images],
            )
            synchronize_device(device_str)
            total_ms = (time.perf_counter() - started) * 1000.0
            total_boxes = 0
            for item in processed:
                labels = item.get("labels", [])
                total_boxes += sum(
                    label is not None and bool(str(label).strip())
                    for label in labels
                ) if len(labels) else len(item.get("boxes", []))
            first_ms = total_ms / max(len(paths), 1)
            return first_ms, total_boxes, len(paths)
        return runner


def make_raw_torchvision_runner(
    detector: models.Detector,
    paths: List[str],
    device_str: str,
    stream: bool = False,
):
    """Run Torchvision tensors and model without Klygo output containers."""
    device = torch.device(device_str)

    if stream:
        def runner():
            first_ms = 0.0
            total_boxes = 0
            started = time.perf_counter()
            for index, path in enumerate(paths):
                tensor = media.to_tensor(_load_pil(path)).to(device)
                with torch.inference_mode():
                    raw = detector.model([tensor])[0]
                total_boxes += len(raw["boxes"])
                if index == 0:
                    synchronize_device(device_str)
                    first_ms = (time.perf_counter() - started) * 1000.0
            return first_ms, total_boxes, len(paths)
        return runner
    else:
        def runner():
            synchronize_device(device_str)
            started = time.perf_counter()
            tensors = [media.to_tensor(_load_pil(path)).to(device) for path in paths]
            with torch.inference_mode():
                raw_list = detector.model(tensors)
            synchronize_device(device_str)
            total_ms = (time.perf_counter() - started) * 1000.0
            total_boxes = sum(len(r["boxes"]) for r in raw_list)
            first_ms = total_ms / max(len(paths), 1)
            return first_ms, total_boxes, len(paths)
        return runner


def make_raw_keras_runner(
    detector: models.Detector,
    paths: List[str],
    device_str: str,
    stream: bool = False,
):
    """Run the KerasHub preprocessing and decode pipeline without Klygo outputs."""
    import keras

    model = detector.model
    preprocessor = model.preprocessor
    image_converter = getattr(preprocessor, "image_converter", None)
    target_size = getattr(image_converter, "image_size", None)

    if stream:
        def runner():
            first_ms = 0.0
            total_boxes = 0
            started = time.perf_counter()
            for index, path in enumerate(paths):
                image = media.to_array(_load_pil(path))
                inputs = keras.ops.convert_to_tensor(np.expand_dims(image, axis=0), dtype="float32")
                prepared_input = (
                    keras.ops.image.resize(inputs, target_size)
                    if target_size is not None
                    else inputs
                )
                prepared = preprocessor(prepared_input)
                predictions = model(prepared, training=False)
                decoded = model.decode_predictions(predictions, inputs)
                score_values = decoded.get(
                    "confidence", decoded.get("scores", decoded.get("detection_scores"))
                )
                class_values = decoded.get(
                    "classes", decoded.get("labels", decoded.get("detection_classes"))
                )
                scores = keras.ops.convert_to_numpy(score_values)[0]
                classes = keras.ops.convert_to_numpy(class_values)[0]
                total_boxes += int(np.count_nonzero((scores >= 0.0) & (classes >= 0)))
                if index == 0:
                    synchronize_device(device_str)
                    first_ms = (time.perf_counter() - started) * 1000.0
            return first_ms, total_boxes, len(paths)
        return runner
    else:
        def runner():
            synchronize_device(device_str)
            started = time.perf_counter()
            images = [media.to_array(_load_pil(path)) for path in paths]
            total_boxes = 0
            for index, img_arr in enumerate(images):
                inputs = keras.ops.convert_to_tensor(np.expand_dims(img_arr, axis=0), dtype="float32")
                prepared_input = (
                    keras.ops.image.resize(inputs, target_size)
                    if target_size is not None
                    else inputs
                )
                prepared = preprocessor(prepared_input)
                predictions = model(prepared, training=False)
                decoded = model.decode_predictions(predictions, inputs)
                score_values = decoded.get(
                    "confidence", decoded.get("scores", decoded.get("detection_scores"))
                )
                class_values = decoded.get(
                    "classes", decoded.get("labels", decoded.get("detection_classes"))
                )
                scores = keras.ops.convert_to_numpy(score_values)[0]
                classes = keras.ops.convert_to_numpy(class_values)[0]
                total_boxes += int(np.count_nonzero((scores >= 0.0) & (classes >= 0)))
            synchronize_device(device_str)
            total_ms = (time.perf_counter() - started) * 1000.0
            first_ms = total_ms / max(len(paths), 1)
            return first_ms, total_boxes, len(paths)
        return runner


def make_raw_runner(
    model_key: str,
    detector: models.Detector,
    paths: List[str],
    prompt: Any,
    device_str: str,
    imgsz: int,
    stream: bool = False,
):
    """Build the matching upstream-framework runner for one model family."""
    if model_key in {"yolo", "yolo11", "yolo26", "yolo-world"}:
        return make_raw_ultralytics_runner(detector, paths, prompt, device_str, imgsz, stream=stream)
    if model_key == "grounding-dino":
        return make_raw_grounding_dino_runner(detector, paths, prompt, device_str, stream=stream)
    if model_key == "faster-rcnn":
        return make_raw_torchvision_runner(detector, paths, device_str, stream=stream)
    if model_key == "d-fine":
        return make_raw_keras_runner(detector, paths, device_str, stream=stream)
    raise ValueError(f"Raw benchmark is unsupported for model {model_key!r}")


# =============================================================================
# 5. TEST INPUT PREPARATION
# =============================================================================

def prepare_test_images(source_arg: Optional[str], max_frames: int) -> List[str]:
    """Resolve or generate test images using Klygo public APIs."""
    if source_arg and files.exists(source_arg):
        if files.is_dir(source_arg):
            found = files.find(source_arg, pattern="*.jpg") or files.find(source_arg, pattern="*.png")
            if found:
                return [str(p) for p in found[:max_frames]]
        elif files.is_file(source_arg):
            return [str(files.resolve(source_arg))]

    # Default fallback: check integration COCO-15
    default_coco = "artifacts/integration/coco15/images"
    if files.exists(default_coco):
        found = files.find(default_coco, pattern="*.jpg")
        if found:
            return [str(p) for p in found[:max_frames]]

    # Synthetic fallback: generate test images in artifacts/benchmark/synthetic
    synth_dir = "artifacts/benchmark/synthetic"
    files.mkdir(synth_dir)
    generated = []
    for i in range(min(max_frames, 5)):
        img_path = files.join(synth_dir, f"test_frame_{i}.jpg")
        if not files.exists(img_path):
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (640, 480), color=(50 + i * 30, 80 + i * 20, 120))
            draw = ImageDraw.Draw(img)
            draw.rectangle([100 + i * 20, 100, 250 + i * 20, 300], fill=(200, 50, 50))
            img.save(img_path, format="JPEG")
        generated.append(str(files.resolve(img_path)))

    return generated


# =============================================================================
# 6. REPORT FORMATTERS (JSON & MARKDOWN)
# =============================================================================

def generate_markdown_report(report_data: Dict[str, Any]) -> str:
    """Format benchmark metrics into a comprehensive Markdown scorecard."""
    lines = [
        "# Klygo Multi-Backend & Multi-GPU Benchmark Report",
        "",
        f"- **Benchmark Title**: {report_data.get('benchmark_title', 'Multi-Backend Benchmark')}",
        f"- **Execution Timestamp**: {report_data.get('timestamp', 'N/A')}",
        f"- **Repeats per Mode**: {report_data.get('repeats', 3)}",
        f"- **Evaluated Images**: {report_data.get('test_images_count', 0)}",
        "",
        "## Evaluated Hardware Accelerators",
        "",
    ]

    for dev in report_data.get("devices", []):
        lines.append(f"- **{dev.get('device_id')}**: {dev.get('name', 'N/A')} ({dev.get('total_vram_gb', 0)} GB VRAM)")

    lines.extend([
        "",
        "## Multi-Model & Multi-Mode Performance Scorecard",
        "",
        "All backends support pure native execution (`native=True`) receiving direct inputs without media preprocessing or coordinate traces.",
        "",
        r"| Device | Model Architecture | Backend | Execution Mode | Throughput (FPS) | First-Frame | Latency/Frame | Peak Heap | $\Delta$ RSS | Boxes | Raw Parity |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for dev_entry in report_data.get("runs", []):
        dev_id = dev_entry.get("device_id", "cuda:0")
        for model in dev_entry.get("models", []):
            m_name = model.get("model_name", "Unknown")
            b_name = model.get("backend", "Unknown")
            for mode_name, metrics in model.get("modes", {}).items():
                fps = metrics.get("fps_median", 0.0)
                first_ms = metrics.get("first_frame_ms_median", 0.0)
                lat_ms = metrics.get("ms_per_frame_median", 0.0)
                heap_mb = metrics.get("peak_heap_mb_median", 0.0)
                rss_mb = metrics.get("rss_delta_mb_median", 0.0)
                boxes = metrics.get("total_boxes", 0)
                parity = metrics.get("parity_matches_raw")
                parity_text = "N/A" if parity is None else "Yes" if parity else "No"
                lines.append(
                    f"| `{dev_id}` | **{m_name}** | {b_name} | {mode_name} | **{fps} FPS** | {first_ms} ms | {lat_ms} ms | {heap_mb} MB | {rss_mb} MB | {boxes} | {parity_text} |"
                )

    # CPU vs GPU Direct Comparison Matrix
    gpu_runs = [r for r in report_data.get("runs", []) if r.get("device_id", "").startswith("cuda") or r.get("device_id") == "mps"]
    cpu_runs = [r for r in report_data.get("runs", []) if r.get("device_id") == "cpu"]

    if gpu_runs and cpu_runs:
        primary_gpu = gpu_runs[0]
        primary_cpu = cpu_runs[0]
        gpu_id = primary_gpu.get("device_id", "GPU")

        lines.extend([
            "",
            "## CPU vs GPU Direct Hardware Comparison Matrix",
            "",
            f"| Model Architecture | Backend | Execution Mode | GPU ({gpu_id}) Throughput | CPU Throughput | Speedup (GPU/CPU) | GPU Latency | CPU Latency | GPU Heap | CPU Heap |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        cpu_lookup = {}
        for m in primary_cpu.get("models", []):
            m_name = m.get("model_name")
            for mode_name, metrics in m.get("modes", {}).items():
                cpu_lookup[(m_name, mode_name)] = metrics

        for m in primary_gpu.get("models", []):
            m_name = m.get("model_name")
            b_name = m.get("backend")
            for mode_name, gpu_metrics in m.get("modes", {}).items():
                cpu_metrics = cpu_lookup.get((m_name, mode_name))
                gpu_fps = gpu_metrics.get("fps_median", 0.0)
                gpu_lat = gpu_metrics.get("ms_per_frame_median", 0.0)
                gpu_heap = gpu_metrics.get("peak_heap_mb_median", 0.0)

                if cpu_metrics:
                    cpu_fps = cpu_metrics.get("fps_median", 0.0)
                    cpu_lat = cpu_metrics.get("ms_per_frame_median", 0.0)
                    cpu_heap = cpu_metrics.get("peak_heap_mb_median", 0.0)
                    speedup_val = f"{round(gpu_fps / max(cpu_fps, 0.001), 1)}x" if cpu_fps > 0 else "N/A"
                    lines.append(
                        f"| **{m_name}** | {b_name} | {mode_name} | **{gpu_fps} FPS** | {cpu_fps} FPS | **{speedup_val}** | {gpu_lat} ms | {cpu_lat} ms | {gpu_heap} MB | {cpu_heap} MB |"
                    )
                else:
                    lines.append(
                        f"| **{m_name}** | {b_name} | {mode_name} | **{gpu_fps} FPS** | — | — | {gpu_lat} ms | — | {gpu_heap} MB | — |"
                    )

        # Row-by-Row Paired CPU vs GPU Layout (Direct Visual Inspection)
        discovery_info = report_data.get("discovery", {})
        gpu_name = primary_gpu.get("name", "NVIDIA GPU")
        for d in report_data.get("devices", []):
            if d.get("device_id") == gpu_id:
                gpu_name = d.get("name", gpu_name)
        cpu_name = discovery_info.get("cpu_name", "Host CPU")

        lines.extend([
            "",
            "## Paired Row-by-Row Hardware Matrix",
            "",
            f"- **GPU**: {gpu_name} ({primary_gpu.get('total_vram_gb', 4.0)} GB VRAM)",
            f"- **CPU**: {cpu_name} ({discovery_info.get('cpu_threads', 8)} Threads)",
            "",
            r"| Model Architecture | Execution Mode | Hardware | Throughput (FPS) | First-Frame | Latency / Frame | Peak Heap | $\Delta$ RSS | Boxes |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for m in primary_gpu.get("models", []):
            m_name = m.get("model_name")
            for mode_name, gpu_metrics in m.get("modes", {}).items():
                cpu_metrics = cpu_lookup.get((m_name, mode_name))
                g_fps = gpu_metrics.get("fps_median", 0.0)
                g_1st = gpu_metrics.get("first_frame_ms_median", 0.0)
                g_lat = gpu_metrics.get("ms_per_frame_median", 0.0)
                g_heap = gpu_metrics.get("peak_heap_mb_median", 0.0)
                g_rss = gpu_metrics.get("rss_delta_mb_median", 0.0)
                g_box = gpu_metrics.get("total_boxes", 0)

                lines.append(
                    f"| **{m_name}** | {mode_name} | **GPU ({gpu_name})** | **{g_fps} FPS** | {g_1st} ms | {g_lat} ms | {g_heap} MB | {g_rss} MB | {g_box} |"
                )

                if cpu_metrics:
                    c_fps = cpu_metrics.get("fps_median", 0.0)
                    c_1st = cpu_metrics.get("first_frame_ms_median", 0.0)
                    c_lat = cpu_metrics.get("ms_per_frame_median", 0.0)
                    c_heap = cpu_metrics.get("peak_heap_mb_median", 0.0)
                    c_rss = cpu_metrics.get("rss_delta_mb_median", 0.0)
                    c_box = cpu_metrics.get("total_boxes", 0)
                    lines.append(
                        f"| | | **CPU ({cpu_name})** | {c_fps} FPS | {c_1st} ms | {c_lat} ms | {c_heap} MB | {c_rss} MB | {c_box} |"
                    )

    lines.extend([
        "",
        "## Key Architectural Findings",
        "",
        "1. Compare each Klygo mode with the Raw Framework row for the same model and device; cross-model FPS is not an overhead measurement.",
        "2. `parity_matches_raw` verifies detection-count parity. Numeric coordinate and score parity requires a dedicated model-specific parity run.",
        r"3. Peak heap and $\Delta$ RSS are observations for this bounded run and do not establish a memory-leak guarantee.",
        "",
    ])

    return "\n".join(lines)


# =============================================================================
# 7. MAIN CLI WORKFLOW
# =============================================================================

def run_benchmark(
    device_arg: str = "auto",
    models_arg: str = "all",
    modes_arg: str = "all",
    source_arg: Optional[str] = None,
    max_frames: int = 10,
    repeats: int = 3,
    warmup: int = 2,
    imgsz: int = 320,
    output_json: str = "artifacts/benchmark/report.json",
    output_md: Optional[str] = "artifacts/benchmark/report.md",
) -> Dict[str, Any]:
    """Execute complete multi-backend multi-GPU benchmark suite."""
    discovery = discover_available_devices()
    target_devices = resolve_target_devices(device_arg, discovery)

    # Resolve models
    if models_arg.strip().lower() == "all":
        selected_model_keys = list(MODEL_REGISTRY.keys())
    else:
        selected_model_keys = [k.strip().lower() for k in models_arg.split(",") if k.strip().lower() in MODEL_REGISTRY]

    # Resolve modes
    all_modes = [
        "raw-non-stream",
        "raw-stream",
        "native-non-stream",
        "native-stream",
        "managed-non-stream",
        "managed-stream",
    ]
    if modes_arg.strip().lower() in ("all", "all-modes"):
        selected_modes = list(all_modes)
    else:
        raw_tokens = [m.strip().lower() for m in modes_arg.split(",") if m.strip()]
        selected_modes = []
        for t in raw_tokens:
            if t == "raw":
                selected_modes.extend(["raw-non-stream", "raw-stream"])
            elif t == "native":
                selected_modes.extend(["native-non-stream", "native-stream"])
            elif t == "managed":
                selected_modes.extend(["managed-non-stream", "managed-stream"])
            elif t in all_modes:
                selected_modes.append(t)
        seen = set()
        selected_modes = [m for m in selected_modes if not (m in seen or seen.add(m))]

    test_images = prepare_test_images(source_arg, max_frames)
    print(f"\n========================================================")
    print(f"Klygo Multi-Backend & Multi-GPU Benchmark")
    print(f"========================================================")
    print(f"Hardware Discovery : {discovery['cuda_device_count']} CUDA GPU(s), MPS={discovery['mps_available']}")
    print(f"Target Devices     : {target_devices}")
    print(f"Models Selected    : {selected_model_keys}")
    print(f"Modes Selected     : {selected_modes}")
    print(f"Test Images Count  : {len(test_images)} images")
    print(f"Repeats per Mode   : {repeats} runs (Warmup: {warmup})")
    print(f"========================================================\n")

    benchmark_runs = []

    for dev_str in target_devices:
        print(f"\n>>> Running Benchmark on Device: {dev_str} <<<")
        dev_entry = {
            "device_id": dev_str,
            "models": [],
        }

        for model_key in selected_model_keys:
            name, backend, factory, has_raw = MODEL_REGISTRY[model_key]
            print(f"\n--- Model: {name} ({backend}) on {dev_str} ---")

            try:
                detector, prompt = factory(dev_str, imgsz=imgsz)
            except Exception as e:
                print(f"[SKIP] Failed to load {name} on {dev_str}: {e}")
                continue

            # Warmup
            print(f"  Warming up ({warmup} runs)...")
            for _ in range(warmup):
                _ = detector.predict(test_images[:min(2, len(test_images))], prompt=prompt, verbose=False)

            model_data = {
                "model_key": model_key,
                "model_name": name,
                "backend": backend,
                "native_execution": "backend-native",
                "modes": {},
            }

            runners = {}
            if "raw-non-stream" in selected_modes and has_raw:
                runners["Raw Non-stream"] = make_raw_runner(
                    model_key, detector, test_images, prompt, dev_str, imgsz, stream=False
                )
            if "raw-stream" in selected_modes and has_raw:
                runners["Raw Stream"] = make_raw_runner(
                    model_key, detector, test_images, prompt, dev_str, imgsz, stream=True
                )
            if "native-non-stream" in selected_modes:
                runners["Klygo Native Non-stream"] = make_non_stream_runner(
                    detector, test_images, prompt, native=True, device_str=dev_str
                )
            if "native-stream" in selected_modes:
                runners["Klygo Native Stream"] = make_stream_runner(
                    detector, test_images, prompt, native=True, device_str=dev_str
                )
            if "managed-non-stream" in selected_modes:
                runners["Klygo Managed Non-stream"] = make_non_stream_runner(
                    detector, test_images, prompt, native=False, device_str=dev_str
                )
            if "managed-stream" in selected_modes:
                runners["Klygo Managed Stream"] = make_stream_runner(
                    detector, test_images, prompt, native=False, device_str=dev_str
                )

            samples = {mode_name: [] for mode_name in runners}
            mode_names = list(runners)
            for repeat in range(repeats):
                offset = repeat % max(len(mode_names), 1)
                ordered_modes = mode_names[offset:] + mode_names[:offset]
                print(f"  Measurement pass {repeat + 1}/{repeats}: {ordered_modes}")
                for mode_name in ordered_modes:
                    samples[mode_name].append(
                        measure_run(runners[mode_name], dev_str)
                    )

            for mode_name, runs in samples.items():
                model_data["modes"][mode_name] = aggregate_metrics(runs)

            raw_metrics = model_data["modes"].get("Raw Stream") or model_data["modes"].get("Raw Non-stream")
            raw_boxes = raw_metrics.get("total_boxes") if raw_metrics else None
            for mode_name, metrics in model_data["modes"].items():
                metrics["parity_matches_raw"] = (
                    None
                    if raw_boxes is None
                    else metrics.get("total_boxes") == raw_boxes
                )
                print(
                    f"    [{mode_name}] FPS: {metrics['fps_median']} | "
                    f"1st: {metrics['first_frame_ms_median']}ms | "
                    f"Boxes: {metrics['total_boxes']} | "
                    f"Parity: {metrics['parity_matches_raw']}"
                )

            dev_entry["models"].append(model_data)

            # Unload model and free GPU cache
            detector.unload()
            synchronize_device(dev_str)

        benchmark_runs.append(dev_entry)

    devices_list = list(discovery.get("gpus", []))
    if any(r.get("device_id") == "cpu" for r in benchmark_runs) or not discovery.get("cuda_available"):
        devices_list.append({"device_id": "cpu", "name": f"Host CPU ({discovery.get('cpu_threads', 1)} threads)", "total_vram_gb": 0.0})

    full_report = {
        "benchmark_title": "Klygo Multi-Backend & Multi-GPU Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "discovery": discovery,
        "devices": devices_list,
        "test_images_count": len(test_images),
        "repeats": repeats,
        "runs": benchmark_runs,
    }

    # Save JSON report
    files.mkdir(files.parent(output_json))
    files.save(output_json, full_report, overwrite=True, verbose=False)
    print(f"\n[OK] JSON benchmark report saved to: {output_json}")

    # Save and print Markdown report
    if output_md:
        md_content = generate_markdown_report(full_report)
        files.mkdir(files.parent(output_md))
        temporary_md = files.with_extension(output_md, ".txt")
        files.save(temporary_md, md_content, overwrite=True, verbose=False)
        files.move(temporary_md, output_md, overwrite=True)
        print(f"[OK] Markdown summary scorecard saved to: {output_md}")

    return full_report


def main():
    parser = argparse.ArgumentParser(description="Klygo Unified Multi-Backend & Multi-GPU Benchmark Tool")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'all-gpus', 'cuda:0', 'cuda:1', 'cpu', 'mps'")
    parser.add_argument("--models", type=str, default="all", help="Models: 'all' or comma-separated 'yolo,yolo11,yolo26,yolo-world,grounding-dino,faster-rcnn,d-fine'")
    parser.add_argument("--modes", type=str, default="all", help="Modes: 'all' or comma-separated 'raw-non-stream,raw-stream,native-non-stream,native-stream,managed-non-stream,managed-stream'")
    parser.add_argument("--source", type=str, default=None, help="Path to image directory, video, or image file")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum number of frames/images to evaluate per run")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeat runs for statistical aggregation")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs before measurement")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference resolution (default: 320)")
    parser.add_argument("--output", type=str, default="artifacts/benchmark/report.json", help="Path to destination JSON report")
    parser.add_argument("--markdown", type=str, default="artifacts/benchmark/report.md", help="Path to destination Markdown scorecard")

    args = parser.parse_args()

    run_benchmark(
        device_arg=args.device,
        models_arg=args.models,
        modes_arg=args.modes,
        source_arg=args.source,
        max_frames=args.max_frames,
        repeats=args.repeats,
        warmup=args.warmup,
        imgsz=args.imgsz,
        output_json=args.output,
        output_md=args.markdown,
    )


if __name__ == "__main__":
    main()
