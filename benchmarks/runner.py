"""Official benchmark runner for Klygo multi-backend detection models.

Evaluates intra-model performance across 6 execution modes:
1. Raw Non-stream
2. Raw Stream
3. Klygo Native Non-stream
4. Klygo Native Stream (model_stream=True)
5. Klygo Managed Non-stream
6. Klygo Managed Stream

Outputs are saved to:
- benchmarks/metrics/
- benchmarks/evaluations/
"""

import argparse
import datetime
import gc
import os
import sys
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import torch
from PIL import Image

from klygo import files, media, models, outputs

sys.stdout.reconfigure(line_buffering=True)


def ensure_sample_video(target_path: str = "benchmarks/data/sample_4frames.mp4") -> str:
    """Ensure a lightweight 4-frame video exists for deterministic benchmarking."""
    resolved = files.path(target_path)
    if files.exists(resolved):
        return str(resolved)

    parent_dir = files.parent(resolved)
    if not files.exists(parent_dir):
        files.mkdir(parent_dir)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(resolved), fourcc, 30.0, (320, 240))
    for i in range(4):
        frame = np.full((240, 320, 3), (40 + i * 40, 80 + i * 20, 120 + i * 15), dtype=np.uint8)
        cv2.putText(
            frame,
            f"Benchmark Frame {i + 1}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        writer.write(frame)
    writer.release()
    return str(resolved)


def get_process_rss_mb() -> float:
    """Return process resident set size in megabytes."""
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)


def get_cuda_peak_vram_mb(device_str: str) -> float:
    """Return peak allocated CUDA VRAM in megabytes."""
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated(device_str) / (1024 * 1024), 2)
    return 0.0


def sync_device(device_str: str) -> None:
    """Synchronize device compute queue."""
    if device_str.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device_str)


def cleanup_memory(device_str: str) -> None:
    """Flush Python garbage collection and CUDA cache."""
    gc.collect()
    if device_str.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def benchmark_mode(
    runner_fn: Callable[[], Tuple[float, int]], device_str: str
) -> Dict[str, Any]:
    """Execute and profile a single execution mode."""
    cleanup_memory(device_str)
    if device_str.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_str)

    tracemalloc.start()
    sync_device(device_str)
    t0 = time.perf_counter()

    first_frame_ms, frame_count = runner_fn()

    sync_device(device_str)
    total_sec = time.perf_counter() - t0

    peak_heap = round(tracemalloc.get_traced_memory()[1] / (1024 * 1024), 2)
    tracemalloc.stop()

    peak_vram = get_cuda_peak_vram_mb(device_str)
    current_rss = get_process_rss_mb()

    fps = round(frame_count / total_sec, 2) if total_sec > 0 else 0.0
    ms_per_frame = (
        round((total_sec / frame_count) * 1000.0, 2) if frame_count > 0 else 0.0
    )

    cleanup_memory(device_str)

    return {
        "fps": fps,
        "ms_per_frame": ms_per_frame,
        "first_frame_ms": round(first_frame_ms, 2),
        "total_sec": round(total_sec, 3),
        "ram_mb": current_rss,
        "peak_heap_mb": peak_heap,
        "peak_vram_mb": peak_vram,
        "frames": frame_count,
    }


def benchmark_yolo(
    model_id: str, device_str: str, video_path: str, prompt: Any = None
) -> Dict[str, Any]:
    """Benchmark YOLO architecture across 6 execution modes."""
    import ultralytics

    results: Dict[str, Any] = {}

    # Mode 1: Raw Non-stream
    def run_raw_non_stream():
        raw_m = ultralytics.YOLO(model_id)
        if prompt:
            raw_m.set_classes(prompt)
        t_start = time.perf_counter()
        res = raw_m.predict(
            source=video_path, stream=False, device=device_str, verbose=False
        )
        ttff = (time.perf_counter() - t_start) * 1000.0
        return ttff, len(res)

    results["Raw Non-stream"] = benchmark_mode(run_raw_non_stream, device_str)

    # Mode 2: Raw Stream
    def run_raw_stream():
        raw_m = ultralytics.YOLO(model_id)
        if prompt:
            raw_m.set_classes(prompt)
        t_start = time.perf_counter()
        stream_gen = raw_m.predict(
            source=video_path, stream=True, device=device_str, verbose=False
        )
        first_frame_ms = 0.0
        count = 0
        for _ in stream_gen:
            if count == 0:
                first_frame_ms = (time.perf_counter() - t_start) * 1000.0
            count += 1
        return first_frame_ms, count

    results["Raw Stream"] = benchmark_mode(run_raw_stream, device_str)

    # Mode 3: Klygo Native Non-stream
    det = models.load(model_id, device=device_str)

    def run_klygo_native_non_stream():
        t_start = time.perf_counter()
        res = det.predict(video_path, prompt=prompt, native=True, verbose=False)
        ttff = (time.perf_counter() - t_start) * 1000.0
        return ttff, len(res)

    results["Klygo Native Non-stream"] = benchmark_mode(
        run_klygo_native_non_stream, device_str
    )

    # Mode 4: Klygo Native Stream (model_stream=True)
    def run_klygo_native_stream():
        t_start = time.perf_counter()
        res = det.predict(
            video_path, prompt=prompt, native=True, model_stream=True, verbose=False
        )
        first_frame_ms = 0.0
        count = 0
        for _ in res:
            if count == 0:
                first_frame_ms = (time.perf_counter() - t_start) * 1000.0
            count += 1
        return first_frame_ms, count

    results["Klygo Native Stream"] = benchmark_mode(
        run_klygo_native_stream, device_str
    )

    # Mode 5: Klygo Managed Non-stream
    def run_klygo_managed_non_stream():
        t_start = time.perf_counter()
        res = det.predict(
            video_path, prompt=prompt, native=False, stream=False, verbose=False
        )
        ttff = (time.perf_counter() - t_start) * 1000.0
        return ttff, len(res)

    results["Klygo Managed Non-stream"] = benchmark_mode(
        run_klygo_managed_non_stream, device_str
    )

    # Mode 6: Klygo Managed Stream
    def run_klygo_managed_stream():
        t_start = time.perf_counter()
        res = det.predict(
            video_path, prompt=prompt, native=False, stream=True, verbose=False
        )
        first_frame_ms = 0.0
        count = 0
        for _ in res:
            if count == 0:
                first_frame_ms = (time.perf_counter() - t_start) * 1000.0
            count += 1
        return first_frame_ms, count

    results["Klygo Managed Stream"] = benchmark_mode(
        run_klygo_managed_stream, device_str
    )

    return results


def benchmark_grounding_dino(
    device_str: str, video_path: str
) -> Dict[str, Any]:
    """Benchmark Grounding DINO architecture across execution modes."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    prompt = "person. car. chair. bottle. bowl."
    results: Dict[str, Any] = {}

    cap = cv2.VideoCapture(video_path)
    pil_images = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pil_images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()

    size_cfg = {"shortest_edge": 240, "longest_edge": 320}

    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    raw_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny"
    )
    if device_str.startswith("cuda") and torch.cuda.is_available():
        raw_model.to(device_str)
    raw_model.eval()

    # Mode 1: Raw Non-stream
    def run_raw_non_stream():
        t_start = time.perf_counter()
        inputs = processor(
            images=pil_images,
            text=[prompt] * len(pil_images),
            size=size_cfg,
            return_tensors="pt",
        )
        if device_str.startswith("cuda") and torch.cuda.is_available():
            inputs = {k: v.to(device_str) for k, v in inputs.items()}
        with torch.no_grad():
            outputs_tensor = raw_model(**inputs)
        raw_res = processor.post_process_grounded_object_detection(
            outputs=outputs_tensor,
            input_ids=inputs["input_ids"],
            threshold=0.25,
            target_sizes=[img.size[::-1] for img in pil_images],
        )
        ttff = (time.perf_counter() - t_start) * 1000.0
        return ttff, len(raw_res)

    results["Raw Non-stream"] = benchmark_mode(run_raw_non_stream, device_str)
    results["Raw Stream"] = "Not Supported by Upstream"

    det = models.load("grounding-dino-tiny")
    if device_str.startswith("cuda") and torch.cuda.is_available():
        det.model.to(device_str)

    # Mode 3: Klygo Native Non-stream
    def run_klygo_native_non_stream():
        t_start = time.perf_counter()
        res = det.predict(
            video_path,
            prompt=prompt,
            size=size_cfg,
            batch=2,
            native=True,
            verbose=False,
        )
        ttff = (time.perf_counter() - t_start) * 1000.0
        return ttff, len(res)

    results["Klygo Native Non-stream"] = benchmark_mode(
        run_klygo_native_non_stream, device_str
    )
    results["Klygo Native Stream"] = "Not Supported (No Upstream Engine)"

    # Mode 5: Klygo Managed Non-stream
    def run_klygo_managed_non_stream():
        t_start = time.perf_counter()
        res = det.predict(
            video_path,
            prompt=prompt,
            size=size_cfg,
            batch=2,
            native=False,
            stream=False,
            verbose=False,
        )
        ttff = (time.perf_counter() - t_start) * 1000.0
        return ttff, len(res)

    results["Klygo Managed Non-stream"] = benchmark_mode(
        run_klygo_managed_non_stream, device_str
    )

    # Mode 6: Klygo Managed Stream
    def run_klygo_managed_stream():
        t_start = time.perf_counter()
        res = det.predict(
            video_path,
            prompt=prompt,
            size=size_cfg,
            batch=2,
            native=False,
            stream=True,
            verbose=False,
        )
        first_frame_ms = 0.0
        count = 0
        for _ in res:
            if count == 0:
                first_frame_ms = (time.perf_counter() - t_start) * 1000.0
            count += 1
        return first_frame_ms, count

    results["Klygo Managed Stream"] = benchmark_mode(
        run_klygo_managed_stream, device_str
    )

    return results


def save_reports(
    data: Dict[str, Any], output_root: str, video_path: str
) -> Tuple[str, str]:
    """Save separated metrics and evaluations."""
    metrics_dir = files.path(output_root) / "metrics"
    eval_dir = files.path(output_root) / "evaluations"
    files.mkdir(metrics_dir)
    files.mkdir(eval_dir)

    # 1. Raw JSON metrics (dogfooding files.save)
    json_path = metrics_dir / "benchmark_metrics.json"
    files.save(json_path, data, overwrite=True, verbose=False)

    # 2. Markdown tables
    lines = [
        "# Klygo Intra-Model Benchmark Performance Tables",
        f"**Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Sample Clip:** 4 frames (320x240 resolution) from `{video_path}`",
        "",
        "---",
    ]

    for dev, dev_data in data.items():
        lines.append(f"## Device: {dev.upper()}")
        for model_name, modes in dev_data.items():
            lines.append(f"### Model: {model_name}")
            lines.append(
                "| Execution Mode | Status | FPS | Latency (ms/frame) | TTFF (ms) | Peak RAM (MB) | Peak VRAM (MB) |"
            )
            lines.append(
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
            )
            for mode_name, metrics in modes.items():
                if isinstance(metrics, dict):
                    status = "✅ Supported"
                    fps = f"{metrics['fps']:.2f}"
                    lat = f"{metrics['ms_per_frame']:.2f}"
                    ttff = f"{metrics['first_frame_ms']:.2f}"
                    ram = f"{metrics['peak_heap_mb']:.2f} (Heap) / {metrics['ram_mb']:.1f} (RSS)"
                    vram = (
                        f"{metrics['peak_vram_mb']:.2f}"
                        if metrics["peak_vram_mb"] > 0
                        else "0.00"
                    )
                    lines.append(
                        f"| **{mode_name}** | {status} | **{fps}** | {lat} | {ttff} | {ram} | {vram} |"
                    )
                else:
                    lines.append(
                        f"| **{mode_name}** | ❌ {metrics} | - | - | - | - | - |"
                    )
            lines.append("")

    tables_path = metrics_dir / "benchmark_tables.md"
    files.save(tables_path, "\n".join(lines), overwrite=True, verbose=False)

    return str(json_path), str(tables_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Klygo Benchmark Harness")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "cuda:0"],
        default="auto",
        help="Target device for evaluation",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names (yolov8, yolo-world, grounding-dino) or 'all'",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks",
        help="Root directory for saving metrics and evaluations",
    )
    args = parser.parse_args()

    video_path = ensure_sample_video()

    devices = []
    if args.device == "auto":
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda:0")
    elif args.device == "cuda":
        devices = ["cuda:0"]
    else:
        devices = [args.device]

    target_models = (
        ["yolov8", "yolo-world", "grounding-dino"]
        if args.models == "all"
        else [m.strip().lower() for m in args.models.split(",")]
    )

    print("===============================================================", flush=True)
    print("KLYGO MULTI-BACKEND BENCHMARK RUNNER", flush=True)
    print(f"Target Devices : {devices}", flush=True)
    print(f"Target Models  : {target_models}", flush=True)
    print(f"Test Video     : {video_path}", flush=True)
    print("===============================================================", flush=True)

    full_results: Dict[str, Any] = {}

    for dev in devices:
        print(f"\n[+] Testing on device: {dev.upper()}", flush=True)
        full_results[dev] = {}

        if "yolov8" in target_models:
            print(f"  - Testing YOLOv8n ({dev})...", flush=True)
            full_results[dev]["YOLOv8n"] = benchmark_yolo(
                "yolov8n.pt", dev, video_path
            )
            print("    -> Done YOLOv8n.", flush=True)

        if "yolo-world" in target_models:
            print(f"  - Testing YOLO-World v2 ({dev})...", flush=True)
            prompt = ["person", "car", "chair", "bottle", "bowl"]
            full_results[dev]["YOLO-World"] = benchmark_yolo(
                "yolov8s-worldv2.pt", dev, video_path, prompt=prompt
            )
            print("    -> Done YOLO-World.", flush=True)

        if "grounding-dino" in target_models:
            print(f"  - Testing Grounding DINO Tiny ({dev})...", flush=True)
            full_results[dev]["Grounding DINO Tiny"] = benchmark_grounding_dino(
                dev, video_path
            )
            print("    -> Done Grounding DINO Tiny.", flush=True)

    json_path, tables_path = save_reports(full_results, args.output_dir, video_path)

    print("\n===============================================================", flush=True)
    print("ALL BENCHMARKS COMPLETE!", flush=True)
    print(f"Metrics JSON : {json_path}", flush=True)
    print(f"Tables MD    : {tables_path}", flush=True)
    print("===============================================================", flush=True)


if __name__ == "__main__":
    main()
