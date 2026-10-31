"""Benchmark image-level mining with NumPy and CuPy Detections storage."""

from statistics import median
from time import perf_counter

import numpy as np
from PIL import Image

from klygo import files, media, mining, outputs


FRAMES = 64
BOXES_PER_FRAME = 8
REPEATS = 5


def build_dataset() -> outputs.Detections:
    """Create a deterministic annotated image collection outside timed regions."""
    root = files.join("artifacts", "benchmark", "mining", "images")
    files.mkdir(root)
    names = {0: "car", 1: "person", 2: "bike", 3: "bus"}
    image_paths = []
    for image_index in range(16):
        path = files.join(root, f"source_{image_index:02d}.png")
        rng = np.random.default_rng(image_index)
        pixels = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
        media.save(path, Image.fromarray(pixels), overwrite=True, verbose=False)
        image_paths.append(path)

    frames = []
    for frame_index in range(FRAMES):
        boxes = []
        for box_index in range(BOXES_PER_FRAME):
            left = 2 + (box_index * 13) % 90
            top = 3 + (box_index * 9) % 60
            class_id = (frame_index + box_index) % len(names)
            boxes.append(outputs.Box(
                [left, top, left + 20, top + 18],
                label=names[class_id],
                class_id=class_id,
            ))
        path = image_paths[frame_index % len(image_paths)]
        frames.append(outputs.Detection(
            boxes,
            source_id=f"benchmark:{frame_index}",
            source_path=str(path),
            frame_index=0,
            image_size=(128, 96),
            names=names,
        ))
    return outputs.Detections(frames, names=names)


def synchronize(device: str) -> None:
    """Wait for queued CuPy work when measuring CUDA storage."""
    if device.startswith("cuda"):
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()


def measure_pair(cpu_function, gpu_function) -> tuple[float, float]:
    """Return paired median milliseconds after one warm-up per device."""
    cpu_function()
    gpu_function()
    synchronize("cuda:0")
    samples = {"cpu": [], "cuda:0": []}
    functions = {"cpu": cpu_function, "cuda:0": gpu_function}
    for repeat in range(REPEATS):
        order = ("cpu", "cuda:0") if repeat % 2 == 0 else ("cuda:0", "cpu")
        for device in order:
            synchronize(device)
            started = perf_counter()
            functions[device]()
            synchronize(device)
            samples[device].append((perf_counter() - started) * 1000.0)
    return median(samples["cpu"]), median(samples["cuda:0"])


def run() -> dict:
    """Run CPU/GPU parity checks and return benchmark measurements."""
    cpu = build_dataset()
    gpu = cpu.copy().cuda()
    cpu_scores = mining.score(cpu, weights={"rarity": 0.7, "coverage": 0.3})
    gpu_scores = mining.score(gpu, weights={"rarity": 0.7, "coverage": 0.3})

    operations = {
        "copy": lambda data, scores: data.copy(),
        "stats_annotations": lambda data, scores: mining.stats(data, quality=False),
        "score_annotations": lambda data, scores: mining.score(data, weights={"rarity": 0.7, "coverage": 0.3}),
        "select_weighted": lambda data, scores: mining.select(data, method="weighted", scores=scores, limit=32),
        "balance": lambda data, scores: mining.balance(data, target="median"),
        "stats_quality": lambda data, scores: mining.stats(data, quality=True),
        "find_duplicates": lambda data, scores: mining.find_duplicates(data, threshold=0.98),
        "cluster": lambda data, scores: mining.cluster(data, features=("histogram", "boxes", "classes"), clusters=8),
    }
    measurements = {}
    for name, operation in operations.items():
        cpu_ms, gpu_ms = measure_pair(
            lambda: operation(cpu, cpu_scores),
            lambda: operation(gpu, gpu_scores),
        )
        measurements[name] = {
            "cpu_ms": round(cpu_ms, 3),
            "gpu_ms": round(gpu_ms, 3),
            "gpu_over_cpu": round(gpu_ms / cpu_ms, 3) if cpu_ms else 0.0,
        }

    cpu_summary = mining.stats(cpu, quality=False).to_dict()
    gpu_summary = mining.stats(gpu, quality=False).to_dict()
    selected = mining.select(gpu, method="weighted", scores=gpu_scores, limit=32)
    return {
        "configuration": {"frames": FRAMES, "boxes_per_frame": BOXES_PER_FRAME, "repeats": REPEATS},
        "parity": {
            "stats_equal": cpu_summary == gpu_summary,
            "input_device": gpu.device,
            "selected_device": selected.device,
            "selected_frame_devices": sorted(set(frame.device for frame in selected)),
        },
        "measurements": measurements,
    }


def markdown(result: dict) -> str:
    """Format benchmark results as a compact Markdown table."""
    lines = [
        "# Mining CPU/GPU benchmark",
        "",
        f"Dataset: {FRAMES} images, {BOXES_PER_FRAME} boxes per image, median of {REPEATS} runs after warm-up.",
        "",
        "| Operation | CPU (ms) | GPU-backed (ms) | GPU / CPU |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in result["measurements"].items():
        lines.append(f"| {name} | {values['cpu_ms']} | {values['gpu_ms']} | {values['gpu_over_cpu']}x |")
    lines.extend(["", f"Parity: `{result['parity']}`", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    benchmark = run()
    root = files.join("artifacts", "benchmark")
    files.mkdir(root)
    files.save(files.join(root, "mining_cpu_gpu.json"), benchmark, overwrite=True, verbose=False)
    files.save(files.join(root, "mining_cpu_gpu.md"), markdown(benchmark), overwrite=True, verbose=False)
    print(markdown(benchmark))
