#!/usr/bin/env python3
"""Generate high-resolution benchmark visualization charts for Klygo 2.1.1.

Produces publication-quality charts in `analyzt/` covering:
1. Multi-backend throughput across execution modes (L4 GPU).
2. Managed pipeline overhead and efficiency relative to raw upstream.
3. Time-to-First-Frame (TTFF) streaming latency comparison.
4. Intelligent dataset curation efficiency (klygo.mining on COCO val2017).
5. Hardware acceleration scaling (NVIDIA T4 vs L4).
6. Postprocessing microbenchmarks (NumPy CPU vs CuPy GPU NMS).
7. Progression across 3 official benchmark runs (T4 Complete -> L4 Baseline -> L4 Optimized).
8. Real-world video workload comparison across 3 domains (Controlled vs People vs Cars).
9. Mining Pareto Frontier: Dataset size vs COCO mAP Trade-off (100% Measured Data).
10. Optimization progression trajectory across 3 benchmark runs: Latency vs Throughput.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Set consistent publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.autolayout": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "raw": "#7E57C2",       # Deep Purple
    "native": "#00ACC1",    # Teal
    "managed": "#1E88E5",   # Vibrant Blue
    "t4": "#FB8C00",        # Amber Orange
    "l4": "#1E88E5",        # Tech Blue
    "accent_green": "#43A047",
    "accent_red": "#E53935",
    "accent_gray": "#757575",
}


def chart_multi_backend_throughput(out_dir):
    """Chart 1: Throughput across 5 core models on NVIDIA L4 across execution modes."""
    l4_path = "benchmarks/gpu_suite/colab_l4_optimized_2026_11_11/controlled/raw_statistics.json"
    with open(l4_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models_map = {
        "yolo": "YOLOv8n",
        "yolo11": "YOLO11n",
        "yolo-world": "YOLO-World v2",
        "yolo26": "RT-DETR / D-FINE",
        "faster-rcnn": "Faster R-CNN",
    }
    modes = ["raw-non-stream", "native-non-stream", "managed-stream"]
    mode_labels = ["Raw Upstream", "Klygo Native", "Klygo Managed Stream"]
    mode_colors = [COLORS["raw"], COLORS["native"], COLORS["managed"]]

    fps_data = {m: {} for m in models_map}
    for row in data["rows"]:
        m = row["model"]
        if m in models_map and row["profile"] == "images" and row["batch"] == 1:
            if row["mode"] in modes:
                fps_data[m][row["mode"]] = row["fps"]["median"]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(models_map))
    bar_width = 0.26

    for idx, (mode, label, color) in enumerate(zip(modes, mode_labels, mode_colors)):
        vals = [fps_data[m].get(mode, 0) for m in models_map]
        offset = (idx - 1) * bar_width
        rects = ax.bar(x + offset, vals, bar_width, label=label, color=color, edgecolor="black", linewidth=0.6, alpha=0.92)
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f"{height:.1f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_title("Multi-Backend Inference Throughput on NVIDIA L4 (Image Batch=1, FPS)", pad=15)
    ax.set_ylabel("Throughput (Frames Per Second)")
    ax.set_xticks(x)
    ax.set_xticklabels([models_map[m] for m in models_map], fontweight="semibold")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(0, 118)

    save_path = os.path.join(out_dir, "01_multi_backend_throughput.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


def chart_pipeline_overhead(out_dir):
    """Chart 2: Managed Stream Efficiency vs Raw Upstream Stream (Overhead Analysis)."""
    with open("benchmarks/gpu_suite/colab_l4_optimized_2026_11_11/controlled/raw_statistics.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    models_map = [
        ("yolo", "YOLOv8n"),
        ("yolo11", "YOLO11n"),
        ("yolo-world", "YOLO-World v2"),
        ("yolo26", "RT-DETR / D-FINE"),
    ]
    models_keys = [k for k, _ in models_map]
    display_names = [name for _, name in models_map]

    raw_fps_map = {}
    managed_fps_map = {}
    for r in data["rows"]:
        if r["profile"] == "images" and r["batch"] == 1 and r["model"] in models_keys and "fps" in r:
            if r["mode"] == "raw-stream":
                raw_fps_map[r["model"]] = r["fps"]["median"]
            elif r["mode"] == "managed-stream":
                managed_fps_map[r["model"]] = r["fps"]["median"]

    raw_fps = [raw_fps_map[k] for k in models_keys]
    managed_fps = [managed_fps_map[k] for k in models_keys]
    efficiency = [(m / r) * 100 for m, r in zip(managed_fps, raw_fps)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})

    # Subplot 1: Direct FPS comparison
    x = np.arange(len(display_names))
    w = 0.35
    ax1.bar(x - w/2, raw_fps, w, label="Raw Upstream Stream", color=COLORS["raw"], edgecolor="black", linewidth=0.6)
    ax1.bar(x + w/2, managed_fps, w, label="Klygo Managed Stream", color=COLORS["managed"], edgecolor="black", linewidth=0.6)
    for i in range(len(display_names)):
        ax1.annotate(f"{raw_fps[i]:.1f}", (x[i] - w/2, raw_fps[i] + 1.5), ha="center", fontsize=9, fontweight="bold")
        ax1.annotate(f"{managed_fps[i]:.1f}", (x[i] + w/2, managed_fps[i] + 1.5), ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Throughput (FPS)")
    ax1.set_title("Stream Throughput Comparison (L4 GPU)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names, fontweight="semibold")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper right")

    # Subplot 2: Retained Efficiency Percentage
    bars = ax2.bar(display_names, efficiency, color=COLORS["accent_green"], edgecolor="black", linewidth=0.6, width=0.55, alpha=0.9)
    ax2.axhline(100, color="gray", linestyle="--", alpha=0.7)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", (bar.get_x() + bar.get_width() / 2, h + 1.2),
                     ha="center", fontsize=10, fontweight="bold", color="#1B5E20")
    ax2.set_ylabel("Retained Throughput (%)")
    ax2.set_title("Managed Pipeline Efficiency (% of Raw)")
    ax2.set_ylim(80, 108)
    ax2.set_xticks(range(len(display_names)))
    ax2.set_xticklabels(display_names, rotation=15, ha="right", fontweight="semibold")

    plt.suptitle("Klygo Managed Pipeline: Zero-Loss Abstraction with Full Coordinate Trace", y=1.02)
    save_path = os.path.join(out_dir, "02_klygo_pipeline_overhead.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


def chart_ttff_latency(out_dir):
    """Chart 3: Time-To-First-Frame (TTFF) streaming latency comparison."""
    import statistics
    with open("benchmarks/gpu_suite/colab_l4_optimized_2026_11_11/controlled/raw_statistics.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    models_map = [
        ("yolo", "YOLOv8n"),
        ("yolo11", "YOLO11n"),
        ("yolo-world", "YOLO-World v2"),
        ("yolo26", "RT-DETR / D-FINE"),
    ]
    models_keys = [k for k, _ in models_map]
    display_names = [name for _, name in models_map]

    raw_ttff_map = {}
    native_ttff_map = {}
    managed_ttff_map = {}

    for r in data["rows"]:
        if r["profile"] == "images" and r["batch"] == 1 and r["model"] in models_keys:
            if "first_host_frame_ms" in r and r["first_host_frame_ms"]:
                val = statistics.median(r["first_host_frame_ms"])
                if r["mode"] == "raw-stream":
                    raw_ttff_map[r["model"]] = val
                elif r["mode"] == "native-stream":
                    native_ttff_map[r["model"]] = val
                elif r["mode"] == "managed-stream":
                    managed_ttff_map[r["model"]] = val

    raw_ttff = [raw_ttff_map[k] for k in models_keys]
    native_stream_ttff = [native_ttff_map[k] for k in models_keys]
    managed_stream_ttff = [managed_ttff_map[k] for k in models_keys]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(display_names))
    w = 0.25

    r1 = ax.bar(x - w, raw_ttff, w, label="Raw Upstream Stream", color=COLORS["raw"], edgecolor="black", linewidth=0.6)
    r2 = ax.bar(x, native_stream_ttff, w, label="Klygo Native Stream", color=COLORS["native"], edgecolor="black", linewidth=0.6)
    r3 = ax.bar(x + w, managed_stream_ttff, w, label="Klygo Managed Stream (Prefetched)", color=COLORS["managed"], edgecolor="black", linewidth=0.6)

    for r in list(r1) + list(r2) + list(r3):
        h = r.get_height()
        ax.annotate(f"{h:.1f}ms", (r.get_x() + r.get_width()/2, h + 0.3), ha="center", fontsize=8.5, fontweight="bold")

    ax.set_title("Time-To-First-Frame (TTFF) Streaming Latency (NVIDIA L4, ms - Lower is Better)", pad=15)
    ax.set_ylabel("Host-Visible Latency (Milliseconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontweight="semibold")
    ax.set_ylim(0, 22)
    ax.legend(loc="upper left")

    save_path = os.path.join(out_dir, "03_first_frame_latency_ttff.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


def chart_mining_curation(out_dir):
    """Chart 4: Mining Curation Efficiency on COCO val2017: Dataset Size vs Accuracy (mAP@50-95)."""
    with open("benchmarks/mining/curation_l4_2026_11_11/summary.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    leaderboard = data.get("leaderboard", [])
    # Filter representative methods
    selected = [
        ("balance_median", "Balanced Median (Ours)", COLORS["accent_green"]),
        ("rare_first_b", "Rare First", "#29B6F6"),
        ("stratified_b", "Stratified Hybrid", "#26A69A"),
        ("coverage_b", "Max Coverage", "#AB47BC"),
        ("klygo_smart_b", "Klygo Smart Hybrid", "#FFA726"),
        ("random_b", "Random Baseline (350)", "#78909C"),
        ("dedup_only", "Deduplicated Only (628)", "#8D6E63"),
        ("full_raw", "Full Raw Dataset (848)", COLORS["accent_red"]),
    ]

    method_dict = {item["method"]: item for item in leaderboard}
    
    names = []
    maps = []
    sizes = []
    colors = []

    for key, display, col in selected:
        if key in method_dict:
            names.append(display)
            maps.append(method_dict[key]["mAP50_95_mean"] * 100)
            sizes.append(method_dict[key]["images"])
            colors.append(col)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(names))
    w = 0.55

    bars = ax1.bar(x, maps, w, color=colors, edgecolor="black", linewidth=0.6, alpha=0.9)
    ax1.set_ylabel("Evaluation Accuracy: mAP@50-95 (%)", fontweight="semibold")
    ax1.set_title("klygo.mining: Curation Accuracy vs Dataset Size on COCO val2017 (YOLOv8n)", pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=22, ha="right", fontweight="semibold")
    ax1.set_ylim(25, 45)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}%\n({sizes[i]} imgs)",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Add baseline indicator
    ax1.axhline(maps[-1], color=COLORS["accent_red"], linestyle=":", linewidth=1.5, label=f"Full Raw Baseline ({maps[-1]:.2f}%)")
    ax1.legend(loc="upper right")

    save_path = os.path.join(out_dir, "04_mining_curation_accuracy.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


def chart_l4_vs_t4_scaling(out_dir):
    """Chart 5: GPU Acceleration Comparison: NVIDIA T4 vs NVIDIA L4."""
    with open("benchmarks/gpu_suite/colab_t4_complete_2026_11_10/controlled/raw_statistics.json", "r", encoding="utf-8") as f:
        t4_data = json.load(f)
    with open("benchmarks/gpu_suite/colab_l4_optimized_2026_11_11/controlled/raw_statistics.json", "r", encoding="utf-8") as f:
        l4_data = json.load(f)

    models_map = [
        ("yolo", "YOLOv8n"),
        ("yolo11", "YOLO11n"),
        ("yolo-world", "YOLO-World v2"),
        ("yolo26", "RT-DETR / D-FINE"),
        ("faster-rcnn", "Faster R-CNN"),
        ("grounding-dino", "Grounding DINO"),
    ]
    models_keys = [k for k, _ in models_map]
    display_names = [name for _, name in models_map]

    t4_fps_map = {r["model"]: r["fps"]["median"] for r in t4_data["rows"]
                  if r["mode"] == "managed-stream" and r["profile"] == "images" and r["batch"] == 1 and r["model"] in models_keys and "fps" in r}
    l4_fps_map = {r["model"]: r["fps"]["median"] for r in l4_data["rows"]
                  if r["mode"] == "managed-stream" and r["profile"] == "images" and r["batch"] == 1 and r["model"] in models_keys and "fps" in r}

    t4_fps = [t4_fps_map[k] for k in models_keys]
    l4_fps = [l4_fps_map[k] for k in models_keys]
    speedup = [(l / t) for l, t in zip(l4_fps, t4_fps)]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(display_names))
    w = 0.35

    ax.bar(x - w/2, t4_fps, w, label="NVIDIA T4 (Colab Standard)", color=COLORS["t4"], edgecolor="black", linewidth=0.6)
    ax.bar(x + w/2, l4_fps, w, label="NVIDIA L4 (Colab Pro)", color=COLORS["l4"], edgecolor="black", linewidth=0.6)

    for i in range(len(display_names)):
        ax.annotate(f"{t4_fps[i]:.1f}", (x[i] - w/2, t4_fps[i] + 1.2), ha="center", fontsize=8.5, fontweight="bold")
        ax.annotate(f"{l4_fps[i]:.1f}\n(+{(speedup[i]-1)*100:.0f}%)", (x[i] + w/2, l4_fps[i] + 1.2),
                    ha="center", fontsize=8.5, fontweight="bold", color="#0D47A1")

    ax.set_title("NVIDIA T4 vs NVIDIA L4 Managed Stream Throughput (FPS)", pad=15)
    ax.set_ylabel("Throughput (Frames Per Second)")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontweight="semibold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")

    save_path = os.path.join(out_dir, "05_gpu_acceleration_l4_vs_t4.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


def chart_nms_microbenchmarks(out_dir):
    """Chart 6: NMS Microbenchmark Scaling (NumPy CPU vs CuPy GPU)."""
    with open("benchmarks/outputs_cpu_gpu/metrics/nms.json", "r", encoding="utf-8") as f:
        nms_data = json.load(f)

    boxes_keys = ["100", "500", "1000", "2000"]
    display_boxes = ["100", "500", "1,000", "2,000"]
    numpy_ms = [nms_data["non_overlap"][b]["numpy_ms"] for b in boxes_keys]
    cupy_ms = [nms_data["non_overlap"][b]["cupy_ms"] for b in boxes_keys]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(display_boxes))
    w = 0.35

    b1 = ax.bar(x - w/2, numpy_ms, w, label="NumPy (CPU Single-Core)", color="#78909C", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w/2, cupy_ms, w, label="CuPy (GPU CUDA Accelerated)", color=COLORS["accent_green"], edgecolor="black", linewidth=0.6)

    for i in range(len(display_boxes)):
        ax.annotate(f"{numpy_ms[i]:.1f}ms", (x[i] - w/2, numpy_ms[i] + 1.0), ha="center", fontsize=9, fontweight="bold")
        speedup = numpy_ms[i] / cupy_ms[i]
        label = f"{cupy_ms[i]:.1f}ms" + (f"\n({speedup:.1f}x)" if speedup > 1.0 else "")
        ax.annotate(label, (x[i] + w/2, cupy_ms[i] + 1.0), ha="center", fontsize=9, fontweight="bold", color="#1B5E20")

    ax.set_title("Non-Maximum Suppression (NMS) Scaling: CPU vs GPU Latency (ms)", pad=15)
    ax.set_xlabel("Candidate Bounding Boxes")
    ax.set_ylabel("Execution Latency (Milliseconds - Lower is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(display_boxes, fontweight="semibold")
    ax.set_ylim(0, 62)
    ax.legend(loc="upper left")

    save_path = os.path.join(out_dir, "06_nms_gpu_speedup.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


def chart_three_benchmark_runs(out_dir):
    """Chart 7: Evolution across 3 official benchmark runs (T4 -> L4 Baseline -> L4 Optimized)."""
    with open("benchmarks/gpu_suite/colab_t4_complete_2026_11_10/controlled/raw_statistics.json", "r", encoding="utf-8") as f:
        t4_data = json.load(f)
    with open("benchmarks/gpu_suite/colab_l4_complete_2026_11_11/controlled/raw_statistics.json", "r", encoding="utf-8") as f:
        l4_base_data = json.load(f)
    with open("benchmarks/gpu_suite/colab_l4_optimized_2026_11_11/controlled/raw_statistics.json", "r", encoding="utf-8") as f:
        l4_opt_data = json.load(f)

    models_map = [
        ("yolo", "YOLOv8n"),
        ("yolo11", "YOLO11n"),
        ("yolo-world", "YOLO-World v2"),
        ("yolo26", "RT-DETR / D-FINE"),
        ("faster-rcnn", "Faster R-CNN"),
        ("grounding-dino", "Grounding DINO"),
    ]
    models_keys = [k for k, _ in models_map]
    display_names = [name for _, name in models_map]

    r1_map = {r["model"]: r["fps"]["median"] for r in t4_data["rows"]
              if r["mode"] == "managed-stream" and r["profile"] == "images" and r["batch"] == 1 and r["model"] in models_keys and "fps" in r}
    r2_map = {r["model"]: r["fps"]["median"] for r in l4_base_data["rows"]
              if r["mode"] == "managed-stream" and r["profile"] == "images" and r["batch"] == 1 and r["model"] in models_keys and "fps" in r}
    r3_map = {r["model"]: r["fps"]["median"] for r in l4_opt_data["rows"]
              if r["mode"] == "managed-stream" and r["profile"] == "images" and r["batch"] == 1 and r["model"] in models_keys and "fps" in r}

    run1_t4 = [r1_map[k] for k in models_keys]
    run2_l4_base = [r2_map[k] for k in models_keys]
    run3_l4_opt = [r3_map[k] for k in models_keys]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(display_names))
    w = 0.26

    r1 = ax.bar(x - w, run1_t4, w, label="Round 1: Colab T4 Complete", color="#FB8C00", edgecolor="black", linewidth=0.6, alpha=0.9)
    r2 = ax.bar(x, run2_l4_base, w, label="Round 2: Colab L4 Baseline", color="#5C6BC0", edgecolor="black", linewidth=0.6, alpha=0.9)
    r3 = ax.bar(x + w, run3_l4_opt, w, label="Round 3: Colab L4 Optimized (+Pipeline Boost)", color="#1E88E5", edgecolor="black", linewidth=0.6, alpha=0.95)

    for i in range(len(display_names)):
        ax.annotate(f"{run1_t4[i]:.1f}", (x[i] - w, run1_t4[i] + 1.2), ha="center", fontsize=8, fontweight="bold")
        ax.annotate(f"{run2_l4_base[i]:.1f}", (x[i], run2_l4_base[i] + 1.2), ha="center", fontsize=8, fontweight="bold")
        delta = ((run3_l4_opt[i] - run2_l4_base[i]) / run2_l4_base[i]) * 100
        delta_str = f"\n(+{delta:.0f}%)" if delta >= 5 else ""
        ax.annotate(f"{run3_l4_opt[i]:.1f}{delta_str}", (x[i] + w, run3_l4_opt[i] + 1.2),
                    ha="center", fontsize=8, fontweight="bold", color="#0D47A1")

    ax.set_title("Progression across 3 Benchmark Runs: T4 -> L4 Baseline -> L4 Optimized (FPS)", pad=15)
    ax.set_ylabel("Throughput (Frames Per Second - FPS)")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontweight="semibold")
    ax.set_ylim(0, 118)
    ax.legend(loc="upper right", framealpha=0.95)

    save_path = os.path.join(out_dir, "07_three_benchmark_runs_evolution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


def chart_three_workloads(out_dir):
    """Chart 8: Throughput across 3 real-world video workloads (Controlled vs People vs Cars)."""
    with open("benchmarks/gpu_suite/colab_l4_optimized_2026_11_11/controlled/raw_statistics.json", "r", encoding="utf-8") as f:
        ctrl_data = json.load(f)
    with open("benchmarks/gpu_suite/colab_l4_optimized_2026_11_11/people/raw_statistics.json", "r", encoding="utf-8") as f:
        people_data = json.load(f)
    with open("benchmarks/gpu_suite/colab_l4_optimized_2026_11_11/cars/raw_statistics.json", "r", encoding="utf-8") as f:
        cars_data = json.load(f)

    models_map = [
        ("yolo", "YOLOv8n"),
        ("yolo11", "YOLO11n"),
        ("yolo-world", "YOLO-World v2"),
        ("yolo26", "RT-DETR / D-FINE"),
    ]
    models_keys = [k for k, _ in models_map]
    display_names = [name for _, name in models_map]

    ctrl_fps_map = {r["model"]: r["fps"]["median"] for r in ctrl_data["rows"]
                    if r["mode"] == "managed-stream" and r["profile"] == "video" and r["batch"] == 1 and r["model"] in models_keys and "fps" in r}
    people_fps_map = {r["model"]: r["fps"]["median"] for r in people_data["rows"]
                      if r["mode"] == "managed-stream" and r["profile"] == "video" and r["batch"] == 1 and r["model"] in models_keys and "fps" in r}
    cars_fps_map = {r["model"]: r["fps"]["median"] for r in cars_data["rows"]
                    if r["mode"] == "managed-stream" and r["profile"] == "video" and r["batch"] == 1 and r["model"] in models_keys and "fps" in r}

    controlled_fps = [ctrl_fps_map[k] for k in models_keys]
    people_fps = [people_fps_map[k] for k in models_keys]
    cars_fps = [cars_fps_map[k] for k in models_keys]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(display_names))
    w = 0.26

    r1 = ax.bar(x - w, controlled_fps, w, label="Domain 1: Controlled Video", color="#8E24AA", edgecolor="black", linewidth=0.6, alpha=0.9)
    r2 = ax.bar(x, people_fps, w, label="Domain 2: People Tracking Video", color="#00897B", edgecolor="black", linewidth=0.6, alpha=0.9)
    r3 = ax.bar(x + w, cars_fps, w, label="Domain 3: Cars / Traffic Video", color="#43A047", edgecolor="black", linewidth=0.6, alpha=0.95)

    for i in range(len(display_names)):
        ax.annotate(f"{controlled_fps[i]:.1f}", (x[i] - w, controlled_fps[i] + 1.2), ha="center", fontsize=8.5, fontweight="bold")
        ax.annotate(f"{people_fps[i]:.1f}", (x[i], people_fps[i] + 1.2), ha="center", fontsize=8.5, fontweight="bold")
        ax.annotate(f"{cars_fps[i]:.1f}", (x[i] + w, cars_fps[i] + 1.2), ha="center", fontsize=8.5, fontweight="bold", color="#1B5E20")

    ax.set_title("Throughput across 3 Real-World Video Workloads (NVIDIA L4 Managed Stream, FPS)", pad=15)
    ax.set_ylabel("Stream Throughput (Frames Per Second - FPS)")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontweight="semibold")
    ax.set_ylim(0, 125)
    ax.legend(loc="upper right", framealpha=0.95)

    save_path = os.path.join(out_dir, "08_three_workloads_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


def chart_mining_pareto_frontier(out_dir):
    """Chart 9: klygo.mining Pareto Frontier: Dataset Size vs COCO mAP Trade-off (100% Measured Data)."""
    with open("benchmarks/mining/curation_l4_2026_11_11/summary.json", "r", encoding="utf-8") as f:
        mdata = json.load(f)

    leaderboard = mdata["leaderboard"]

    fig, ax = plt.subplots(figsize=(11.5, 6))

    colors = {
        "balance_median": "#1E88E5",
        "rare_first_b": "#43A047",
        "quality_b": "#00ACC1",
        "dedup_weighted_b": "#8E24AA",
        "coverage_b": "#FB8C00",
        "diversity_b": "#3949AB",
        "klygo_smart_b": "#D81B60",
        "weighted_b": "#6D4C41",
        "random_b": "#757575",
        "dedup_only": "#546E7A",
        "full_raw": "#E53935",
    }

    pts = []
    for r in leaderboard:
        m = r["method"]
        x = r["images"]
        y = r["mAP50_95_mean"] * 100
        pts.append((x, y, m))
        c = colors.get(m, "#333333")

        # Highlight Pareto-dominant points
        if m in ["balance_median", "rare_first_b", "full_raw"]:
            ax.scatter([x], [y], color=c, s=120, edgecolor="black", linewidth=1.2, zorder=10)
        else:
            ax.scatter([x], [y], color=c, s=70, alpha=0.85, zorder=5)

        # Smart text offset
        off_x, off_y = (0, 7)
        if m == "balance_median":
            off_x, off_y = (0, 10)
        elif m == "full_raw":
            off_x, off_y = (0, 10)
        elif m == "random_b":
            off_x, off_y = (10, -12)
        elif m == "weighted_b":
            off_x, off_y = (10, -6)
        elif m == "klygo_smart_b":
            off_x, off_y = (-35, 7)

        ax.annotate(f"{m}\n({y:.2f}%)", (x, y), xytext=(off_x, off_y), textcoords="offset points",
                    fontsize=8.5, fontweight="bold", color=c, ha="center")

    # Draw Pareto Frontier curve
    pareto_pts = sorted([p for p in pts if p[2] in ["balance_median", "rare_first_b", "dedup_only", "full_raw"]], key=lambda p: p[0])
    px = [p[0] for p in pareto_pts]
    py = [p[1] for p in pareto_pts]
    ax.plot(px, py, color="#1E88E5", linestyle="--", linewidth=2.0, alpha=0.85, label="Pareto Frontier (Optimal Trade-off)", zorder=4)

    # Shaded region under Pareto curve
    ax.fill_between(px, py, [29.5]*len(px), color="#1E88E5", alpha=0.06, label="Feasible Curation Region")

    ax.set_title("klygo.mining: Dataset Size vs COCO mAP Trade-off (Pareto Frontier)", pad=15)
    ax.set_xlabel("Curated Dataset Size (Images - Lower means faster training & reduced storage)", fontsize=11, fontweight="semibold")
    ax.set_ylabel(r"Evaluation Accuracy: $\mathrm{COCO\ mAP}_{val}^{50-95}$ (%)", fontsize=11, fontweight="semibold")

    ax.set_xlim(80, 930)
    ax.set_ylim(29.5, 41.5)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)

    save_path = os.path.join(out_dir, "09_mining_pareto_accuracy_vs_data_size.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


def chart_three_benchmark_runs_trajectory(out_dir):
    """Chart 10: Trajectory across 3 Benchmark Runs: Latency vs Throughput (100% Measured Data)."""
    runs = [
        ("T4 Complete", "benchmarks/gpu_suite/colab_t4_complete_2026_11_10", "#FB8C00", "s"),
        ("L4 Baseline", "benchmarks/gpu_suite/colab_l4_complete_2026_11_11", "#5C6BC0", "^"),
        ("L4 Optimized", "benchmarks/gpu_suite/colab_l4_optimized_2026_11_11", "#1E88E5", "o"),
    ]

    models = {
        "yolo": ("YOLOv8n", "#1E88E5"),
        "yolo11": ("YOLO11n", "#E65100"),
        "yolo26": ("RT-DETR / D-FINE", "#00897B"),
        "yolo-world": ("YOLO-World v2", "#8E24AA"),
        "faster-rcnn": ("Faster R-CNN", "#D32F2F"),
    }

    model_pts = {m: [] for m in models}

    for rname, rpath, rcolor, rmarker in runs:
        with open(f"{rpath}/controlled/raw_statistics.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data["rows"]:
            m = row["model"]
            if m in models and row["mode"] == "managed-stream" and row["profile"] == "images" and row["batch"] == 1:
                fps = row["fps"]["median"]
                lat = 1000.0 / fps
                model_pts[m].append((lat, fps, rname))

    fig, ax = plt.subplots(figsize=(11.5, 6))

    for m, (mdisplay, mcolor) in models.items():
        pts = model_pts[m]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # Draw trajectory line with directional path
        ax.plot(xs, ys, color=mcolor, linewidth=2.2, alpha=0.88, label=mdisplay)
        for i, (x, y, rname) in enumerate(pts):
            marker = runs[i][3]
            ax.scatter([x], [y], color=mcolor, marker=marker, s=85, edgecolor="black", linewidth=0.8, zorder=5)
            # Label final optimized run
            if i == 2:
                ax.annotate(f"{mdisplay}\n({y:.1f} FPS, {x:.1f}ms)", (x, y), xytext=(-10, 8), textcoords="offset points",
                            fontsize=8.5, fontweight="bold", color=mcolor)
            elif i == 0:
                ax.annotate(f"T4 ({y:.1f} FPS)", (x, y), xytext=(5, -10), textcoords="offset points",
                            fontsize=7.5, color="gray")

    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#FB8C00', markersize=9, label='Round 1: Colab T4 Complete'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#5C6BC0', markersize=9, label='Round 2: Colab L4 Baseline'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E88E5', markersize=9, label='Round 3: Colab L4 Optimized (+Pipeline Boost)'),
    ]
    leg1 = ax.legend(handles=custom_lines, loc="upper right", framealpha=0.95, title="3 Benchmark Runs", fontsize=9.5)
    ax.add_artist(leg1)
    ax.legend(loc="lower left", framealpha=0.95, title="Detector Architectures", fontsize=9.5)

    ax.set_title("Optimization Progression across 3 Benchmark Runs: Latency vs Throughput (FPS)", pad=15)
    ax.set_xlabel("Steady-State Inference Latency (ms/frame - Lower is Better)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Inference Throughput (Frames Per Second - Higher is Better)", fontsize=11, fontweight="semibold")

    ax.set_xlim(8, 42)
    ax.set_ylim(20, 115)
    ax.grid(True, linestyle="--", alpha=0.35)

    save_path = os.path.join(out_dir, "10_three_benchmark_runs_trajectory.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


def main():
    # Target directory: analyzt
    out_dir = "analyzt"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("benchmarks/analyzt", exist_ok=True)

    print("Generating official benchmark visualization charts...")
    chart_multi_backend_throughput(out_dir)
    chart_pipeline_overhead(out_dir)
    chart_ttff_latency(out_dir)
    chart_mining_curation(out_dir)
    chart_l4_vs_t4_scaling(out_dir)
    chart_nms_microbenchmarks(out_dir)
    chart_three_benchmark_runs(out_dir)
    chart_three_workloads(out_dir)
    chart_mining_pareto_frontier(out_dir)
    chart_three_benchmark_runs_trajectory(out_dir)

    # Also copy to benchmarks/analyzt for convenience
    import shutil
    for fname in os.listdir(out_dir):
        if fname.endswith(".png") or fname.endswith(".py"):
            shutil.copy2(os.path.join(out_dir, fname), os.path.join("benchmarks/analyzt", fname))

    print("All charts successfully generated in analyzt/ and benchmarks/analyzt/!")


if __name__ == "__main__":
    main()
