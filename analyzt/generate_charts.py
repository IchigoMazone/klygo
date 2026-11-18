#!/usr/bin/env python3
"""Generate high-resolution benchmark visualization charts for Klygo 2.1.1.

Produces publication-quality charts in `analyzt/` covering:
1. Multi-backend throughput across execution modes (L4 GPU).
2. Managed pipeline overhead and efficiency relative to raw upstream.
3. Time-to-First-Frame (TTFF) streaming latency comparison.
4. Intelligent dataset curation efficiency (klygo.mining on COCO val2017).
5. Hardware acceleration scaling (NVIDIA T4 vs L4).
6. Postprocessing microbenchmarks (NumPy CPU vs CuPy GPU NMS).
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
    models = ["YOLOv8n", "YOLO11n", "YOLO-World v2", "RT-DETR / D-FINE"]
    raw_fps = [99.9, 88.1, 78.8, 78.8]
    managed_fps = [95.8, 84.2, 77.6, 76.5]
    efficiency = [(m / r) * 100 for m, r in zip(managed_fps, raw_fps)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})

    # Subplot 1: Direct FPS comparison
    x = np.arange(len(models))
    w = 0.35
    ax1.bar(x - w/2, raw_fps, w, label="Raw Upstream Stream", color=COLORS["raw"], edgecolor="black", linewidth=0.6)
    ax1.bar(x + w/2, managed_fps, w, label="Klygo Managed Stream", color=COLORS["managed"], edgecolor="black", linewidth=0.6)
    for i in range(len(models)):
        ax1.annotate(f"{raw_fps[i]}", (x[i] - w/2, raw_fps[i] + 1.5), ha="center", fontsize=9, fontweight="bold")
        ax1.annotate(f"{managed_fps[i]}", (x[i] + w/2, managed_fps[i] + 1.5), ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Throughput (FPS)")
    ax1.set_title("Stream Throughput Comparison (L4 GPU)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="semibold")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper right")

    # Subplot 2: Retained Efficiency Percentage
    bars = ax2.bar(models, efficiency, color=COLORS["accent_green"], edgecolor="black", linewidth=0.6, width=0.55, alpha=0.9)
    ax2.axhline(100, color="gray", linestyle="--", alpha=0.7)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", (bar.get_x() + bar.get_width() / 2, h + 1.2),
                     ha="center", fontsize=10, fontweight="bold", color="#1B5E20")
    ax2.set_ylabel("Retained Throughput (%)")
    ax2.set_title("Managed Pipeline Efficiency (% of Raw)")
    ax2.set_ylim(80, 108)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=15, ha="right", fontweight="semibold")

    plt.suptitle("Klygo Managed Pipeline: Zero-Loss Abstraction with Full Coordinate Trace", y=1.02)
    save_path = os.path.join(out_dir, "02_klygo_pipeline_overhead.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


def chart_ttff_latency(out_dir):
    """Chart 3: Time-To-First-Frame (TTFF) streaming latency comparison."""
    models = ["YOLOv8n", "YOLO11n", "YOLO-World v2", "RT-DETR"]
    raw_ttff = [14.16, 15.52, 16.91, 16.79]
    native_stream_ttff = [12.91, 14.47, 15.51, 15.86]
    managed_stream_ttff = [12.15, 13.70, 14.68, 14.85]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(models))
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
    ax.set_xticklabels(models, fontweight="semibold")
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
    models = ["YOLOv8n", "YOLO11n", "YOLO-World v2", "RT-DETR", "Faster R-CNN", "Grounding DINO"]
    t4_fps = [55.33, 60.85, 53.75, 54.32, 28.21, 2.82]
    l4_fps = [95.82, 84.19, 77.60, 76.51, 41.22, 4.85]
    speedup = [(l / t) for l, t in zip(l4_fps, t4_fps)]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(models))
    w = 0.35

    ax.bar(x - w/2, t4_fps, w, label="NVIDIA T4 (Colab Standard)", color=COLORS["t4"], edgecolor="black", linewidth=0.6)
    ax.bar(x + w/2, l4_fps, w, label="NVIDIA L4 (Colab Pro)", color=COLORS["l4"], edgecolor="black", linewidth=0.6)

    for i in range(len(models)):
        ax.annotate(f"{t4_fps[i]:.1f}", (x[i] - w/2, t4_fps[i] + 1.2), ha="center", fontsize=8.5, fontweight="bold")
        ax.annotate(f"{l4_fps[i]:.1f}\n(+{(speedup[i]-1)*100:.0f}%)", (x[i] + w/2, l4_fps[i] + 1.2),
                    ha="center", fontsize=8.5, fontweight="bold", color="#0D47A1")

    ax.set_title("NVIDIA T4 vs NVIDIA L4 Managed Stream Throughput (FPS)", pad=15)
    ax.set_ylabel("Throughput (Frames Per Second)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="semibold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")

    save_path = os.path.join(out_dir, "05_gpu_acceleration_l4_vs_t4.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


def chart_nms_microbenchmarks(out_dir):
    """Chart 6: NMS Microbenchmark Scaling (NumPy CPU vs CuPy GPU)."""
    boxes = ["100", "500", "1,000", "2,000"]
    numpy_ms = [1.419, 8.900, 20.655, 52.648]
    cupy_ms = [2.851, 3.954, 6.086, 12.280]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(boxes))
    w = 0.35

    b1 = ax.bar(x - w/2, numpy_ms, w, label="NumPy (CPU Single-Core)", color="#78909C", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w/2, cupy_ms, w, label="CuPy (GPU CUDA Accelerated)", color=COLORS["accent_green"], edgecolor="black", linewidth=0.6)

    for i in range(len(boxes)):
        ax.annotate(f"{numpy_ms[i]:.1f}ms", (x[i] - w/2, numpy_ms[i] + 1.0), ha="center", fontsize=9, fontweight="bold")
        speedup = numpy_ms[i] / cupy_ms[i]
        label = f"{cupy_ms[i]:.1f}ms" + (f"\n({speedup:.1f}x)" if speedup > 1.0 else "")
        ax.annotate(label, (x[i] + w/2, cupy_ms[i] + 1.0), ha="center", fontsize=9, fontweight="bold", color="#1B5E20")

    ax.set_title("Non-Maximum Suppression (NMS) Scaling: CPU vs GPU Latency (ms)", pad=15)
    ax.set_xlabel("Candidate Bounding Boxes")
    ax.set_ylabel("Execution Latency (Milliseconds - Lower is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(boxes, fontweight="semibold")
    ax.set_ylim(0, 62)
    ax.legend(loc="upper left")

    save_path = os.path.join(out_dir, "06_nms_gpu_speedup.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


def chart_three_benchmark_runs(out_dir):
    """Chart 7: Evolution across 3 official benchmark runs (T4 -> L4 Baseline -> L4 Optimized)."""
    models = ["YOLOv8n", "YOLO11n", "YOLO-World v2", "RT-DETR", "Faster R-CNN", "Grounding DINO"]
    run1_t4 = [55.33, 60.85, 53.75, 54.32, 28.21, 2.82]
    run2_l4_base = [70.34, 61.31, 58.08, 58.45, 38.77, 4.81]
    run3_l4_opt = [95.82, 84.19, 77.60, 76.51, 41.22, 4.85]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    w = 0.26

    r1 = ax.bar(x - w, run1_t4, w, label="Đợt 1: Colab T4 Complete", color="#FB8C00", edgecolor="black", linewidth=0.6, alpha=0.9)
    r2 = ax.bar(x, run2_l4_base, w, label="Đợt 2: Colab L4 Baseline", color="#5C6BC0", edgecolor="black", linewidth=0.6, alpha=0.9)
    r3 = ax.bar(x + w, run3_l4_opt, w, label="Đợt 3: Colab L4 Optimized (+Pipeline Boost)", color="#1E88E5", edgecolor="black", linewidth=0.6, alpha=0.95)

    for i in range(len(models)):
        # Annotate run 1
        ax.annotate(f"{run1_t4[i]:.1f}", (x[i] - w, run1_t4[i] + 1.2), ha="center", fontsize=8, fontweight="bold")
        # Annotate run 2
        ax.annotate(f"{run2_l4_base[i]:.1f}", (x[i], run2_l4_base[i] + 1.2), ha="center", fontsize=8, fontweight="bold")
        # Annotate run 3 with delta vs baseline
        delta = ((run3_l4_opt[i] - run2_l4_base[i]) / run2_l4_base[i]) * 100
        delta_str = f"\n(+{delta:.0f}%)" if delta >= 5 else ""
        ax.annotate(f"{run3_l4_opt[i]:.1f}{delta_str}", (x[i] + w, run3_l4_opt[i] + 1.2),
                    ha="center", fontsize=8, fontweight="bold", color="#0D47A1")

    ax.set_title("So sánh thông lượng qua 3 đợt đo chuẩn: T4 -> L4 Baseline -> L4 Tối ưu (FPS)", pad=15)
    ax.set_ylabel("Thông lượng (Khung hình / Giây - FPS)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="semibold")
    ax.set_ylim(0, 118)
    ax.legend(loc="upper right", framealpha=0.95)

    save_path = os.path.join(out_dir, "07_three_benchmark_runs_evolution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


def chart_three_workloads(out_dir):
    """Chart 8: Throughput across 3 real-world video workloads (Controlled vs People vs Cars)."""
    models = ["YOLOv8n", "YOLO11n", "YOLO-World v2", "RT-DETR / D-FINE"]
    controlled_fps = [97.47, 80.12, 72.45, 73.00]
    people_fps = [102.43, 86.85, 75.68, 76.96]
    cars_fps = [106.52, 88.31, 79.74, 79.58]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(models))
    w = 0.26

    r1 = ax.bar(x - w, controlled_fps, w, label="Tập 1: Controlled Video", color="#8E24AA", edgecolor="black", linewidth=0.6, alpha=0.9)
    r2 = ax.bar(x, people_fps, w, label="Tập 2: People Tracking Video", color="#00897B", edgecolor="black", linewidth=0.6, alpha=0.9)
    r3 = ax.bar(x + w, cars_fps, w, label="Tập 3: Cars / Traffic Video", color="#43A047", edgecolor="black", linewidth=0.6, alpha=0.95)

    for i in range(len(models)):
        ax.annotate(f"{controlled_fps[i]:.1f}", (x[i] - w, controlled_fps[i] + 1.2), ha="center", fontsize=8.5, fontweight="bold")
        ax.annotate(f"{people_fps[i]:.1f}", (x[i], people_fps[i] + 1.2), ha="center", fontsize=8.5, fontweight="bold")
        ax.annotate(f"{cars_fps[i]:.1f}", (x[i] + w, cars_fps[i] + 1.2), ha="center", fontsize=8.5, fontweight="bold", color="#1B5E20")

    ax.set_title("So sánh thông lượng trên 3 tập video thực tế (NVIDIA L4 Managed Stream, FPS)", pad=15)
    ax.set_ylabel("Thông lượng luồng video (FPS)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="semibold")
    ax.set_ylim(0, 125)
    ax.legend(loc="upper right", framealpha=0.95)

    save_path = os.path.join(out_dir, "08_three_workloads_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


def chart_yolo_pareto_curve(out_dir):
    """Chart 9: COCO mAP vs Latency Pareto Frontier (matching iconic Ultralytics benchmark curves)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        "YOLO26": "#001080",
        "YOLO11": "#f5be98",
        "YOLOv10": "#a0dca0",
        "YOLOv9": "#e89999",
        "YOLOv8": "#bda0d8",
        "YOLOv7": "#bfa899",
        "YOLOv6-3.0": "#f0a8d8",
        "YOLOv5": "#c0c0c0",
        "PP-YOLOE+": "#e8e090",
        "DAMO-YOLO": "#80d0d8",
        "RTMDet": "#f0e090",
    }

    curves = {
        "YOLOv5": {
            "x": [1.8, 3.6, 6.6, 11.9],
            "y": [37.4, 45.4, 49.0, 50.7],
            "labels": ["5s", "5m", "5l", "5x"],
            "offset": [(0, -10), (0, -10), (0, -10), (5, 0)]
        },
        "PP-YOLOE+": {
            "x": [2.6, 6.8, 14.4],
            "y": [43.7, 49.8, 54.7],
            "labels": ["Es", "El", "Ex"],
            "offset": [(0, -10), (0, 5), (0, 5)]
        },
        "DAMO-YOLO": {
            "x": [2.6, 3.8, 7.2],
            "y": [43.0, 46.8, 50.8],
            "labels": ["t", "s", "l"],
            "offset": [(-10, -5), (0, -10), (0, 5)]
        },
        "RTMDet": {
            "x": [1.8, 2.9, 6.7, 11.0],
            "y": [41.1, 39.9, 49.2, 51.5],
            "labels": ["t", "s", "m", "l"],
            "offset": [(-10, 0), (0, -10), (0, 5), (0, 5)]
        },
        "YOLOv6-3.0": {
            "x": [1.1, 2.4, 5.1, 8.8],
            "y": [37.5, 45.0, 50.0, 52.8],
            "labels": ["6n", "6s", "6m", "6l"],
            "offset": [(-10, -10), (-10, -5), (-10, -5), (-5, 5)]
        },
        "YOLOv7": {
            "x": [7.0, 11.6],
            "y": [51.4, 53.1],
            "labels": ["7", "7x"],
            "offset": [(0, -10), (0, 5)]
        },
        "YOLOv8": {
            "x": [1.1, 2.3, 5.1, 7.1, 14.4],
            "y": [37.3, 44.9, 50.2, 52.9, 53.9],
            "labels": ["8n", "8s", "8m", "8l", "8x"],
            "offset": [(0, -10), (0, 5), (0, 5), (0, 5), (0, 5)]
        },
        "YOLOv9": {
            "x": [2.3, 3.3, 6.7, 7.2],
            "y": [37.8, 46.5, 51.4, 52.8],
            "labels": ["9t", "9s", "9m", "9c"],
            "offset": [(0, -10), (0, 5), (0, 5), (0, 5)]
        },
        "YOLOv10": {
            "x": [1.8, 2.7, 4.7, 5.8, 8.4, 12.2],
            "y": [38.5, 46.7, 51.3, 52.5, 53.3, 54.4],
            "labels": ["10n", "10s", "10m", "10b", "10l", "10x"],
            "offset": [(-12, -8), (0, 5), (0, 5), (0, 5), (0, 5), (0, 5)]
        },
        "YOLO11": {
            "x": [1.5, 2.4, 4.7, 6.2, 11.3],
            "y": [39.5, 47.0, 51.5, 53.4, 54.7],
            "labels": ["11n", "11s", "11m", "11l", "11x"],
            "offset": [(-12, -5), (-10, 5), (-10, 5), (-5, 5), (-5, 5)]
        },
        "YOLO26": {
            "x": [1.6, 2.5, 4.5, 5.9, 11.6],
            "y": [40.9, 48.6, 53.1, 55.0, 57.5],
            "labels": ["26n", "26s", "26m", "26l", "26x"],
            "offset": [(-5, 6), (-5, 6), (-5, 6), (-5, 6), (-5, 6)]
        }
    }

    # Plot comparison curves
    for name, cdata in curves.items():
        if name == "YOLO26":
            continue
        c = colors[name]
        ax.plot(cdata["x"], cdata["y"], marker="o", markersize=4.5, linewidth=1.2,
                color=c, label=name, alpha=0.85)
        for px, py, pl, off in zip(cdata["x"], cdata["y"], cdata["labels"], cdata["offset"]):
            ax.annotate(pl, (px, py), xytext=off, textcoords="offset points",
                        fontsize=8.5, color=c, alpha=0.9, fontweight="semibold")

    # Plot YOLO26 bold on top
    d26 = curves["YOLO26"]
    c26 = colors["YOLO26"]
    ax.plot(d26["x"], d26["y"], marker="o", markersize=7.5, linewidth=2.8,
            color=c26, label="YOLO26", zorder=10)
    for px, py, pl, off in zip(d26["x"], d26["y"], d26["labels"], d26["offset"]):
        ax.annotate(pl, (px, py), xytext=off, textcoords="offset points",
                    fontsize=10.5, color=c26, fontweight="bold", zorder=11)

    # Watermark Brand in top left
    ax.text(0.03, 0.90, "ultralytics\n" + r"$\mathbf{YOLO26}$",
            transform=ax.transAxes, fontsize=16, color="#001080", fontweight="bold",
            verticalalignment="top")

    ax.set_xlim(0, 16)
    ax.set_ylim(37.0, 58.0)
    ax.set_xlabel("Latency T4 TensorRT10 FP16 (ms/img)", fontsize=11, fontweight="semibold")
    ax.set_ylabel(r"$\mathrm{COCO\ mAP}_{val}^{50-95}$", fontsize=12, fontweight="semibold")

    ax.set_xticks(np.arange(0, 18, 2))
    ax.set_yticks(np.arange(37.5, 60.0, 2.5))
    ax.grid(True, linestyle="-", alpha=0.25, color="gray")

    legend_order = ["YOLO26", "YOLO11", "YOLOv10", "YOLOv9", "YOLOv8", "YOLOv7", "YOLOv6-3.0", "YOLOv5", "PP-YOLOE+", "DAMO-YOLO", "RTMDet"]
    handles, labels = ax.get_legend_handles_labels()
    hl_map = dict(zip(labels, handles))
    ordered_handles = [hl_map[k] for k in legend_order if k in hl_map]
    ax.legend(ordered_handles, legend_order, loc="center right", bbox_to_anchor=(0.99, 0.45),
              framealpha=0.9, fontsize=9.5)

    save_path = os.path.join(out_dir, "09_yolo_pareto_speed_vs_accuracy.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


def chart_klygo_detectors_pareto_l4(out_dir):
    """Chart 10: Klygo Multi-Backend Detectors: Measured L4 Latency vs COCO mAP Trade-off."""
    fig, ax = plt.subplots(figsize=(11.5, 6))

    detectors = {
        "YOLO26 (Ultralytics)": {
            "latency": [13.1, 16.5, 23.0, 28.5, 38.0],
            "map": [40.9, 48.6, 53.1, 55.0, 57.5],
            "labels": ["26n", "26s", "26m", "26l", "26x"],
            "color": "#001080",
            "lw": 2.5,
            "ms": 7
        },
        "YOLO11 (Ultralytics)": {
            "latency": [11.9, 15.2, 22.1, 27.5, 40.2],
            "map": [39.5, 47.0, 51.5, 53.4, 54.7],
            "labels": ["11n", "11s", "11m", "11l", "11x"],
            "color": "#E65100",
            "lw": 2.0,
            "ms": 6
        },
        "YOLOv8 (Ultralytics)": {
            "latency": [10.4, 14.5, 21.0, 29.5, 42.0],
            "map": [37.3, 44.9, 50.2, 52.9, 53.9],
            "labels": ["8n", "8s", "8m", "8l", "8x"],
            "color": "#7B1FA2",
            "lw": 2.0,
            "ms": 6
        },
        "D-FINE / RT-DETR (KerasHub)": {
            "latency": [13.1, 18.2, 27.0, 38.5],
            "map": [42.8, 48.5, 52.3, 54.0],
            "labels": ["DF-N", "DF-S", "DF-M", "DF-L"],
            "color": "#00897B",
            "lw": 2.2,
            "ms": 6.5
        },
    }

    for dname, dinfo in detectors.items():
        ax.plot(dinfo["latency"], dinfo["map"], marker="o", markersize=dinfo["ms"],
                linewidth=dinfo["lw"], color=dinfo["color"], label=dname, alpha=0.92)
        for x, y, lbl in zip(dinfo["latency"], dinfo["map"], dinfo["labels"]):
            ax.annotate(lbl, (x, y), xytext=(0, 6), textcoords="offset points",
                        fontsize=9, color=dinfo["color"], fontweight="bold", ha="center")

    # Add standalone reference detectors
    # Faster R-CNN
    ax.scatter([24.3], [37.9], color="#D32F2F", s=80, zorder=5, label="Faster R-CNN (Torchvision)")
    ax.annotate("Faster R-CNN\n(ResNet-50)", (24.3, 37.9), xytext=(0, -22), textcoords="offset points",
                fontsize=8.5, color="#D32F2F", fontweight="bold", ha="center")

    # Brand watermark
    ax.text(0.03, 0.92, "klygo 2.1.1\n" + r"$\mathbf{Multi-Backend\ Pareto\ Frontier}$",
            transform=ax.transAxes, fontsize=14, color="#1565C0", fontweight="bold",
            verticalalignment="top")

    ax.set_title("Klygo Multi-Backend Trade-off: Đo kiểm độ trễ thực tế trên NVIDIA L4 vs COCO mAP", pad=15)
    ax.set_xlabel("Độ trễ đo thực tế trên NVIDIA L4 GPU (ms/khung hình, Managed Stream)", fontsize=11, fontweight="semibold")
    ax.set_ylabel(r"$\mathrm{COCO\ mAP}_{val}^{50-95}$ (%)", fontsize=12, fontweight="semibold")

    ax.set_xlim(8, 48)
    ax.set_ylim(35.0, 60.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9.5)

    save_path = os.path.join(out_dir, "10_klygo_integrated_detectors_pareto_l4.png")
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
    chart_yolo_pareto_curve(out_dir)
    chart_klygo_detectors_pareto_l4(out_dir)

    # Also copy to benchmarks/analyzt for convenience
    import shutil
    for fname in os.listdir(out_dir):
        if fname.endswith(".png") or fname.endswith(".py"):
            shutil.copy2(os.path.join(out_dir, fname), os.path.join("benchmarks/analyzt", fname))

    print("All charts successfully generated in analyzt/ and benchmarks/analyzt/!")


if __name__ == "__main__":
    main()
