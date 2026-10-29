# Comprehensive Evaluation & Architectural Analysis

**Run ID:** `run_20261029_194342`  
**Date:** 2026-10-29  
**Target Architectures:** YOLOv8n, YOLO-World v2, Grounding DINO Tiny  
**Devices:** CPU (Multi-core Host) vs GPU (NVIDIA Quadro M1200 4GB)

---

## 1. Executive Summary

This evaluation analyzes the performance and resource footprint across three tiers of execution:
1. **Raw Upstream Framework** (`ultralytics.YOLO`, `transformers.AutoModelForZeroShotObjectDetection`)
2. **Klygo Native Mode** (`det.predict(..., native=True)` with `model_stream=True`)
3. **Klygo Managed Mode** (`det.predict(..., native=False)` with `stream=False` and `stream=True`)

### Key Takeaways
- **First Frame Responsiveness (TTFF)**: Streaming modes yield frame #1 in **285 ms – 399 ms** on CPU/GPU for YOLO models, compared to **1.8 s – 17.2 s** when using blocking non-streaming modes.
- **Upstream Feature Gaps Closed by Klygo**: Hugging Face Grounding DINO does not support lazy streaming natively. `Klygo Managed Stream` fills this void, enabling real-time token/frame processing with 50% lower initial response time (2,031 ms vs 5,270 ms on GPU).
- **Memory Predictability**: Accumulating results without streaming consumes hundreds of megabytes on the Python heap (up to 360 MB for YOLO-World). In contrast, Klygo streaming modes cap Heap memory at a near-constant **~2.2 MB – 2.5 MB**.

---

## 2. Intra-Model Comparison Matrix

| Model | Fastest Mode (FPS) | Lowest Latency | Lowest TTFF | Peak Heap (Stream) | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n** | Klygo Native Stream (8.43 CPU) / Klygo Managed (7.20 GPU) | 118.67 ms (CPU) / 138.98 ms (GPU) | **285.47 ms** (Managed Stream) | **2.26 MB** | **27.81 MB** |
| **YOLO-World v2** | Klygo Managed Non-stream (4.07 CPU / 3.74 GPU) | 245.61 ms (CPU) / 267.49 ms (GPU) | **285.11 ms** (Managed Stream) | **2.26 MB** | **114.84 MB** |
| **Grounding DINO** | Klygo Managed Stream (1.03 GPU) | 974.07 ms (GPU) | **2,031.44 ms** (Managed Stream) | **1.17 MB** | **1,551.43 MB** |

---

## 3. Detailed Architectural Evaluations

### 3.1. Raw vs Klygo Native
- **Behavior**: Both modes invoke the upstream model directly without passing through Klygo's `media` loading and `preprocessing` coordinate transformations.
- **Observation**:
  - In non-stream modes, Raw Upstream incurs noticeable overhead when gathering frames and initializing runtime instances.
  - Klygo Native leverages persistent model instances and caching, achieving higher sustained FPS (e.g., 3.43 vs 0.22 FPS for YOLOv8n on CPU).
  - Passing `model_stream=True` in Klygo Native bypasses framework wrappers cleanly while returning standardized `outputs.Detection` objects.

### 3.2. Native vs Klygo Managed
- **Behavior**: Managed mode introduces Klygo's non-destructive image pipelines (`media.MediaFrames`, `preprocessing.compose`), tracking affine transformation traces (e.g., letterbox, resize, autocontrast) to automatically restore bounding boxes back to the original source coordinates.
- **Observation**:
  - The pipeline overhead of Managed Mode is negligible (< 10 ms per frame), while offering full trace restoration, unified batching, and progress tracking.
  - On GPU, Klygo Managed Non-stream actually outperformed Raw and Native modes (7.20 FPS vs 1.18 / 3.20 FPS for YOLOv8n) due to optimized tensor preparation and device synchronization.

### 3.3. Streaming vs Non-Streaming
- **Non-Stream**:
  - Requires loading and buffering all video frames before yielding detections.
  - Results in high Time To First Frame (TTFF) and large RAM spikes.
  - Suitable only for short offline validation batches where full dataset collation is needed at once.
- **Stream**:
  - Yields detections frame-by-frame as an iterator.
  - Essential for real-time cameras, video processing, and edge deployments.
  - Heap memory usage remains flat and minimal throughout execution.

---

## 4. Production Recommendations

1. **For Real-Time Video & CCTV Streams**:
   - Use **`Klygo Managed Stream`** (`detector.predict(source, stream=True)`).
   - Provides minimal TTFF, flat memory footprint, and non-destructive bounding box restoration back to original camera resolution.

2. **For Zero-Shot Prompt-Based Models (Grounding DINO)**:
   - Always run on **GPU** with `size={"shortest_edge": 320, "longest_edge": 320}` or bounded resolution to prevent CPU deformable attention stalls.
   - Use `stream=True` in Klygo to avoid out-of-memory errors on long videos.

3. **For High-Throughput Batch Inference**:
   - Use **`Klygo Managed Non-stream`** with an explicit batch size (e.g., `batch=4` or `batch=8` on capable GPUs).
