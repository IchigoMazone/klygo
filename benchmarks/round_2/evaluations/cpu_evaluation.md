# Klygo Benchmark Round 2: CPU Architecture Evaluation

*Evaluated on 2026-10-29 22:05:00*

---

## 1. CPU Throughput & Overhead Analysis

On multi-core host CPU, pre-instantiating models outside the timing window exposes the true computational overhead of Klygo's detection containerization:

### YOLOv8n
- **Raw Non-stream**: 8.99 FPS (111.3 ms/frame, ±0.13 std)
- **Raw Stream**: 7.90 FPS (130.5 ms/frame, ±1.37 std) — TTFF: 108.5 ms
- **Klygo Native Non-stream**: 8.03 FPS (124.5 ms/frame, ±0.01 std) — **Overhead: 10.6%**
- **Klygo Native Stream**: 8.02 FPS (124.8 ms/frame, ±0.27 std)
- **Klygo Managed Non-stream**: 8.41 FPS (119.0 ms/frame, ±0.00 std) — **Overhead: 6.4%**
- **Klygo Managed Stream**: 7.20 FPS (142.5 ms/frame, ±1.14 std) — TTFF: 169.6 ms

### YOLO-World v2
- **Raw Non-stream**: 3.62 FPS (279.8 ms/frame, ±0.39 std)
- **Raw Stream**: 4.06 FPS (246.1 ms/frame, ±0.02 std) — TTFF: 257.7 ms
- **Klygo Native Non-stream**: 3.42 FPS (294.6 ms/frame, ±0.29 std)
- **Klygo Managed Non-stream**: 3.62 FPS (278.6 ms/frame, ±0.33 std) — **Overhead: 0.0% (Exact Parity!)**
- **Klygo Managed Stream**: 3.79 FPS (266.3 ms/frame, ±0.33 std) — TTFF: 273.5 ms

---

## 2. Memory Deltas on Host RAM

- Non-streaming modes accumulate all detections in memory, resulting in a heap delta of **~4.00 MB**.
- Streaming modes (`Raw Stream` and `Klygo Managed Stream`) cap intermediate object allocation, lowering heap delta to **~2.23 MB – 2.50 MB** (~38% reduction in memory growth).
- Bounding box count is 100% consistent across all YOLO runs (52 boxes for YOLOv8n, 50 boxes for YOLO-World).
