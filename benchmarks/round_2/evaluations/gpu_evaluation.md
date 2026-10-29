# Klygo Benchmark Round 2: GPU Acceleration & VRAM Evaluation

*Evaluated on 2026-10-29 22:05:00*

Target Hardware: **NVIDIA Quadro M1200 (4GB GDDR5 VRAM, WDDM Driver 572.76, CUDA 12.8)**

---

## 1. GPU Throughput & Latency

GPU acceleration brings higher throughput and lower per-frame latency across all architectures:

### YOLOv8n
- **Raw Non-stream**: 11.31 FPS (88.3 ms/frame, ±0.04 std)
- **Raw Stream**: 10.84 FPS (92.5 ms/frame, ±0.51 std) — TTFF: 98.0 ms
- **Klygo Native Non-stream**: 8.29 FPS (120.7 ms/frame, ±0.02 std)
- **Klygo Native Stream**: 7.99 FPS (125.3 ms/frame, ±0.26 std)
- **Klygo Managed Non-stream**: 8.68 FPS (115.2 ms/frame, ±0.18 std)
- **Klygo Managed Stream**: 8.65 FPS (115.6 ms/frame, ±0.14 std) — TTFF: 310.0 ms

### YOLO-World v2
- **Raw Non-stream**: 4.26 FPS (234.9 ms/frame, ±0.03 std)
- **Raw Stream**: 4.16 FPS (240.3 ms/frame, ±0.04 std) — TTFF: 443.7 ms
- **Klygo Native Non-stream**: 3.86 FPS (259.0 ms/frame, ±0.00 std)
- **Klygo Managed Non-stream**: 3.99 FPS (250.8 ms/frame, ±0.05 std)
- **Klygo Managed Stream**: 4.13 FPS (241.9 ms/frame, ±0.06 std) — TTFF: 410.5 ms — **Overhead: 3.0%**

### Grounding DINO Tiny
- **Raw Non-stream**: 0.95 FPS (1044.1 ms/frame, ±0.01 std)
- **Klygo Managed Stream**: 0.14 FPS (7028.5 ms/frame, ±0.00 std) — TTFF: 7193.3 ms
*(Note: Klygo Grounding DINO performs text prompt tokenization and coordinate transformation on-the-fly per batch rather than consuming pre-tokenized raw tensors).*

---

## 2. VRAM Containment & Bounded Allocations

- **YOLOv8n**: Net VRAM delta during inference is **15.7 MB** for Klygo Managed modes compared to **25.7 MB** for Raw/Native modes, confirming that Klygo's streaming iterator drops temporary frame tensors immediately.
- **YOLO-World**: Net VRAM delta is **33.4 MB** for Klygo Managed vs **41.3 MB** for Raw/Native.
- **Grounding DINO**: Stays comfortably within **2.2 GB VRAM**, safely below the 4 GB hardware ceiling of the Quadro M1200 without OOM or throttling.
