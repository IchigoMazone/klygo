# Klygo Benchmark Round 2: Standardized Performance Scorecards

*Generated on 2026-10-29 22:02:51*

Methodology: Isolated model preloading, symmetric warmup, 3 repeats (Median/Jitter), real 10-frame clip.

## CPU Multi-Core

### Model: `YOLOv8n`

| Execution Mode | Median FPS | FPS Jitter (Std) | Latency (ms/frame) | TTFF (ms) | Peak Heap $\Delta$ | Peak RSS $\Delta$ | Peak VRAM $\Delta$ | Total Boxes | Parity Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Non-stream** | **8.99** | ±0.13 | 111.3 ms | — | 4.00 MB | 0.0 MB | 0.0 MB | 52 | `PASS` |
| **Raw Stream** | **7.90** | ±1.37 | 130.5 ms | 108.5 ms | 2.23 MB | 0.0 MB | 0.0 MB | 52 | `PASS` |
| **Klygo Native Non-stream** | **8.03** | ±0.01 | 124.5 ms | — | 4.02 MB | 17.9 MB | 0.0 MB | 52 | `PASS` |
| **Klygo Native Stream** | **8.02** | ±0.27 | 124.8 ms | 1248.0 ms | 2.32 MB | 0.0 MB | 0.0 MB | 52 | `PASS` |
| **Klygo Managed Non-stream** | **8.41** | ±0.00 | 119.0 ms | — | 2.56 MB | 0.0 MB | 0.0 MB | 52 | `PASS` |
| **Klygo Managed Stream** | **7.20** | ±1.14 | 142.5 ms | 169.6 ms | 2.50 MB | 0.0 MB | 0.0 MB | 52 | `PASS` |

### Model: `YOLO-World`

| Execution Mode | Median FPS | FPS Jitter (Std) | Latency (ms/frame) | TTFF (ms) | Peak Heap $\Delta$ | Peak RSS $\Delta$ | Peak VRAM $\Delta$ | Total Boxes | Parity Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Non-stream** | **3.62** | ±0.39 | 279.8 ms | — | 3.99 MB | 18.9 MB | 0.0 MB | 50 | `PASS` |
| **Raw Stream** | **4.06** | ±0.02 | 246.1 ms | 257.7 ms | 2.22 MB | 7.4 MB | 0.0 MB | 50 | `PASS` |
| **Klygo Native Non-stream** | **3.42** | ±0.29 | 294.6 ms | — | 4.02 MB | 10.1 MB | 0.0 MB | 50 | `PASS` |
| **Klygo Native Stream** | **3.44** | ±0.23 | 292.4 ms | 2924.1 ms | 2.29 MB | 0.0 MB | 0.0 MB | 50 | `PASS` |
| **Klygo Managed Non-stream** | **3.62** | ±0.33 | 278.6 ms | — | 2.53 MB | 0.0 MB | 0.0 MB | 50 | `PASS` |
| **Klygo Managed Stream** | **3.79** | ±0.33 | 266.3 ms | 273.5 ms | 2.50 MB | 1.2 MB | 0.0 MB | 50 | `PASS` |

### Model: `Grounding DINO Tiny`

| Execution Mode | Median FPS | FPS Jitter (Std) | Latency (ms/frame) | TTFF (ms) | Peak Heap $\Delta$ | Peak RSS $\Delta$ | Peak VRAM $\Delta$ | Total Boxes | Parity Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Non-stream** | **0.40** | ±0.01 | 2523.5 ms | — | 0.13 MB | 85.9 MB | 0.0 MB | 155 | `PASS` |
| **Raw Stream** | *Not Supported by Upstream* | — | — | — | — | — | — | — | — |
| **Klygo Native Non-stream** | **0.07** | ±0.00 | 14808.3 ms | — | 0.86 MB | 12.8 MB | 0.0 MB | 119 | `DIFF (119 vs 155)` |
| **Klygo Native Stream** | *Not Supported by Upstream* | — | — | — | — | — | — | — | — |
| **Klygo Managed Non-stream** | **0.07** | ±0.00 | 14519.1 ms | — | 0.84 MB | 0.0 MB | 0.0 MB | 119 | `DIFF (119 vs 155)` |
| **Klygo Managed Stream** | **0.07** | ±0.00 | 14840.1 ms | 14666.8 ms | 0.82 MB | 0.0 MB | 0.0 MB | 119 | `DIFF (119 vs 155)` |

## NVIDIA CUDA GPU (cuda:0)

### Model: `YOLOv8n`

| Execution Mode | Median FPS | FPS Jitter (Std) | Latency (ms/frame) | TTFF (ms) | Peak Heap $\Delta$ | Peak RSS $\Delta$ | Peak VRAM $\Delta$ | Total Boxes | Parity Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Non-stream** | **11.31** | ±0.04 | 88.3 ms | — | 4.00 MB | 1.9 MB | 25.7 MB | 52 | `PASS` |
| **Raw Stream** | **10.84** | ±0.51 | 92.5 ms | 98.0 ms | 2.45 MB | 0.0 MB | 25.7 MB | 52 | `PASS` |
| **Klygo Native Non-stream** | **8.29** | ±0.02 | 120.7 ms | — | 4.03 MB | 0.1 MB | 25.7 MB | 52 | `PASS` |
| **Klygo Native Stream** | **7.99** | ±0.26 | 125.3 ms | 1252.4 ms | 2.54 MB | 0.1 MB | 25.7 MB | 52 | `PASS` |
| **Klygo Managed Non-stream** | **8.68** | ±0.18 | 115.2 ms | — | 2.56 MB | 0.1 MB | 15.7 MB | 52 | `PASS` |
| **Klygo Managed Stream** | **8.65** | ±0.14 | 115.6 ms | 310.0 ms | 2.50 MB | 0.2 MB | 15.7 MB | 52 | `PASS` |

### Model: `YOLO-World`

| Execution Mode | Median FPS | FPS Jitter (Std) | Latency (ms/frame) | TTFF (ms) | Peak Heap $\Delta$ | Peak RSS $\Delta$ | Peak VRAM $\Delta$ | Total Boxes | Parity Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Non-stream** | **4.26** | ±0.03 | 234.9 ms | — | 4.00 MB | 1.0 MB | 41.3 MB | 50 | `PASS` |
| **Raw Stream** | **4.16** | ±0.04 | 240.3 ms | 443.7 ms | 2.45 MB | 0.1 MB | 41.3 MB | 50 | `PASS` |
| **Klygo Native Non-stream** | **3.86** | ±0.00 | 259.0 ms | — | 4.02 MB | 0.0 MB | 41.3 MB | 50 | `PASS` |
| **Klygo Native Stream** | **4.00** | ±0.04 | 249.8 ms | 2498.1 ms | 2.52 MB | 0.0 MB | 41.3 MB | 50 | `PASS` |
| **Klygo Managed Non-stream** | **3.99** | ±0.05 | 250.8 ms | — | 2.53 MB | 4.2 MB | 33.4 MB | 50 | `PASS` |
| **Klygo Managed Stream** | **4.13** | ±0.06 | 241.9 ms | 410.5 ms | 2.50 MB | 4.5 MB | 33.4 MB | 50 | `PASS` |

### Model: `Grounding DINO Tiny`

| Execution Mode | Median FPS | FPS Jitter (Std) | Latency (ms/frame) | TTFF (ms) | Peak Heap $\Delta$ | Peak RSS $\Delta$ | Peak VRAM $\Delta$ | Total Boxes | Parity Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Non-stream** | **0.95** | ±0.01 | 1044.1 ms | — | 0.13 MB | 0.3 MB | 115.8 MB | 154 | `PASS` |
| **Raw Stream** | *Not Supported by Upstream* | — | — | — | — | — | — | — | — |
| **Klygo Native Non-stream** | **0.14** | ±0.00 | 7022.8 ms | — | 0.86 MB | 1.2 MB | 1092.4 MB | 119 | `DIFF (119 vs 154)` |
| **Klygo Native Stream** | *Not Supported by Upstream* | — | — | — | — | — | — | — | — |
| **Klygo Managed Non-stream** | **0.14** | ±0.00 | 7024.9 ms | — | 0.84 MB | 0.3 MB | 1092.4 MB | 119 | `DIFF (119 vs 154)` |
| **Klygo Managed Stream** | **0.14** | ±0.00 | 7028.5 ms | 7193.3 ms | 0.82 MB | 4.2 MB | 1092.4 MB | 119 | `DIFF (119 vs 154)` |
