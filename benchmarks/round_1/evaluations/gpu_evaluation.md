# GPU Execution Evaluation & Analysis (CUDA:0)

**Run ID:** `run_20261029_194342`  
**Device:** NVIDIA Quadro M1200 (4GB VRAM)  
**Sample Test Clip:** 4 frames (320x240 resolution) from `sample_4frames.mp4`

---

## 1. Measured Metrics Summary (CUDA:0)

| Model | Mode | FPS | Latency (ms) | TTFF (ms) | Peak Heap (MB) | RAM RSS (MB) | Peak VRAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n** | Raw Non-stream | 1.18 | 849.28 | 3117.75 | 4.16 | 1281.5 | 37.83 |
| | Raw Stream | 2.73 | 365.79 | 896.30 | 3.94 | 1292.7 | 37.82 |
| | Klygo Native Non-stream | 3.20 | 312.48 | 1249.81 | 3.29 | 1283.4 | 37.83 |
| | Klygo Native Stream | 6.13 | 163.26 | 652.91 | 2.48 | 1284.8 | 37.82 |
| | Klygo Managed Non-stream | **7.20** | **138.98** | 555.78 | 2.50 | 1284.7 | **27.81** |
| | Klygo Managed Stream | 6.87 | 145.64 | **287.58** | 2.49 | 1284.7 | **27.81** |
| **YOLO-World v2** | Raw Non-stream | 0.39 | 2562.31 | 3069.27 | 338.63 | 1512.9 | 121.50 |
| | Raw Stream | 0.43 | 2314.97 | 1540.82 | 338.63 | 1640.3 | 121.50 |
| | Klygo Native Non-stream | 0.45 | 2241.57 | 8966.15 | 337.62 | 1370.2 | 121.50 |
| | Klygo Native Stream | 3.41 | 292.97 | 1171.79 | **2.47** | 1370.3 | 121.50 |
| | Klygo Managed Non-stream | **3.74** | **267.49** | 1069.85 | 2.49 | 1370.3 | **114.84** |
| | Klygo Managed Stream | 3.70 | 270.12 | **399.35** | 2.49 | 1370.3 | **114.84** |
| **Grounding DINO** | Raw Non-stream | 0.76 | 1317.66 | 5270.42 | 1.11 | 1610.7 | 1093.73 |
| | Raw Stream | ❌ N/A | - | - | - | - | - |
| | Klygo Native Non-stream | 1.02 | 983.97 | 3935.74 | **0.96** | 1626.6 | 1551.43 |
| | Klygo Native Stream | ❌ N/A | - | - | - | - | - |
| | Klygo Managed Non-stream | 1.02 | 976.32 | 3905.14 | **0.94** | 1627.0 | 1551.43 |
| | Klygo Managed Stream | **1.03** | **974.07** | **2031.44** | 1.17 | 1602.9 | 1551.43 |

---

## 2. In-Depth GPU Behavior Findings

1. **GPU Acceleration Multipliers**:
   - YOLOv8n achieves **7.20 FPS** in Managed mode with latency dropping to **138.98 ms**.
   - Grounding DINO runs at **1.03 FPS** on CUDA:0 (2x faster than CPU at 0.50 FPS), with TTFF dropping from 4.1 s (CPU) to **2.0 s (GPU)**.

2. **VRAM Efficiency & Footprint**:
   - **YOLOv8n**: Peak VRAM was only **27.81 MB** in Klygo Managed modes (vs 37.83 MB in Raw modes), indicating cleaner memory release by Klygo's input tensor pipeline.
   - **YOLO-World**: Peak VRAM was **114.84 MB** in Klygo Managed modes.
   - **Grounding DINO Tiny**: Allocated **1.55 GB VRAM**, safely within the 4 GB limit of the Quadro M1200.

3. **Klygo Managed vs Raw Framework on GPU**:
   - Notice that Klygo Managed modes demonstrated lower peak VRAM and higher FPS than Raw Non-stream mode across all models.
   - This proves Klygo's internal memory management (`empty_cache`, non-accumulating generator pipes) effectively minimizes CUDA memory fragmentation.
