# CPU Execution Evaluation & Analysis

**Run ID:** `run_20261029_194342`  
**Device:** Host CPU (Multi-core)  
**Sample Test Clip:** 4 frames (320x240 resolution) from `sample_4frames.mp4`

---

## 1. Measured Metrics Summary (CPU)

| Model | Mode | FPS | Latency (ms) | TTFF (ms) | Heap Peak (MB) | RAM RSS (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n** | Raw Non-stream | 0.22 | 4467.96 | 17193.68 | 79.41 | 713.4 |
| | Raw Stream | 2.85 | 350.98 | 806.82 | 3.71 | 715.3 |
| | Klygo Native Non-stream | 3.43 | 291.50 | 1165.97 | 3.29 | 713.0 |
| | Klygo Native Stream | **8.43** | **118.67** | 474.64 | **2.26** | 697.8 |
| | Klygo Managed Non-stream | 7.93 | 126.09 | 504.33 | 2.50 | 717.8 |
| | Klygo Managed Stream | 5.20 | 192.26 | **285.47** | 2.49 | 717.9 |
| **YOLO-World v2** | Raw Non-stream | 0.35 | 2861.70 | 1883.44 | 360.62 | 838.1 |
| | Raw Stream | 0.46 | 2175.51 | 1132.84 | 338.63 | 883.6 |
| | Klygo Native Non-stream | 0.44 | 2267.48 | 9069.89 | 337.62 | 874.1 |
| | Klygo Native Stream | 3.76 | 266.17 | 1064.64 | **2.26** | 840.4 |
| | Klygo Managed Non-stream | **4.07** | **245.61** | 982.42 | 2.49 | 843.2 |
| | Klygo Managed Stream | 3.84 | 260.69 | **285.11** | 2.50 | 844.0 |
| **Grounding DINO** | Raw Non-stream | 0.63 | 1585.40 | 6306.52 | 1.29 | 1421.2 |
| | Raw Stream | ❌ N/A | - | - | - | - |
| | Klygo Native Non-stream | 0.52 | 1916.49 | 7665.94 | **0.79** | 2083.2 |
| | Klygo Native Stream | ❌ N/A | - | - | - | - |
| | Klygo Managed Non-stream | 0.53 | 1888.81 | 7555.23 | 1.17 | 1983.8 |
| | Klygo Managed Stream | 0.50 | 1983.76 | **4125.20** | 1.17 | 2077.5 |

---

## 2. In-Depth CPU Behavior Findings

1. **CPU Memory Pressure during Non-Streaming**:
   - In non-stream modes, CPU RAM RSS climbed to **2.08 GB** during Grounding DINO execution, while Heap allocations for YOLO-World reached **360 MB** due to tensor buffering in memory.
   - Streaming modes immediately release intermediate activation frames, keeping Heap usage flat at **2.26 – 2.50 MB**.

2. **TTFF Reduction on CPU**:
   - CPU video decoding and frame buffering create massive lag in non-streaming modes (up to **17.2 seconds** for YOLOv8n).
   - In `Klygo Managed Stream`, TTFF on CPU dropped to **285 ms**, making real-time CPU preview viable.

3. **Grounding DINO CPU Bottleneck**:
   - Grounding DINO requires heavy deformable multi-head attention. On CPU, it delivers ~0.5 FPS even on 320x240 clips.
   - Conclusion for CPU: Suitable only for lightweight zero-shot verification, not high-speed video feeds.
