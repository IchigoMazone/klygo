# Klygo Intra-Model Benchmark Report
**Timestamp:** 2026-10-29 19:46:51
**Sample Clip:** 4 frames (320x240 resolution) from `artifacts/benchmarks/sample_4frames.mp4`

---
## Device: CPU
### Model: YOLOv8n
| Execution Mode | Status | FPS | Latency (ms/frame) | TTFF (ms) | Peak RAM (MB) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Non-stream** | ✅ Supported | **0.22** | 4467.96 | 17193.68 | 79.41 (Heap) / 713.4 (RSS) | 0.00 |
| **Raw Stream** | ✅ Supported | **2.85** | 350.98 | 806.82 | 3.71 (Heap) / 715.3 (RSS) | 0.00 |
| **Klygo Native Non-stream** | ✅ Supported | **3.43** | 291.50 | 1165.97 | 3.29 (Heap) / 713.0 (RSS) | 0.00 |
| **Klygo Native Stream** | ✅ Supported | **8.43** | 118.67 | 474.64 | 2.26 (Heap) / 697.8 (RSS) | 0.00 |
| **Klygo Managed Non-stream** | ✅ Supported | **7.93** | 126.09 | 504.33 | 2.50 (Heap) / 717.8 (RSS) | 0.00 |
| **Klygo Managed Stream** | ✅ Supported | **5.20** | 192.26 | 285.47 | 2.49 (Heap) / 717.9 (RSS) | 0.00 |

### Model: YOLO-World
| Execution Mode | Status | FPS | Latency (ms/frame) | TTFF (ms) | Peak RAM (MB) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Non-stream** | ✅ Supported | **0.35** | 2861.70 | 1883.44 | 360.62 (Heap) / 838.1 (RSS) | 0.00 |
| **Raw Stream** | ✅ Supported | **0.46** | 2175.51 | 1132.84 | 338.63 (Heap) / 883.6 (RSS) | 0.00 |
| **Klygo Native Non-stream** | ✅ Supported | **0.44** | 2267.48 | 9069.89 | 337.62 (Heap) / 874.1 (RSS) | 0.00 |
| **Klygo Native Stream** | ✅ Supported | **3.76** | 266.17 | 1064.64 | 2.26 (Heap) / 840.4 (RSS) | 0.00 |
| **Klygo Managed Non-stream** | ✅ Supported | **4.07** | 245.61 | 982.42 | 2.49 (Heap) / 843.2 (RSS) | 0.00 |
| **Klygo Managed Stream** | ✅ Supported | **3.84** | 260.69 | 285.11 | 2.50 (Heap) / 844.0 (RSS) | 0.00 |

### Model: Grounding DINO Tiny
| Execution Mode | Status | FPS | Latency (ms/frame) | TTFF (ms) | Peak RAM (MB) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Non-stream** | ✅ Supported | **0.63** | 1585.40 | 6306.52 | 1.29 (Heap) / 1421.2 (RSS) | 0.00 |
| **Raw Stream** | ❌ Not Supported by Upstream | - | - | - | - | - |
| **Klygo Native Non-stream** | ✅ Supported | **0.52** | 1916.49 | 7665.94 | 0.79 (Heap) / 2083.2 (RSS) | 0.00 |
| **Klygo Native Stream** | ❌ Not Supported (No Upstream Engine) | - | - | - | - | - |
| **Klygo Managed Non-stream** | ✅ Supported | **0.53** | 1888.81 | 7555.23 | 1.17 (Heap) / 1983.8 (RSS) | 0.00 |
| **Klygo Managed Stream** | ✅ Supported | **0.50** | 1983.76 | 4125.20 | 1.17 (Heap) / 2077.5 (RSS) | 0.00 |

## Device: CUDA:0
### Model: YOLOv8n
| Execution Mode | Status | FPS | Latency (ms/frame) | TTFF (ms) | Peak RAM (MB) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Non-stream** | ✅ Supported | **1.18** | 849.28 | 3117.75 | 4.16 (Heap) / 1281.5 (RSS) | 37.83 |
| **Raw Stream** | ✅ Supported | **2.73** | 365.79 | 896.30 | 3.94 (Heap) / 1292.7 (RSS) | 37.82 |
| **Klygo Native Non-stream** | ✅ Supported | **3.20** | 312.48 | 1249.81 | 3.29 (Heap) / 1283.4 (RSS) | 37.83 |
| **Klygo Native Stream** | ✅ Supported | **6.13** | 163.26 | 652.91 | 2.48 (Heap) / 1284.8 (RSS) | 37.82 |
| **Klygo Managed Non-stream** | ✅ Supported | **7.20** | 138.98 | 555.78 | 2.50 (Heap) / 1284.7 (RSS) | 27.81 |
| **Klygo Managed Stream** | ✅ Supported | **6.87** | 145.64 | 287.58 | 2.49 (Heap) / 1284.7 (RSS) | 27.81 |

### Model: YOLO-World
| Execution Mode | Status | FPS | Latency (ms/frame) | TTFF (ms) | Peak RAM (MB) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Non-stream** | ✅ Supported | **0.39** | 2562.31 | 3069.27 | 338.63 (Heap) / 1512.9 (RSS) | 121.50 |
| **Raw Stream** | ✅ Supported | **0.43** | 2314.97 | 1540.82 | 338.63 (Heap) / 1640.3 (RSS) | 121.50 |
| **Klygo Native Non-stream** | ✅ Supported | **0.45** | 2241.57 | 8966.15 | 337.62 (Heap) / 1370.2 (RSS) | 121.50 |
| **Klygo Native Stream** | ✅ Supported | **3.41** | 292.97 | 1171.79 | 2.47 (Heap) / 1370.3 (RSS) | 121.50 |
| **Klygo Managed Non-stream** | ✅ Supported | **3.74** | 267.49 | 1069.85 | 2.49 (Heap) / 1370.3 (RSS) | 114.84 |
| **Klygo Managed Stream** | ✅ Supported | **3.70** | 270.12 | 399.35 | 2.49 (Heap) / 1370.3 (RSS) | 114.84 |

### Model: Grounding DINO Tiny
| Execution Mode | Status | FPS | Latency (ms/frame) | TTFF (ms) | Peak RAM (MB) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Non-stream** | ✅ Supported | **0.76** | 1317.66 | 5270.42 | 1.11 (Heap) / 1610.7 (RSS) | 1093.73 |
| **Raw Stream** | ❌ Not Supported by Upstream | - | - | - | - | - |
| **Klygo Native Non-stream** | ✅ Supported | **1.02** | 983.97 | 3935.74 | 0.96 (Heap) / 1626.6 (RSS) | 1551.43 |
| **Klygo Native Stream** | ❌ Not Supported (No Upstream Engine) | - | - | - | - | - |
| **Klygo Managed Non-stream** | ✅ Supported | **1.02** | 976.32 | 3905.14 | 0.94 (Heap) / 1627.0 (RSS) | 1551.43 |
| **Klygo Managed Stream** | ✅ Supported | **1.03** | 974.07 | 2031.44 | 1.17 (Heap) / 1602.9 (RSS) | 1551.43 |
