# NVIDIA L4 Complete Evaluation Report (2026-11-15)

- **GPU**: NVIDIA L4 (22563.1 MB VRAM, `is_nvidia_l4=True`)
- **OS / Python / CUDA**: `Linux-6.6.159+-x86_64-with-glibc2.39` / Python `3.13.16` / CUDA `13.0`
- **Branch / Commit**: `detection-evaluation` @ `7187774a40f838d14dea48671489a49bbd97edfb`
- **Note**: In `klygo/metrics/evaluation.py`, `confusion_matrix` uses `int32` on CuPy CUDA 13 (`cupy.add.at` does not support `int64`). `locate-anything` was benchmarked on the same NVIDIA L4 (`62c2567`) in an isolated venv (`transformers==4.57.1`), while the other 8 models ran in Colab's main env (`transformers==5.22.1`). Peak VRAM for `locate-anything`: `7910.11 MiB` (`float16`), `15031.63 MiB` (`float32`).

## 1. Model Inference Runs (9 Models x `float32` / `float16`)

| Model | Dtype | Status | Actual Dtype | Mode | FPS (Median) | p50 (ms) | Parity |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| yolo | float32 | ok | torch.float32 | raw-non-stream | 100.32 | 10.34 | True |
| yolo | float32 | ok | torch.float32 | raw-stream | 99.92 | 9.94 | True |
| yolo | float32 | ok | torch.float32 | native-non-stream | 90.0 | 11.93 | True |
| yolo | float32 | ok | torch.float32 | native-stream | 90.78 | 11.49 | True |
| yolo | float32 | ok | torch.float32 | managed-non-stream | 89.99 | 11.37 | True |
| yolo | float32 | ok | torch.float32 | managed-stream | 92.0 | 11.12 | True |
| yolo | float16 | ok | torch.float16 | raw-non-stream | 96.9 | 10.34 | True |
| yolo | float16 | ok | torch.float16 | raw-stream | 98.22 | 10.23 | True |
| yolo | float16 | ok | torch.float16 | native-non-stream | 86.3 | 11.52 | True |
| yolo | float16 | ok | torch.float16 | native-stream | 87.78 | 11.25 | True |
| yolo | float16 | ok | torch.float16 | managed-non-stream | 87.6 | 11.05 | True |
| yolo | float16 | ok | torch.float16 | managed-stream | 87.5 | 11.02 | True |
| yolo11 | float32 | ok | torch.float32 | raw-non-stream | 85.87 | 11.46 | True |
| yolo11 | float32 | ok | torch.float32 | raw-stream | 86.08 | 11.3 | True |
| yolo11 | float32 | ok | torch.float32 | native-non-stream | 76.99 | 12.49 | True |
| yolo11 | float32 | ok | torch.float32 | native-stream | 77.77 | 12.17 | True |
| yolo11 | float32 | ok | torch.float32 | managed-non-stream | 78.36 | 12.13 | True |
| yolo11 | float32 | ok | torch.float32 | managed-stream | 78.67 | 12.18 | True |
| yolo11 | float16 | ok | torch.float16 | raw-non-stream | 85.62 | 11.96 | True |
| yolo11 | float16 | ok | torch.float16 | raw-stream | 84.52 | 12.16 | True |
| yolo11 | float16 | ok | torch.float16 | native-non-stream | 74.56 | 13.91 | True |
| yolo11 | float16 | ok | torch.float16 | native-stream | 76.74 | 13.26 | True |
| yolo11 | float16 | ok | torch.float16 | managed-non-stream | 76.21 | 13.08 | True |
| yolo11 | float16 | ok | torch.float16 | managed-stream | 75.67 | 13.18 | True |
| yolo26 | float32 | ok | torch.float32 | raw-non-stream | 79.18 | 12.06 | True |
| yolo26 | float32 | ok | torch.float32 | raw-stream | 77.74 | 12.33 | True |
| yolo26 | float32 | ok | torch.float32 | native-non-stream | 72.84 | 13.57 | True |
| yolo26 | float32 | ok | torch.float32 | native-stream | 73.01 | 13.26 | True |
| yolo26 | float32 | ok | torch.float32 | managed-non-stream | 73.25 | 13.17 | True |
| yolo26 | float32 | ok | torch.float32 | managed-stream | 73.91 | 13.17 | True |
| yolo26 | float16 | ok | torch.float16 | raw-non-stream | 75.98 | 12.87 | True |
| yolo26 | float16 | ok | torch.float16 | raw-stream | 76.26 | 13.25 | True |
| yolo26 | float16 | ok | torch.float16 | native-non-stream | 68.49 | 14.43 | True |
| yolo26 | float16 | ok | torch.float16 | native-stream | 68.68 | 14.26 | True |
| yolo26 | float16 | ok | torch.float16 | managed-non-stream | 68.84 | 14.45 | True |
| yolo26 | float16 | ok | torch.float16 | managed-stream | 69.26 | 14.35 | True |
| yolo-world | float32 | ok | torch.float32 | raw-non-stream | 79.86 | 12.25 | True |
| yolo-world | float32 | ok | torch.float32 | raw-stream | 78.85 | 12.61 | True |
| yolo-world | float32 | ok | torch.float32 | native-non-stream | 73.21 | 13.25 | True |
| yolo-world | float32 | ok | torch.float32 | native-stream | 71.87 | 13.63 | True |
| yolo-world | float32 | ok | torch.float32 | managed-non-stream | 73.82 | 13.27 | True |
| yolo-world | float32 | ok | torch.float32 | managed-stream | 72.6 | 13.47 | True |
| yolo-world | float16 | ok | torch.float16 | raw-non-stream | 77.77 | 12.73 | True |
| yolo-world | float16 | ok | torch.float16 | raw-stream | 77.19 | 12.72 | True |
| yolo-world | float16 | ok | torch.float16 | native-non-stream | 71.18 | 13.71 | True |
| yolo-world | float16 | ok | torch.float16 | native-stream | 70.94 | 13.96 | True |
| yolo-world | float16 | ok | torch.float16 | managed-non-stream | 71.53 | 13.73 | True |
| yolo-world | float16 | ok | torch.float16 | managed-stream | 71.43 | 13.81 | True |
| grounding-dino | float32 | N/A | torch.float32 | raw-stream | N/A | N/A | N/A |
| grounding-dino | float32 | N/A | torch.float32 | native-stream | N/A | N/A | N/A |
| grounding-dino | float32 | ok | torch.float32 | raw-non-stream | 5.61 | 179.68 | True |
| grounding-dino | float32 | ok | torch.float32 | native-non-stream | 5.56 | 179.83 | True |
| grounding-dino | float32 | ok | torch.float32 | managed-non-stream | 5.55 | 179.15 | True |
| grounding-dino | float32 | ok | torch.float32 | managed-stream | 5.53 | 180.13 | True |
| grounding-dino | float16 | N/A | torch.float16 | raw-stream | N/A | N/A | N/A |
| grounding-dino | float16 | N/A | torch.float16 | native-stream | N/A | N/A | N/A |
| grounding-dino | float16 | ok | torch.float16 | raw-non-stream | 8.11 | 122.46 | True |
| grounding-dino | float16 | ok | torch.float16 | native-non-stream | 8.0 | 125.44 | True |
| grounding-dino | float16 | ok | torch.float16 | managed-non-stream | 7.95 | 129.55 | True |
| grounding-dino | float16 | ok | torch.float16 | managed-stream | 7.94 | 128.26 | True |
| grounding-dino-base | float32 | N/A | torch.float32 | raw-stream | N/A | N/A | N/A |
| grounding-dino-base | float32 | N/A | torch.float32 | native-stream | N/A | N/A | N/A |
| grounding-dino-base | float32 | ok | torch.float32 | raw-non-stream | 4.33 | 228.82 | True |
| grounding-dino-base | float32 | ok | torch.float32 | native-non-stream | 4.28 | 232.63 | True |
| grounding-dino-base | float32 | ok | torch.float32 | managed-non-stream | 4.35 | 233.51 | True |
| grounding-dino-base | float32 | ok | torch.float32 | managed-stream | 4.28 | 232.0 | True |
| grounding-dino-base | float16 | N/A | torch.float16 | raw-stream | N/A | N/A | N/A |
| grounding-dino-base | float16 | N/A | torch.float16 | native-stream | N/A | N/A | N/A |
| grounding-dino-base | float16 | ok | torch.float16 | raw-non-stream | 7.43 | 134.64 | True |
| grounding-dino-base | float16 | ok | torch.float16 | native-non-stream | 7.29 | 136.93 | True |
| grounding-dino-base | float16 | ok | torch.float16 | managed-non-stream | 7.27 | 138.28 | True |
| grounding-dino-base | float16 | ok | torch.float16 | managed-stream | 7.26 | 137.79 | True |
| faster-rcnn | float32 | N/A | torch.float32 | raw-stream | N/A | N/A | N/A |
| faster-rcnn | float32 | N/A | torch.float32 | native-stream | N/A | N/A | N/A |
| faster-rcnn | float32 | ok | torch.float32 | raw-non-stream | 42.91 | 22.77 | True |
| faster-rcnn | float32 | ok | torch.float32 | native-non-stream | 41.38 | 24.1 | True |
| faster-rcnn | float32 | ok | torch.float32 | managed-non-stream | 40.09 | 24.84 | True |
| faster-rcnn | float32 | ok | torch.float32 | managed-stream | 39.67 | 24.88 | True |
| faster-rcnn | float16 | N/A | torch.float16 | raw-stream | N/A | N/A | N/A |
| faster-rcnn | float16 | N/A | torch.float16 | native-stream | N/A | N/A | N/A |
| faster-rcnn | float16 | ok | torch.float16 | raw-non-stream | 48.05 | 20.03 | True |
| faster-rcnn | float16 | ok | torch.float16 | native-non-stream | 46.19 | 20.88 | True |
| faster-rcnn | float16 | ok | torch.float16 | managed-non-stream | 44.85 | 22.39 | True |
| faster-rcnn | float16 | ok | torch.float16 | managed-stream | 44.02 | 22.0 | True |
| d-fine | float32 | N/A | float32 | raw-stream | N/A | N/A | N/A |
| d-fine | float32 | N/A | float32 | native-stream | N/A | N/A | N/A |
| d-fine | float32 | ok | float32 | raw-non-stream | 2.7 | 364.35 | True |
| d-fine | float32 | ok | float32 | native-non-stream | 2.7 | 371.39 | True |
| d-fine | float32 | ok | float32 | managed-non-stream | 2.71 | 370.49 | True |
| d-fine | float32 | ok | float32 | managed-stream | 2.7 | 371.29 | True |
| d-fine | float16 | N/A | float16 | raw-stream | N/A | N/A | N/A |
| d-fine | float16 | N/A | float16 | native-stream | N/A | N/A | N/A |
| d-fine | float16 | ok | float16 | raw-non-stream | 2.61 | 384.72 | True |
| d-fine | float16 | ok | float16 | native-non-stream | 2.6 | 385.39 | True |
| d-fine | float16 | ok | float16 | managed-non-stream | 2.61 | 380.89 | True |
| d-fine | float16 | ok | float16 | managed-stream | 2.6 | 383.3 | True |
| locate-anything | float32 | N/A | torch.float32 | raw-stream | N/A | N/A | N/A |
| locate-anything | float32 | N/A | torch.float32 | native-stream | N/A | N/A | N/A |
| locate-anything | float32 | ok | torch.float32 | raw-non-stream | 0.99 | 1010.96 | True |
| locate-anything | float32 | ok | torch.float32 | native-non-stream | 0.99 | 1011.09 | True |
| locate-anything | float32 | ok | torch.float32 | managed-non-stream | 0.99 | 1011.99 | True |
| locate-anything | float32 | ok | torch.float32 | managed-stream | 0.99 | 1018.6 | True |
| locate-anything | float16 | N/A | torch.float16 | raw-stream | N/A | N/A | N/A |
| locate-anything | float16 | N/A | torch.float16 | native-stream | N/A | N/A | N/A |
| locate-anything | float16 | ok | torch.float16 | raw-non-stream | 2.49 | 406.34 | True |
| locate-anything | float16 | ok | torch.float16 | native-non-stream | 2.48 | 405.96 | True |
| locate-anything | float16 | ok | torch.float16 | managed-non-stream | 2.47 | 409.79 | True |
| locate-anything | float16 | ok | torch.float16 | managed-stream | 2.48 | 404.26 | True |

## 2. Scaling Matrix (`batch`, `workers`, `prefetch`, `packed`) — Selected Highlights

| Model | Dtype | Batch | Workers | Prefetch | Packed | Mode | FPS (Median) | p50 (ms) | Parity |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| yolo | float32 | 1 | 1 | 1 | True | raw-non-stream | 102.23 | 9.52 | True |
| yolo | float32 | 1 | 1 | 1 | True | managed-non-stream | 91.01 | 10.68 | True |
| yolo | float32 | 1 | 1 | 1 | True | managed-stream | 92.98 | 10.66 | True |
| yolo | float32 | 4 | 2 | 2 | False | raw-non-stream | 205.31 | 0.01 | True |
| yolo | float32 | 4 | 2 | 2 | False | managed-non-stream | 217.91 | 0.01 | True |
| yolo | float32 | 4 | 2 | 2 | False | managed-stream | 217.38 | 0.03 | True |
| yolo | float32 | 4 | 2 | 2 | True | raw-non-stream | 208.94 | 0.01 | True |
| yolo | float32 | 4 | 2 | 2 | True | managed-non-stream | 214.28 | 0.08 | True |
| yolo | float32 | 4 | 2 | 2 | True | managed-stream | 222.58 | 0.03 | True |
| yolo | float32 | 8 | 4 | 4 | False | raw-non-stream | 258.06 | 0.01 | True |
| yolo | float32 | 8 | 4 | 4 | False | managed-non-stream | 280.75 | 0.01 | True |
| yolo | float32 | 8 | 4 | 4 | False | managed-stream | 281.51 | 0.03 | True |
| yolo | float32 | 8 | 4 | 4 | True | raw-non-stream | 258.21 | 0.01 | True |
| yolo | float32 | 8 | 4 | 4 | True | managed-non-stream | 269.92 | 0.06 | True |
| yolo | float32 | 8 | 4 | 4 | True | managed-stream | 274.52 | 0.03 | True |
| yolo | float16 | 1 | 1 | 1 | True | raw-non-stream | 97.26 | 9.92 | True |
| yolo | float16 | 1 | 1 | 1 | True | managed-non-stream | 87.62 | 11.28 | True |
| yolo | float16 | 1 | 1 | 1 | True | managed-stream | 89.72 | 11.1 | True |
| yolo | float16 | 4 | 2 | 2 | False | raw-non-stream | 198.9 | 0.01 | True |
| yolo | float16 | 4 | 2 | 2 | False | managed-non-stream | 207.97 | 0.01 | True |
| yolo | float16 | 4 | 2 | 2 | False | managed-stream | 210.67 | 0.04 | True |
| yolo | float16 | 4 | 2 | 2 | True | raw-non-stream | 204.17 | 0.01 | True |
| yolo | float16 | 4 | 2 | 2 | True | managed-non-stream | 208.34 | 0.08 | True |
| yolo | float16 | 4 | 2 | 2 | True | managed-stream | 217.83 | 0.03 | True |
| yolo | float16 | 8 | 4 | 4 | False | raw-non-stream | 255.45 | 0.01 | True |
| yolo | float16 | 8 | 4 | 4 | False | managed-non-stream | 278.32 | 0.01 | True |
| yolo | float16 | 8 | 4 | 4 | False | managed-stream | 279.67 | 0.03 | True |
| yolo | float16 | 8 | 4 | 4 | True | raw-non-stream | 252.59 | 0.01 | True |
| yolo | float16 | 8 | 4 | 4 | True | managed-non-stream | 268.28 | 0.07 | True |
| yolo | float16 | 8 | 4 | 4 | True | managed-stream | 274.59 | 0.03 | True |
| yolo11 | float32 | 1 | 1 | 1 | True | raw-non-stream | 83.68 | 11.92 | True |
| yolo11 | float32 | 1 | 1 | 1 | True | managed-non-stream | 76.51 | 13.31 | True |
| yolo11 | float32 | 1 | 1 | 1 | True | managed-stream | 77.48 | 13.32 | True |
| yolo11 | float32 | 4 | 2 | 2 | False | raw-non-stream | 183.65 | 0.01 | True |
| yolo11 | float32 | 4 | 2 | 2 | False | managed-non-stream | 191.09 | 0.01 | True |
| yolo11 | float32 | 4 | 2 | 2 | False | managed-stream | 188.9 | 0.03 | True |
| yolo11 | float32 | 4 | 2 | 2 | True | raw-non-stream | 187.23 | 0.01 | True |
| yolo11 | float32 | 4 | 2 | 2 | True | managed-non-stream | 189.84 | 0.08 | True |
| yolo11 | float32 | 4 | 2 | 2 | True | managed-stream | 197.2 | 0.03 | True |
| yolo11 | float32 | 8 | 4 | 4 | False | raw-non-stream | 241.97 | 0.01 | True |
| yolo11 | float32 | 8 | 4 | 4 | False | managed-non-stream | 255.35 | 0.01 | True |
| yolo11 | float32 | 8 | 4 | 4 | False | managed-stream | 262.34 | 0.03 | True |
| yolo11 | float32 | 8 | 4 | 4 | True | raw-non-stream | 230.34 | 0.01 | True |
| yolo11 | float32 | 8 | 4 | 4 | True | managed-non-stream | 245.34 | 0.06 | True |
| yolo11 | float32 | 8 | 4 | 4 | True | managed-stream | 253.23 | 0.03 | True |
| yolo11 | float16 | 1 | 1 | 1 | True | raw-non-stream | 79.87 | 12.63 | True |
| yolo11 | float16 | 1 | 1 | 1 | True | managed-non-stream | 71.01 | 13.65 | True |
| yolo11 | float16 | 1 | 1 | 1 | True | managed-stream | 71.24 | 13.88 | True |
| yolo11 | float16 | 4 | 2 | 2 | False | raw-non-stream | 185.49 | 0.01 | True |
| yolo11 | float16 | 4 | 2 | 2 | False | managed-non-stream | 176.23 | 0.01 | True |
| yolo11 | float16 | 4 | 2 | 2 | False | managed-stream | 178.16 | 0.03 | True |
| yolo11 | float16 | 4 | 2 | 2 | True | raw-non-stream | 186.91 | 0.01 | True |
| yolo11 | float16 | 4 | 2 | 2 | True | managed-non-stream | 177.14 | 0.08 | True |
| yolo11 | float16 | 4 | 2 | 2 | True | managed-stream | 181.84 | 0.03 | True |
| yolo11 | float16 | 8 | 4 | 4 | False | raw-non-stream | 238.41 | 0.01 | True |
| yolo11 | float16 | 8 | 4 | 4 | False | managed-non-stream | 254.63 | 0.01 | True |
| yolo11 | float16 | 8 | 4 | 4 | False | managed-stream | 252.71 | 0.03 | True |
| yolo11 | float16 | 8 | 4 | 4 | True | raw-non-stream | 241.3 | 0.01 | True |
| yolo11 | float16 | 8 | 4 | 4 | True | managed-non-stream | 250.06 | 0.07 | True |
| yolo11 | float16 | 8 | 4 | 4 | True | managed-stream | 258.31 | 0.03 | True |
| yolo26 | float32 | 1 | 1 | 1 | True | raw-non-stream | 78.77 | 12.55 | True |
| yolo26 | float32 | 1 | 1 | 1 | True | managed-non-stream | 71.7 | 13.62 | True |
| yolo26 | float32 | 1 | 1 | 1 | True | managed-stream | 71.26 | 13.59 | True |
| yolo26 | float32 | 4 | 2 | 2 | False | raw-non-stream | 173.11 | 0.01 | True |
| yolo26 | float32 | 4 | 2 | 2 | False | managed-non-stream | 176.13 | 0.01 | True |
| yolo26 | float32 | 4 | 2 | 2 | False | managed-stream | 180.81 | 0.03 | True |
| yolo26 | float32 | 4 | 2 | 2 | True | raw-non-stream | 177.3 | 0.01 | True |
| yolo26 | float32 | 4 | 2 | 2 | True | managed-non-stream | 178.57 | 0.08 | True |
| yolo26 | float32 | 4 | 2 | 2 | True | managed-stream | 184.08 | 0.03 | True |
| yolo26 | float32 | 8 | 4 | 4 | False | raw-non-stream | 266.78 | 0.01 | True |
| yolo26 | float32 | 8 | 4 | 4 | False | managed-non-stream | 251.73 | 0.01 | True |
| yolo26 | float32 | 8 | 4 | 4 | False | managed-stream | 250.88 | 0.03 | True |
| yolo26 | float32 | 8 | 4 | 4 | True | raw-non-stream | 227.45 | 0.01 | True |
| yolo26 | float32 | 8 | 4 | 4 | True | managed-non-stream | 240.16 | 0.07 | True |
| yolo26 | float32 | 8 | 4 | 4 | True | managed-stream | 246.23 | 0.03 | True |
| yolo26 | float16 | 1 | 1 | 1 | True | raw-non-stream | 76.28 | 13.44 | True |
| yolo26 | float16 | 1 | 1 | 1 | True | managed-non-stream | 67.44 | 14.6 | True |
| yolo26 | float16 | 1 | 1 | 1 | True | managed-stream | 69.34 | 14.35 | True |
| yolo26 | float16 | 4 | 2 | 2 | False | raw-non-stream | 181.56 | 0.01 | True |
| yolo26 | float16 | 4 | 2 | 2 | False | managed-non-stream | 172.74 | 0.01 | True |
| yolo26 | float16 | 4 | 2 | 2 | False | managed-stream | 176.18 | 0.03 | True |
| yolo26 | float16 | 4 | 2 | 2 | True | raw-non-stream | 180.82 | 0.01 | True |
| yolo26 | float16 | 4 | 2 | 2 | True | managed-non-stream | 171.69 | 0.08 | True |
| yolo26 | float16 | 4 | 2 | 2 | True | managed-stream | 178.46 | 0.03 | True |
| yolo26 | float16 | 8 | 4 | 4 | False | raw-non-stream | 228.78 | 0.01 | True |
| yolo26 | float16 | 8 | 4 | 4 | False | managed-non-stream | 244.86 | 0.01 | True |
| yolo26 | float16 | 8 | 4 | 4 | False | managed-stream | 243.02 | 0.03 | True |
| yolo26 | float16 | 8 | 4 | 4 | True | raw-non-stream | 225.83 | 0.01 | True |
| yolo26 | float16 | 8 | 4 | 4 | True | managed-non-stream | 234.02 | 0.06 | True |
| yolo26 | float16 | 8 | 4 | 4 | True | managed-stream | 241.32 | 0.03 | True |
| yolo-world | float32 | 1 | 1 | 1 | True | raw-non-stream | 80.07 | 12.22 | True |
| yolo-world | float32 | 1 | 1 | 1 | True | managed-non-stream | 72.09 | 13.94 | True |
| yolo-world | float32 | 1 | 1 | 1 | True | managed-stream | 73.56 | 13.75 | True |
| yolo-world | float32 | 4 | 2 | 2 | False | raw-non-stream | 187.92 | 0.01 | True |
| yolo-world | float32 | 4 | 2 | 2 | False | managed-non-stream | 180.07 | 0.01 | True |
| yolo-world | float32 | 4 | 2 | 2 | False | managed-stream | 180.25 | 0.03 | True |
| yolo-world | float32 | 4 | 2 | 2 | True | raw-non-stream | 190.75 | 0.01 | True |
| yolo-world | float32 | 4 | 2 | 2 | True | managed-non-stream | 179.43 | 0.08 | True |
| yolo-world | float32 | 4 | 2 | 2 | True | managed-stream | 181.32 | 0.03 | True |
| yolo-world | float32 | 8 | 4 | 4 | False | raw-non-stream | 202.55 | 0.01 | True |
| yolo-world | float32 | 8 | 4 | 4 | False | managed-non-stream | 195.86 | 0.01 | True |
| yolo-world | float32 | 8 | 4 | 4 | False | managed-stream | 195.78 | 0.02 | True |
| yolo-world | float32 | 8 | 4 | 4 | True | raw-non-stream | 205.97 | 0.01 | True |
| yolo-world | float32 | 8 | 4 | 4 | True | managed-non-stream | 191.68 | 0.07 | True |
| yolo-world | float32 | 8 | 4 | 4 | True | managed-stream | 197.03 | 0.03 | True |
| yolo-world | float16 | 1 | 1 | 1 | True | raw-non-stream | 78.12 | 12.65 | True |
| yolo-world | float16 | 1 | 1 | 1 | True | managed-non-stream | 69.54 | 13.77 | True |
| yolo-world | float16 | 1 | 1 | 1 | True | managed-stream | 71.82 | 13.5 | True |
| yolo-world | float16 | 4 | 2 | 2 | False | raw-non-stream | 185.02 | 0.01 | True |
| yolo-world | float16 | 4 | 2 | 2 | False | managed-non-stream | 179.66 | 0.01 | True |
| yolo-world | float16 | 4 | 2 | 2 | False | managed-stream | 182.88 | 0.03 | True |
| yolo-world | float16 | 4 | 2 | 2 | True | raw-non-stream | 186.77 | 0.01 | True |
| yolo-world | float16 | 4 | 2 | 2 | True | managed-non-stream | 177.43 | 0.08 | True |
| yolo-world | float16 | 4 | 2 | 2 | True | managed-stream | 181.6 | 0.03 | True |
| yolo-world | float16 | 8 | 4 | 4 | False | raw-non-stream | 262.83 | 0.01 | True |
| yolo-world | float16 | 8 | 4 | 4 | False | managed-non-stream | 248.8 | 0.01 | True |
| yolo-world | float16 | 8 | 4 | 4 | False | managed-stream | 249.76 | 0.02 | True |
| yolo-world | float16 | 8 | 4 | 4 | True | raw-non-stream | 259.53 | 0.01 | True |
| yolo-world | float16 | 8 | 4 | 4 | True | managed-non-stream | 233.07 | 0.08 | True |
| yolo-world | float16 | 8 | 4 | 4 | True | managed-stream | 242.5 | 0.03 | True |

## 3. DetectionGroup / Postprocessing / Transfer

| Device | group_keep_ms | post_pipeline_ms | combine_ms | Status |
| :--- | :---: | :---: | :---: | :---: |
| cpu | 1.9049 | 22.4791 | 17.8223 | passed |
| cuda | 45.1931 | 45.0545 | 176.6401 | passed |
| transfer | 2.5639 | 5.9602 | 0.0 | passed |

## 4. Datasets / Media / Visual / Tabulate

| Operation | Median (ms) | Status |
| :--- | :---: | :---: |
| datasets.load_2x_single | 12.5465 | passed |
| datasets.load_group | 8.9749 | passed |
| media.info | 0.1285 | passed |
| media.save_images_letterbox | 11.6655 | passed |
| visual.compare_images | 9.828 | passed |
| utils.tabulate | 1.9552 | passed |

## 5. `klygo.metrics` (`40` configurations)

All 40 configurations (`empty_0_0`, `small_100_25`, `medium_500_150`, `large_1500_500` across `cpu`/`cuda`, `fp16`/`fp32`/`fp64`, `packed=False/True`) passed.

