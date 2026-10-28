# Multi-GPU & Heterogeneous Accelerator Benchmarking Guide for Klygo

This reference guide provides operational instructions, hardware architecture considerations, and performance tuning strategies for benchmarking Klygo across diverse GPUs and accelerators.

---

## 1. Supported Accelerators & Architecture Profiles

| Accelerator Family | Typical VRAM | Compute Capabilities | Recommended Models | Batch & Stream Tuning |
| :--- | :---: | :---: | :--- | :--- |
| **NVIDIA RTX 4090 / 3090** | 24 GB | Ada / Ampere (FP16, BF16, TF32) | All 5 architectures (including full batch Grounding DINO) | `batch=4` to `16`, `stream=True` for high-FPS RTSP |
| **NVIDIA A100 / H100 (Cloud/Cluster)** | 40–80 GB | Ampere / Hopper | All 5 architectures in full resolution (640–1280px) | `batch=16` to `64`, multi-card parallel evaluation |
| **NVIDIA T4 / V100 (Google Colab)** | 16 GB | Turing / Volta (FP16 Tensor Cores) | YOLOv8, YOLO-World, Faster R-CNN, D-FINE Nano | `batch=1` to `4`, `stream=True` |
| **NVIDIA Quadro / Mobile GPUs** | 4–6 GB | Maxwell / Pascal (Limited VRAM) | YOLOv8n, Faster R-CNN MobileNetV3, D-FINE Nano | `batch=1`, `imgsz=320`, `stream=True` to prevent OOM |
| **Apple Silicon (M1/M2/M3/M4 Max)** | Unified (16–128 GB) | MPS (Metal Performance Shaders) | YOLOv8, YOLO-World, Faster R-CNN | PyTorch MPS engine, `device="mps"` |
| **CPU Fallback (x86_64 / ARM64)** | Host RAM | AVX-512 / NEON | Lightweight models (YOLOv8n, D-FINE Nano) | `batch=1`, `workers=4`, `native=False` |

---

## 2. Multi-GPU Device Addressing in Klygo

### 2.1. Addressing Single and Multiple Devices
Klygo's unified benchmark tool detects and addresses devices through standard PyTorch strings:
- `--device auto`: Automatically queries `torch.cuda.is_available()`. Selects `cuda:0` if available, falls back to `mps`, then `cpu`.
- `--device all-gpus`: Iterates over every detected GPU index (`cuda:0`, `cuda:1`, ..., `cuda:N-1`), runs the full benchmark matrix on each card independently, and compiles a side-by-side comparison scorecard.
- `--device cuda:0`: Targets primary GPU index 0.
- `--device cuda:1`: Targets secondary GPU index 1.
- `--device cpu`: Forces CPU execution across all model architectures.

### 2.2. Setting Active GPU via Environment
To isolate a specific GPU before running python:
```bash
# Expose only GPU 1 to CUDA
export CUDA_VISIBLE_DEVICES=1
python examples/integration/multi_backend_benchmark.py --device cuda:0
```
On Windows PowerShell:
```powershell
$env:CUDA_VISIBLE_DEVICES="1"
python examples/integration/multi_backend_benchmark.py --device cuda:0
```

---

## 3. Managing Model Lifecycles Across GPUs

### 3.1. Clean Memory Teardown (`unload()` & Cache Eviction)
When switching models or devices in the same process:
1. Call `detector.unload()` on the Klygo detector.
2. Synchronize the active GPU:
   ```python
   torch.cuda.synchronize(device_str)
   ```
3. PyTorch automatically frees unreferenced tensors, and `detector.unload()` purges framework references.

### 3.2. Preventing Out-Of-Memory (OOM) on Limited VRAM
- Grounding DINO tiny and large Transformer models require ~2.8 GB VRAM. On GPUs with 4 GB VRAM (such as Quadro M1200), keep `batch=1` and ensure other VRAM consumers are closed.
- Always use `stream=True` (`Managed Stream` or `Native Stream`) when evaluating videos or image series exceeding 50 frames to ensure Python GC can reclaim frame memory.

---

## 4. Multi-GPU Execution Recipes

### Recipe 1: Quick Verification on Active GPU
```bash
python examples/integration/multi_backend_benchmark.py --device auto --models yolo,faster-rcnn --repeats 1 --max-frames 5
```

### Recipe 2: Comprehensive All-GPU Evaluation
```bash
python examples/integration/multi_backend_benchmark.py --device all-gpus --models all --modes all --repeats 3 --max-frames 10 --output artifacts/benchmark/multi_gpu_report.json --markdown artifacts/benchmark/multi_gpu_report.md
```

### Recipe 3: High-Resolution Precision Comparison
```bash
python examples/integration/multi_backend_benchmark.py --device cuda:0 --models yolo,yolo-world,d-fine --imgsz 640 --repeats 3 --max-frames 20
```
