# `klygo.models.backend` & `common` Dispatchers

`klygo.models.backend` is the lazy multi-framework execution layer beneath [`klygo.models`](../README.md) and [`models.Detector`](../api/Detector.md). It isolates framework-specific tensor casting, AMP inference, output normalization, accelerator synchronization, and weight serialization so Klygo's core startup remains fast and dependency-free until a specific backend is invoked.

Import the unified dispatcher module with:

```python
from klygo.models.backend import common
```

Or access individual framework adapters lazily via `from klygo.models import backend` (`backend.common`, `backend.huggingface`, `backend.ultralytics`, `backend.torchvision`, `backend.keras`).

---

## 1. How `common` Connects `Detector` to the 4 Backends

Every [`models.Detector`](../api/Detector.md) instance delegates its hardware, inference, and lifecycle operations directly to [`klygo.models.backend.common`](api/common.md):

```text
models.Detector (or custom Task Base Class)
       │
       ▼
klygo.models.backend.common (15 Unified Dispatch Functions)
       ├──► backend.huggingface ("Hugging Face")  ──► torch_runtime
       ├──► backend.ultralytics ("Ultralytics")   ──► torch_runtime
       ├──► backend.torchvision ("Torchvision")   ──► torch_runtime
       └──► backend.keras       ("Keras" / "KerasHub" / "TensorFlow")
```

### Two Ways to Use `common` Functions

1. **Inside a `Detector` subclass (Recommended for Detection)**:
   Call the bound convenience methods on `self` (`self.current_device`, `self.current_dtype`, `self.cast_inputs()`, `self.run_inference()`, `self.get_output_device()`, `self.sync_device()`, `self.format_results()`, `self.save()`, `self.reset()`, `self.unload()`), which automatically pass `self.backend` and `self.model` to `common.*`.
2. **Directly via `from klygo.models.backend import common` (For New Tasks & Low-Level Orchestration)**:
   When authoring a new task base class (such as a custom `Segmenter`, `Classifier`, or standalone benchmark harness), call `common.<func>(backend, ...)` directly to dispatch across all 4 frameworks without writing `if/elif` branches.

---

## 2. Complete Inventory of `klygo.models.backend.common` (15 Functions)

See the [full `common` API reference](api/common.md) for detailed signatures, parameters, and edge cases.

### A. Backend & Hardware Introspection

| Function in `common.py` | `Detector` Equivalent | Purpose |
| :--- | :--- | :--- |
| [`common.backend_family(backend)`](api/common.md#commonbackend_family) | *(internal)* | Strip suffixes like `" (Offline)"` or `" (In-Memory)"` to return canonical family (`"Hugging Face"`, `"Ultralytics"`, `"Torchvision"`, `"Keras"`). |
| [`common.current_device(model, backend=None)`](api/common.md#commoncurrent_device) | `self.current_device` | Detect the active device (`torch.device` or string such as `"cuda:0"` / `"cpu"`) hosting `model`. |
| [`common.current_dtype(model, backend=None)`](api/common.md#commoncurrent_dtype) | `self.current_dtype` | Detect the active numerical precision (`torch.dtype` or `"float32"` / `"float16"` / `"bfloat16"`) of `model`. |
| [`common.is_accelerator_available(backend, device=None)`](api/common.md#commonis_accelerator_available) | *(used by `benchmark`)* | Check whether GPU/accelerator execution is available and compatible with `device`. |

### B. Accelerator Context & Synchronization

| Function in `common.py` | `Detector` Equivalent | Purpose |
| :--- | :--- | :--- |
| [`common.inference_context(backend)`](api/common.md#commoninference_context) | *(used by `predict` & `benchmark`)* | Return `torch.inference_mode()` for PyTorch-backed frameworks or `nullcontext()` when no runtime context applies. |
| [`common.synchronize(backend, device=None, value=None)`](api/common.md#commonsynchronize) | *(used by `benchmark`)* | Synchronize queued CUDA kernels (`torch.cuda.synchronize()`) or materialize lazy Keras/JAX/TF tensors (`ops.convert_to_numpy(value)`) for accurate latency timing. |
| [`common.clear_cache(backend)`](api/common.md#commonclear_cache) | *(used by `reset` & `unload`)* | Release unused cached GPU memory blocks (`torch.cuda.empty_cache()`) without clearing live weights or the Keras graph session. |

### C. Inference Pipeline Dispatchers

| Function in `common.py` | `Detector` Equivalent | Purpose |
| :--- | :--- | :--- |
| [`common.cast_inputs(backend, inputs, dev, dtype)`](api/common.md#commoncast_inputs) | `self.cast_inputs(inputs, dev, dtype)` | Cast floating-point tensors in `BatchFeature` dicts, lists of tensors (`Torchvision`), or NumPy/Keras tensors to `(dev, dtype)` while preserving integer `input_ids`. |
| [`common.run_inference(backend, model, inputs, cur_dtype, **kwargs)`](api/common.md#commonrun_inference) | `self.run_inference(inputs, **kwargs)` | Execute the forward pass: runs AMP `autocast` for Hugging Face, list-of-tensors call for Torchvision, or `ObjectDetector` aspect-preserving resize + `decode_predictions` / `model(inputs, training=False)` for Keras. |
| [`common.get_output_device(backend, outputs, default_device)`](api/common.md#commonget_output_device) | `self.get_output_device(outputs)` | Inspect `outputs.logits` or `outputs.pred_boxes` to determine which GPU holds the output tensors in multi-GPU setups. |
| [`common.sync_device(backend, tensor, target_device)`](api/common.md#commonsync_device) | `self.sync_device(tensor, target_device)` | Move auxiliary tensors (such as `input_ids`) to `target_device` when `device_map="auto"` places input and output layers on different GPUs. |
| [`common.format_results(backend, raw_outputs)`](api/common.md#commonformat_results) | `self.format_results(raw_outputs)` | Convert raw outputs from Ultralytics `Results`, Torchvision `list[dict]`, or KerasHub `decode_predictions` (including `yxyx` $\rightarrow$ `xyxy` conversion and `-1` padding removal) into standard `[{'boxes', 'scores', 'labels', 'class_ids'}]`. |

### D. Lifecycle & Persistence Dispatchers

| Function in `common.py` | `Detector` Equivalent | Purpose |
| :--- | :--- | :--- |
| [`common.save(backend, model, output_dir, metadata, class_module, processor=None)`](api/common.md#commonsave) | `self.save(output_dir)` | Save a complete Klygo model directory: writes `klygo.json`, copies custom `model.py` if outside `klygo.models.*`, and saves framework weights (`save_pretrained`, `.save`, `state_dict`, or `.weights.h5`). |
| [`common.reset(backend, model, processor=None)`](api/common.md#commonreset) | `self.reset()` | Move `model` back to CPU (`model.cpu()`) and call `clear_cache(backend)`. |
| [`common.unload(backend, model, processor=None)`](api/common.md#commonunload) | `self.unload()` | Release accelerator resources for `model` and `processor` and call `clear_cache(backend)`. |

---

## 3. Practical Usage Examples

### Example 1: Using `common` Dispatchers Directly

```python
from klygo.models.backend import common

# 1. Normalize decorated backend strings
family = common.backend_family("Hugging Face (Offline)")
print("Backend family:", family)  # "Hugging Face"

# 2. Query device, dtype, and inference context safely without importing frameworks upfront
dev = common.current_device(None, backend="Torchvision")
dtype = common.current_dtype(None, backend="Torchvision")
print("Default Torchvision device/dtype:", dev, dtype)

# 3. Run inside the backend's no-gradient inference context
with common.inference_context("Torchvision"):
    outputs = common.run_inference(
        "Torchvision",
        model=lambda x: [{"boxes": x, "scores": [0.9], "labels": [1]}],
        inputs=[[10.0, 20.0, 100.0, 120.0]],
        cur_dtype=dtype,
    )
    print("Inference outputs:", outputs)
```

### Example 2: Using `common` Inside a Custom `Detector.forward()`

```python
import numpy as np
from klygo import models, outputs


class CustomKerasDetector(models.Detector):
    def __init__(self, metadata=None, **kwargs):
        super().__init__(metadata=metadata, backend="Keras", **kwargs)
        # Attach a Keras 3 / KerasHub ObjectDetector (e.g., dfine_small_coco) to self.model
        self.names = {0: "person", 1: "bicycle", 2: "car"}

    def forward(self, images, prompt=None):
        batch = np.stack([np.asarray(img.convert("RGB"), dtype=np.float32) for img in images], axis=0)
        # Delegate to common.cast_inputs -> common.run_inference -> common.format_results
        inputs = self.cast_inputs(batch)
        raw_outputs = self.run_inference(inputs)
        formatted = self.format_results(raw_outputs)
        return outputs.build_detections(images, formatted, names=self.names)
```

---

## 4. Framework Adapter Modules

| Adapter Module | Import Path | Optional Extra | Responsibility |
| :--- | :--- | :--- | :--- |
| [`common.py`](api/common.md) | `klygo.models.backend.common` | None (lazy) | Unified dispatcher routing all 15 operations to the active adapter. |
| `huggingface.py` | `klygo.models.backend.huggingface` (`backend.hf`) | `klygo[transformers]` | `BatchFeature` float casting, AMP `autocast`, `ModelOutput` device inspection, `save_pretrained`. |
| `ultralytics.py` | `klygo.models.backend.ultralytics` (`backend.ul`) | `klygo[ultralytics]` | `Results.boxes` extraction (`xyxy`, `conf`, `cls`), `.save()`. |
| `torchvision.py` | `klygo.models.backend.torchvision` | `klygo[torchvision]` | `list[Tensor]` device/dtype casting, tensor-to-Python list formatting, `state_dict` saving. |
| `keras.py` | `klygo.models.backend.keras` (`backend.kerashub`) | `klygo[keras]` | Multi-engine (`torch`/`jax`/`tensorflow`) tensor casting, `KerasHub` `ObjectDetector` resize + `decode_predictions`, `yxyx` $\rightarrow$ `xyxy` conversion, `-1` padding filter. |
| `torch_runtime.py` | `klygo.models.backend.torch_runtime` | `torch` | Shared CUDA/CPU device detection, `torch.cuda.synchronize`, `torch.cuda.empty_cache`, `torch.amp.autocast`, and `torch.inference_mode`. |
