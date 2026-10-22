# `klygo.models.backend.common` API Reference

```python
from klygo.models.backend import common
```

Unified multi-backend dispatcher providing 15 functions for hardware introspection, accelerator synchronization, tensor casting, forward inference, prediction normalization, and model persistence across **Hugging Face**, **Ultralytics**, **Torchvision**, and **Keras / KerasHub**.

---

## 1. Backend & Hardware Introspection

### `common.backend_family`

```python
common.backend_family(backend: Optional[str]) -> str
```

Normalize a decorated backend string by stripping parenthesized suffixes.

- **Parameters**: `backend` (`str | None`) — backend label such as `"Hugging Face (Offline)"`, `"Ultralytics (Offline)"`, `"PyTorch (In-Memory)"`, or `"Keras"`.
- **Returns**: `str` — canonical family name (`"Hugging Face"`, `"Ultralytics"`, `"Torchvision"`, `"Keras"`, `"PyTorch"`, or `""` when `None`).

---

### `common.current_device`

```python
common.current_device(model: Any, backend: Optional[str] = None) -> Any
```

Detect the hardware device currently hosting `model`.

- **Parameters**:
  - `model` (`Any`) — wrapped framework model (`torch.nn.Module`, `YOLO`, `keras.Model`, etc.).
  - `backend` (`str | None`, default `None`) — backend name guiding adapter selection.
- **Returns**: `torch.device` for PyTorch-backed adapters (`Hugging Face`, `Ultralytics`, `Torchvision`), a device string (`"cuda:0"`, `"gpu:0"`, `"cpu"`) for `Keras`, or `"cpu"` for unknown/uninitialized models.

---

### `common.current_dtype`

```python
common.current_dtype(model: Any, backend: Optional[str] = None) -> Any
```

Detect the numerical precision (`dtype`) of `model`'s parameters or compute policy.

- **Parameters**:
  - `model` (`Any`) — wrapped framework model.
  - `backend` (`str | None`, default `None`) — backend name guiding adapter selection.
- **Returns**: `torch.dtype` (e.g., `torch.float32`, `torch.float16`, `torch.bfloat16`) for PyTorch-backed adapters, or a dtype string (e.g., `"float32"`, `"float16"`) for `Keras` and fallback adapters.

---

### `common.is_accelerator_available`

```python
common.is_accelerator_available(backend: str, device: Any = None) -> bool
```

Check whether hardware acceleration is available for `backend` and compatible with `device`.

- **Parameters**:
  - `backend` (`str`) — backend identifier.
  - `device` (`Any`, default `None`) — optional target device to verify (e.g., returns `False` if `device="cpu"` even on a CUDA machine).
- **Returns**: `bool` — `True` when a GPU/accelerator is active and usable for the specified `device`.

---

## 2. Accelerator Context & Synchronization

### `common.inference_context`

```python
common.inference_context(backend: str) -> ContextManager
```

Return the framework's no-gradient inference context manager.

- **Parameters**: `backend` (`str`) — backend identifier.
- **Returns**: `torch.inference_mode()` for `"Hugging Face"`, `"Ultralytics"`, and `"Torchvision"`; `contextlib.nullcontext()` for `"Keras"` (which uses `training=False` inside `run_inference`) and custom backends.

---

### `common.synchronize`

```python
common.synchronize(backend: str, device: Any = None, value: Any = None) -> None
```

Wait for asynchronous accelerator work to finish before recording benchmark timestamps.

- **Parameters**:
  - `backend` (`str`) — backend identifier.
  - `device` (`Any`, default `None`) — device to synchronize (`torch.cuda.synchronize(device)`).
  - `value` (`Any`, default `None`) — output tensor/structure to materialize via `keras.ops.convert_to_numpy()` on lazy Keras engines.
- **Returns**: `None`.

---

### `common.clear_cache`

```python
common.clear_cache(backend: str) -> None
```

Release unused cached GPU memory blocks held by the selected backend.

- **Parameters**: `backend` (`str`) — backend identifier.
- **Returns**: `None`. Calls `torch.cuda.empty_cache()` for PyTorch-backed adapters; intentionally avoids `keras.backend.clear_session()` on Keras so other live models in the same process remain intact.

---

## 3. Inference Pipeline Dispatchers

### `common.cast_inputs`

```python
common.cast_inputs(backend: str, inputs: Any, dev: Any, dtype: Any) -> Any
```

Cast input tensors to the model's device and floating-point precision.

- **Behavior by Backend**:
  - `"Hugging Face"`: Moves all tensors in a `BatchFeature` mapping, list/tuple, or single `torch.Tensor` to `dev`. Floating-point tensors are cast to `dtype` (`float16`/`bfloat16` on CUDA, `float32` on CPU); integer tensors (`input_ids`, `attention_mask`) keep their integer dtype.
  - `"Torchvision"`: Moves and casts each image tensor in `list[torch.Tensor]` via `[img.to(device=dev, dtype=dtype) for img in inputs]`.
  - `"Keras"` / `"KerasHub"`: Converts inputs with `keras.ops.convert_to_tensor(inputs)` and casts floating-point tensors to `dtype`.
  - Other backends: Returns `inputs` unchanged.

---

### `common.run_inference`

```python
common.run_inference(
    backend: str,
    model: Any,
    inputs: Any,
    cur_dtype: Any,
    **model_kwargs,
) -> Any
```

Execute a forward pass using the native runtime conventions of `backend`.

- **Behavior by Backend**:
  - `"Hugging Face"`: Enters `torch_runtime.autocast(...)` based on `cur_dtype` and calls `model(**inputs, **model_kwargs)` (or positional call for tensor/sequence inputs).
  - `"Torchvision"`: Calls `model(inputs, **model_kwargs)` on the list of image tensors.
  - `"Keras"` / `"KerasHub"`:
    - If `model` is a `KerasHub` `ObjectDetector` (exposing `decode_predictions` and `preprocessor`), resizes `inputs` to the converter's target `(H, W)` with `keras.ops.image.resize` so aspect-ratio cropping does not clip image borders, runs `model(x, training=False)`, and calls `model.decode_predictions(raw_preds, inputs)` to scale boxes back to the original image dimensions.
    - Otherwise calls `model(inputs, training=False, **model_kwargs)`.
  - Fallback: Calls `model(inputs, **model_kwargs)` when `model` is callable.

---

### `common.get_output_device`

```python
common.get_output_device(backend: str, outputs: Any, default_device: Any) -> Any
```

Inspect model outputs to determine which device holds the result tensors.

- **Behavior**: For `"Hugging Face"`, checks `outputs.logits.device`, `outputs.pred_boxes.device`, or `outputs.device` before falling back to `default_device`. Returns `default_device` for all other backends.

---

### `common.sync_device`

```python
common.sync_device(backend: str, tensor: Any, target_device: Any) -> Any
```

Move `tensor` to `target_device` when multi-GPU model sharding (`device_map="auto"`) places input and output tensors on different devices.

- **Behavior**: For `"Hugging Face"`, calls `tensor.to(target_device, non_blocking=...)` when `tensor.device != target_device`. Returns `tensor` unchanged for other backends.

---

### `common.format_results`

```python
common.format_results(backend: str, raw_outputs: Any) -> List[Dict[str, Any]]
```

Normalize backend-specific predictions into Klygo's standard per-frame dictionary list:

```python
[
    {
        "boxes": [[x1, y1, x2, y2], ...],   # float xyxy coordinates
        "scores": [0.95, ...],              # float confidence scores
        "labels": ["person", ...],          # str or int labels
        "class_ids": [0, ...],              # int class IDs
    },
    ...
]
```

- **Behavior by Backend**:
  - `"Ultralytics"`: Extracts `b.xyxy[0]`, `b.conf[0]`, `b.cls[0]`, and `res.names[cls_id]` from each Ultralytics `Results` object.
  - `"Torchvision"`: Extracts `item["boxes"]`, `item["scores"]`, and `item["labels"]` tensors on CPU as Python lists.
  - `"Keras"` / `"KerasHub"`: Unpacks `boxes`, `scores`/`confidence`, and `classes`/`labels`, filters out padded `-1` slots using `num_detections` and `score >= 0`, and converts `yxyx` (`[ymin, xmin, ymax, xmax]`) coordinates from KerasHub `ObjectDetector` into `xyxy` (`[xmin, ymin, xmax, ymax]`).

---

## 4. Lifecycle & Persistence Dispatchers

### `common.save`

```python
common.save(
    backend: str,
    model: Any,
    output_dir: str,
    metadata: Dict[str, Any],
    class_module: str,
    processor: Optional[Any] = None,
) -> None
```

Persist a complete, reloadable Klygo model directory at `output_dir`:
1. Writes `klygo.json` containing the model's `metadata` (excluding transient `num_params`).
2. Copies the Python source file defining the wrapper class to `output_dir/model.py` whenever `class_module` is a custom module outside `klygo.models.*`.
3. Delegates weight serialization to the backend adapter:
   - `"Hugging Face"`: `model.save_pretrained(output_dir)` and `processor.save_pretrained(output_dir)`.
   - `"Ultralytics"`: `model.save(output_dir)`.
   - `"Torchvision"`: `torch.save(model.state_dict(), "<output_dir>/model.pt")`.
   - `"Keras"`: `model.save_to_preset(output_dir)`, `model.save("<output_dir>/model.keras")`, or `model.save_weights("<output_dir>/model.weights.h5")`.

---

### `common.reset`

```python
common.reset(backend: str, model: Any, processor: Optional[Any] = None) -> None
```

Move `model` back to CPU (`model.cpu()` where supported) and call `common.clear_cache(backend)`.

---

### `common.unload`

```python
common.unload(backend: str, model: Any, processor: Optional[Any] = None) -> None
```

Release accelerator resources held by `model` and `processor`, then call `common.clear_cache(backend)`.

---

## Complete Executable Example

```python
"""Executable example for klygo.models.backend.common."""

from tempfile import TemporaryDirectory
from klygo import files
from klygo.models.backend import common

# 1. Inspect backend family normalization and default device/dtype fallbacks
print("Family:", common.backend_family("Ultralytics (Offline)"))
print("Device:", common.current_device(None, backend="Custom"))
print("Dtype:", common.current_dtype(None, backend="Custom"))

# 2. Dispatch inference and save a custom model bundle via common.save()
with TemporaryDirectory() as tmp:
    out_dir = files.path(tmp) / "bundle"
    common.save(
        backend="Custom",
        model=None,
        output_dir=str(out_dir),
        metadata={"model_id": "demo", "class": "klygo.models.detection.Detector"},
        class_module="klygo.models.detection.base",
    )
    print("Saved klygo.json:", files.exists(out_dir / "klygo.json"))
```
