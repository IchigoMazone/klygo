# `klygo.models.backend.huggingface` — Hugging Face Backend Guide

```python
from klygo.models.backend import huggingface
# or via lazy alias:
from klygo.models import backend
adapter = backend.huggingface  # also backend.hf
```

Runtime adapter and `Detector` integration guide for **Hugging Face Transformers** models, such as open-vocabulary **Grounding DINO** (`GroundingDinoDetect`), **DETR**, and **RT-DETR**.

---

## 1. Installation & Requirements

```bash
pip install "klygo[transformers]"
```

Select this backend via `backend="Hugging Face"` on [`models.Detector`](../../api/Detector.md) or in [`models.details(..., backend="Hugging Face")`](../../api/details.md).

---

## 2. Adapter Functions (`klygo/models/backend/huggingface.py`)

All functions are automatically invoked when `Detector` dispatches through [`klygo.models.backend.common`](common.md) with `backend="Hugging Face"`:

- `huggingface.current_device(model) -> torch.device`: Return the PyTorch device owning the model's parameters.
- `huggingface.current_dtype(model) -> torch.dtype`: Return the parameter dtype (`torch.float32`, `torch.float16`, `torch.bfloat16`).
- `huggingface.inference_context() -> ContextManager`: Return `torch.inference_mode()`.
- `huggingface.cast_inputs(inputs, dev, dtype) -> Any`: Move all tensors in a Hugging Face `BatchFeature` or `dict` to `dev`. Floating-point tensors (`pixel_values`, `pixel_mask`) are cast to `dtype`, while integer token tensors (`input_ids`, `attention_mask`, `token_type_ids`) preserve their integer dtype.
- `huggingface.run_inference(model, inputs, cur_dtype, **model_kwargs) -> Any`: Execute `model(**inputs, **model_kwargs)` inside a PyTorch AMP `autocast` context matching `cur_dtype`.
- `huggingface.get_output_device(outputs, default_device=None) -> torch.device`: Detect the actual device of `outputs.logits` or `outputs.pred_boxes` (essential when `device_map="auto"` shards a model across multiple GPUs).
- `huggingface.sync_device(tensor, target_device) -> Any`: Move `tensor` (such as `input_ids`) to `target_device` without copying large `pixel_values` tensors across PCIe.
- `huggingface.save(model, processor, output_dir) -> None`: Persist both `model.save_pretrained(output_dir)` and `processor.save_pretrained(output_dir)`.
- `huggingface.reset(model, processor=None)` / `huggingface.unload(model, processor=None)`: Move `model` back to CPU (`model.cpu()`).

---

## 3. Writing a Custom Hugging Face `Detector` Subclass

```python
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from klygo import models, outputs


class CustomGroundingDinoDetector(models.Detector):
    def __init__(self, metadata=None, **kwargs):
        super().__init__(metadata=metadata, backend="Hugging Face", **kwargs)
        model_cfg = dict(self.metadata.get("model", {}))
        proc_cfg = dict(self.metadata.get("processor", {}))
        self.processor = AutoProcessor.from_pretrained(self.model_id, **proc_cfg)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id, **model_cfg
        ).eval()

    def forward(self, images, prompt=None):
        post_cfg = dict(self.metadata.get("post", {}))
        labels = models.utils.normalize_prompt(prompt or ["object"])

        # 1. Preprocess images and text prompts into BatchFeature
        inputs = self.processor(
            images=images,
            text=[labels] * len(images),
            return_tensors="pt",
        )

        # 2. Cast floating tensors to model device/dtype while keeping input_ids int64
        inputs = self.cast_inputs(inputs)

        # 3. Run forward pass with automatic AMP autocast
        raw_outputs = self.run_inference(inputs)

        # 4. Sync input_ids to output device (multi-GPU safe) and post-process
        out_dev = self.get_output_device(raw_outputs)
        input_ids = self.sync_device(inputs.get("input_ids"), out_dev)
        target_sizes = [img.size[::-1] for img in images]

        results = self.processor.post_process_grounded_object_detection(
            raw_outputs,
            input_ids,
            threshold=post_cfg.get("threshold", 0.25),
            text_threshold=post_cfg.get("text_threshold", 0.3),
            target_sizes=target_sizes,
        )
        return outputs.build_detections(
            images,
            results,
            metadata={"prompt": labels, **post_cfg},
        )
```

---

## 4. Configuring and Loading

```python
from klygo import models

flags = models.flags(model=False, processor=True, post=True)
schema = models.metadata(
    flags,
    priority=models.priority(flags, post=("threshold", "text_threshold")),
    details=models.details(
        name="Grounding DINO Base",
        task="Object-Detection",
        backend="Hugging Face",
        library="transformers",
    ),
)
configured = models.configure(
    "IDEA-Research/grounding-dino-base",
    metadata=schema,
    implementation="klygo.models.detection.grounding_dino.GroundingDinoDetect",
    threshold=0.3,
    text_threshold=0.25,
)
print(configured.model_id, configured.details.backend, configured.post.threshold)
```
