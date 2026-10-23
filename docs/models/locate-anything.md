# NVIDIA Locate Anything integration

Klygo's `LocateAnythingDetect` implements the official single-image generation
contract for `nvidia/LocateAnything-3B`. The registered loader name is
`locate-anything`. The pinned revision is
`c32291ca5e996f5a7a485845b4f57a233936bba0`.

Primary upstream references:

- [NVIDIA model card and worker example](https://huggingface.co/nvidia/LocateAnything-3B)
- [Pinned model implementation](https://huggingface.co/nvidia/LocateAnything-3B/blob/c32291ca5e996f5a7a485845b4f57a233936bba0/modeling_locateanything.py)
- [Official Eagle project](https://github.com/NVlabs/Eagle/tree/main/Embodied)
- [Model license](https://huggingface.co/nvidia/LocateAnything-3B/blob/c32291ca5e996f5a7a485845b4f57a233936bba0/LICENSE)

## Installation and execution boundary

Use an isolated Python environment with CUDA-enabled PyTorch and
`pip install "klygo[locate-anything]"`. This extra pins Transformers 4.57.1 and
includes inference dependencies; it does not replace the host's PyTorch build
with a suitable CUDA build automatically. This adapter deliberately rejects
CPU and other Transformers versions. Upstream documents A100/H100 testing;
VRAM use depends strongly on input resolution, generated length and attention
implementation. No universal minimum VRAM is promised. Model files are about
7.8 GB; this is storage size, not an inference memory estimate.

The NVIDIA model license restricts the weights to non-commercial research use.
Klygo's package license does not replace the checkpoint's terms.

The checkpoint uses custom Python code. Klygo requires explicit
`model_trust_remote_code=True`; importing Klygo or this wrapper does not load
weights or execute remote code. A full commit hash is required for revisions.
The default download directory is `artifacts/weights/locate-anything`.

```python
from klygo import models

detector = models.load(
    "locate-anything",
    model_trust_remote_code=True,
    model_device="cuda",
    model_torch_dtype="bfloat16",
)
results = detector("image.jpg", prompt=["person", "car"], max_new_tokens=8192)
later = detector("next-image.jpg")
report = detector.benchmark("image.jpg", iterations=5, warmup=1)
detector.unload()
```

If the GPU lacks bfloat16, choose `model_torch_dtype="float16"` explicitly.
Loading supports the official identifier or a local copy with
`model_type="locateanything"`. A saved Klygo bundle includes the tokenizer.
Use `model_local_files_only=True` to prohibit checkpoint downloads.

## Supported inference

The processor's chat-template and vision-input helpers feed `model.generate()`
with token IDs, pixel values, attention masks, image grids and the tokenizer.
Floating inputs follow model dtype; token IDs remain integers. Generation
requires `use_cache=True`. Images and labels run sequentially, adapting Klygo's
batch/frame interface to upstream's batch-size-one generation API. The adapter
does not claim native batched decoding, pointing, OCR, visual prompts or CPU
support. A non-empty text query is required.

Supported remembered generation options are `generation_mode` (`fast`, `hybrid`,
`slow`), `max_new_tokens`, `temperature`, `top_p`, `repetition_penalty` and
`do_sample`. The default is deterministic token selection (`do_sample=False`);
upstream examples also demonstrate sampling. Runtime source/stream/batch
controls are never sent to generation as inference options.

Box tokens encode XYXY coordinates in `[0, 1000]`. The adapter validates them
and converts to pixels of the image passed to the model. For Klygo preprocessing,
call `results.restore(target="original")` to map back to the original source.
Malformed, truncated or inverted boxes raise errors rather than returning
partial results. An explicit `None` response represents no detections.

**The upstream API does not return detection confidence.** Container scores are
1.0 presence values, marked by `config["confidence_available"] = False`,
`config["score_kind"] = "presence_only"` and `box.metadata["native_score"] = None`.
They must not be compared to calibrated detector confidence. Threshold and
text-threshold options are rejected; Klygo does not invent a score filter.
Raw model responses and the checkpoint revision are retained in result metadata.

Reset clears Klygo prompt/configuration state while keeping the CUDA model on
its device. Cache clearing uses the existing PyTorch backend helper. Benchmark
overrides are isolated; unload releases model, processor and tokenizer.

## Test scope

[Unit tests](../../test/models/test_text_detectors.py) exercise mocked official
processor/generation schemas, token dtypes, coordinate conversion, malformed
outputs, optional dependencies and lifecycle behavior without downloads.

[Real checkpoint smoke tests](../../test/models/test_real_detection_opt_in.py)
require `KLYGO_REAL_LOCATE` pointing to a local checkpoint plus
`KLYGO_REAL_IMAGE` pointing to a local image. Setting these opts into its custom
code. The test skips with a stated reason when CUDA or Transformers 4.57.1 is
unavailable. Mock tests alone do not establish real-checkpoint correctness or
hardware compatibility. This checkout was developed on a CPU-only host; the
Locate Anything checkpoint path still needs validation on supported hardware.
