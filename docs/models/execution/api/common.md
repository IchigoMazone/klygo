# Detector convenience methods and accelerator runtime

The `klygo.models.backend` package has been removed. Existing code must stop
importing its modules. The public Detector convenience methods still work:

| Method | Purpose |
| --- | --- |
| `current_device()`, `current_dtype()` | Inspect the wrapped model |
| `cast_inputs(inputs)` | Optional framework-specific input conversion |
| `run_inference(inputs, **model_kwargs)` | Optional native inference delegate |
| `format_results(raw_outputs)` | Optional standardized result conversion |
| `get_output_device(outputs)`, `sync_device(tensor, target_device)` | Output-shard alignment |
| `clear_cache()` | Release unused accelerator cache blocks |

A custom model can instead implement all native preprocessing, inference and
conversion in `forward`. These convenience methods are not required.

Accelerator lifecycle helpers live in `models.runtime`:

```python
from klygo import models

print(models.runtime.current_device(None, backend="Custom"))
print(models.runtime.current_dtype(None, backend="Custom"))
models.runtime.synchronize("Custom")
models.runtime.clear_cache("Custom")
```

Runtime helpers accept the selected backend name. Unknown runtimes use CPU
metadata and no-op lifecycle operations. Importing the runtime module does not
load optional frameworks. Selected Torch or Keras engines are imported lazily.
Keras uses its configured Torch, JAX or TensorFlow engine.

See [native execution](../README.md) and [Detector](../../api/Detector.md).
