# `models.BaseModel`

```python
models.BaseModel(
    metadata: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = None,
    model_id: Optional[str] = None,
    task: Optional[str] = None,
    **kwargs,
) -> None
```

Define the abstract Layer-1 lifecycle, metadata, and introspection contract for all Klygo model wrappers.

## Contract

`BaseModel` is an abstract base class (`abc.ABC`) that initializes model identity (`model_id`, `class_name`, `task`), normalizes metadata into a dot-accessible `Box`, extracts declared parameter-group `flags`, and sets the initial lifecycle state to `"READY"`.

Subclasses must implement three abstract methods:
- `predict(*args, **kwargs)`
- `benchmark(*args, **kwargs)`
- `help() -> None`

Once an instance transitions to `state == "UNLOADED"` (for example after calling `Detector.unload()`), `__getattribute__` blocks calls to operational methods (`predict`, `forward`, `benchmark`, `warmup`, `save`) and raises `models.errors.InvalidStateError`. Introspection and recovery methods (`reset`, `unload`, `info`, `help`, `supports`, `methods`) remain accessible while unloaded.

## Parameters

- `metadata`: `Mapping[str, Any] | None`, default `None` — model metadata mapping or `Box` created by `models.metadata()` / `models.configure()`.
- `backend`: `str | None`, default `None` — explicit backend override (`"Ultralytics"`, `"Hugging Face"`, `"Torchvision"`, `"Keras"`, `"PyTorch"`). When omitted, `model.backend` auto-detects from `KLYGO_BACKEND`, `self.model`, or `metadata`.
- `model_id`: `str | None`, default `None` — identifier or checkpoint path for the wrapped model. Defaults to `metadata.get("model_id", "custom-model")`.
- `task`: `str | None`, default `None` — task label (such as `"Object-Detection"`). Defaults to `metadata.get("task")` or `"Universal"`.
- `**kwargs`: `Any` — additional keyword arguments accepted by cooperative subclasses.

## Key Properties and Methods

- `model.backend -> str`: Auto-detected or explicitly assigned runtime backend name.
- `model.metadata -> box.Box`: Active metadata (`self._metadata` by default, or call-local runtime metadata inside `self._use_metadata(...)`).
- `model.details -> box.Box`: Descriptive metadata (`name`, `task`, `backend`, `library`, etc.).
- `model.flags -> tuple[str, ...]`: Declared parameter group names.
- `model.settings -> dict[str, Any]`: Current parameter group dictionary.
- `model.params -> dict[str, Any]`: Flattened view of all parameter groups.
- `model.supports(op_name: str) -> bool`: Check whether a method name is implemented and callable.
- `model.has_flag(name: str) -> bool`: Check whether a parameter group is declared in `self.flags`.
- `model.methods() -> list[str]`: Return sorted public method names.
- `model.info() -> None`: Print a formatted Rich table summarizing the model's class, ID, backend, task, state, flags, and parameters.

## Returns

Instantiating a concrete subclass returns a `BaseModel` instance with `state == "READY"`. Instantiating `BaseModel` directly raises `TypeError`.

## Errors and edge cases

- Raises `TypeError` if instantiated directly without implementing `predict`, `benchmark`, and `help`.
- Raises `models.errors.InvalidStateError` when calling operational methods after the model's `state` is set to `"UNLOADED"`.

## Example

See [`BaseModel.py`](../../../examples/models/BaseModel.py) for a standalone executable script.

## Tests

See [`test_class_registry_backends.py`](../../../test/models/test_class_registry_backends.py) and [`test_metadata.py`](../../../test/models/test_metadata.py).

## Complete executable example

```python
"""Executable example for klygo.models.BaseModel."""

from klygo import models


class EchoModel(models.BaseModel):
    def predict(self, x):
        return {"model_id": self.model_id, "backend": self.backend, "input": x}

    def benchmark(self, x, num_runs: int = 5):
        return {"runs": num_runs, "status": "ok"}

    def help(self) -> None:
        print(f"Help for {self.model_id}")


schema = models.configure(
    "echo-v1",
    flags=models.flags(model=False, post=True),
    details=models.details(
        name="EchoModel",
        task="Echo",
        backend="PyTorch",
        library="custom",
    ),
    post_threshold=0.5,
)

wrapper = EchoModel(metadata=schema)
print(wrapper.model_id, wrapper.backend, wrapper.state)
print(wrapper.has_flag("post"), wrapper.supports("predict"))
print(wrapper.predict("sample"))
```
