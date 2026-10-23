# `models.errors`

```python
from klygo.models import errors
```

Public submodule namespace defining the exception hierarchy raised by Klygo model configuration, lifecycle guards, and hardware runtime checks.

## Contract

`klygo.models.errors` provides specific, catchable exceptions so applications can distinguish Klygo model lifecycle and configuration faults from generic runtime errors. All exceptions inherit from `ModelFrameworkError`:

- `errors.ModelFrameworkError(Exception)`: Root exception for all `klygo.models` errors.
- `errors.InvalidStateError(ModelFrameworkError)`: Raised when calling an operational method (`predict`, `forward`, `benchmark`, `warmup`, `save`) on a model whose `state` is `"UNLOADED"`.
- `errors.InvalidConfigError(ModelFrameworkError, ValueError)`: Raised when model metadata or configuration settings violate schema rules (also catchable as `ValueError`).
- `errors.InvalidDeviceError(ModelFrameworkError, ValueError)`: Raised when an unsupported or unavailable hardware device is requested (also catchable as `ValueError`).
- `errors.InvalidDtypeError(ModelFrameworkError, ValueError)`: Raised when an unsupported numerical precision (`dtype`) is requested (also catchable as `ValueError`).

## Returns

Accessing `models.errors` returns the `klygo.models.errors` module object.

## Errors and edge cases

`InvalidConfigError`, `InvalidDeviceError`, and `InvalidDtypeError` multiply inherit from both `ModelFrameworkError` and `ValueError` for backward-compatible exception handling.

## Example

See [`errors.py`](../../../examples/models/errors.py) for a standalone executable script.

## Tests

See [`test_class_registry_backends.py`](../../../test/models/test_class_registry_backends.py).

## Complete executable example

```python
"""Executable example for klygo.models.errors."""

from klygo import models

detector = models.Detector()
detector.unload()

try:
    detector.predict([])
except models.errors.InvalidStateError as exc:
    print("Caught expected InvalidStateError:", exc)

print(issubclass(models.errors.InvalidConfigError, models.errors.ModelFrameworkError))
```
