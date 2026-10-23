# `models.utils`

```python
from klygo.models import utils
```

Public submodule namespace providing metadata normalization, parameter-group resolution, media source/stream resolution, prompt normalization, and warning-suppression utilities.

## Contract

`klygo.models.utils` is exposed as `models.utils` on the package facade. Its public `__all__` exports [`suppress_warnings`](suppress_warnings.md) and [`suppress_ai_warnings`](suppress_ai_warnings.md), and it also provides the schema validation and input resolution helpers used across `BaseModel`, `Detector`, `load`, and `metadata`:

- `utils.suppress_warnings(func=None)`: Context manager or decorator silencing Python and AI framework warnings.
- `utils.suppress_ai_warnings() -> None`: Configure quiet logging levels (`logging.ERROR`) and environment variables for Hugging Face, Torch, and Accelerate without importing them.
- `utils.normalize_flags(value) -> dict[str, bool]`: Validate parameter group names and boolean runtime permissions.
- `utils.normalize_priority(groups, value) -> dict[str, list[str]]`: Validate unprefixed priority parameter mappings.
- `utils.normalize_metadata(value) -> dict[str, Any]`: Convert legacy or raw metadata mappings into canonical top-level form.
- `utils.resolve_metadata(metadata, kwargs=None, *, runtime=False) -> box.Box`: Merge parameter overrides into a copy of `metadata`, enforcing runtime lock permissions when `runtime=True`.
- `utils.resolve_images(source, step=1, max_frames=None, stream=False)`: Resolve image paths, directories, videos, `LazyImage`, PIL images, or tensors via `klygo.media` into a list of images or a lazy streaming generator.
- `utils.normalize_prompt(text_prompt) -> list[str]`: Normalize a string or sequence of class prompts into a clean `list[str]`.

## Returns

Accessing `models.utils` returns the `klygo.models.utils` module object.

## Errors and edge cases

Schema helpers raise `TypeError` or `ValueError` when given malformed flags, reserved group names, duplicate priority aliases, or locked runtime parameter overrides.

## Example

See [`utils.py`](../../../examples/models/utils.py) for a standalone executable script.

## Tests

See [`test_metadata.py`](../../../test/models/test_metadata.py).

## Complete executable example

```python
"""Executable example for klygo.models.utils."""

from klygo import models

flags = models.utils.normalize_flags(("model", "post"))
print("Normalized flags:", flags)

prompts = models.utils.normalize_prompt([" person ", "car", "traffic light "])
print("Normalized prompts:", prompts)
```
