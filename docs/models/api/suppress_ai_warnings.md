# `models.suppress_ai_warnings`

```python
models.suppress_ai_warnings() -> None
```

Configure quiet environment variables and `ERROR`-level logging for optional AI libraries without importing them.

## Contract

Importing heavy frameworks such as `torch` or `transformers` solely to silence their startup logs would violate Klygo's lazy optional-dependency design. `models.suppress_ai_warnings()` instead configures standard Python `logging` loggers by name and sets environment variables before those packages are imported:

- Sets `HF_HUB_DISABLE_SYMLINKS_WARNING="1"`
- Sets `HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING="1"`
- Sets `TOKENIZERS_PARALLELISM="false"`
- Sets `logging.getLogger(name).setLevel(logging.ERROR)` for `"accelerate"`, `"huggingface_hub"`, `"huggingface_hub.utils._http"`, `"transformers"`, `"urllib3"`, and `"torch"`.

## Parameters

None.

## Returns

`None`; updates process environment variables and standard `logging` logger levels in-place.

## Errors and edge cases

Safe to call repeatedly at any point in the process lifecycle; never imports third-party AI packages.

## Example

See [`suppress_ai_warnings.py`](../../../examples/models/suppress_ai_warnings.py) for a standalone executable script.

## Tests

See [`test_optional_dependencies.py`](../../../test/models/test_optional_dependencies.py).

## Complete executable example

```python
"""Executable example for klygo.models.suppress_ai_warnings."""

import logging
import os
from klygo import models

models.suppress_ai_warnings()
print("TOKENIZERS_PARALLELISM:", os.environ.get("TOKENIZERS_PARALLELISM"))
print("transformers log level:", logging.getLogger("transformers").level == logging.ERROR)
```
