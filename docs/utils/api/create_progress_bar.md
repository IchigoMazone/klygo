# `utils.create_progress_bar`

```python
create_progress_bar(total, desc, unit="file", verbose=True, colour="cyan",
                    unit_scale=False, unit_divisor=1024) -> ProgressBar
```

`progress.py` offers a function to create the same `ProgressBar` class.
Its defaults disable automatic unit scaling and use a 1024 divisor. Use
the result with `with` or call `close()` when finished.

```python
from klygo import utils

with utils.create_progress_bar(2, "Saving", verbose=False) as progress:
    progress.update(2)
```

See the [runnable example](../../../examples/utils/create_progress_bar.py),
[tests](../../../test/utils/test_create_progress_bar.py), and
[tutorial](../../../tutorials/utils/Document_Utils_Progress.ipynb).
