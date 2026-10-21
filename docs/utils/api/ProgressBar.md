# `utils.ProgressBar`

```python
ProgressBar(total, desc, unit="file", verbose=True, colour="cyan",
            unit_scale=True, unit_divisor=1000)
```

`progress.py` provides a context-managed wrapper around `tqdm`. `update(n=1)`
advances the bar; `close()` closes it. When `verbose=False`, no renderer is
created, and both methods are safe to call. `total=None` or `0` displays a
fallback total of one.

```python
from klygo import utils

with utils.ProgressBar(2, "Loading", verbose=False) as progress:
    progress.update(2)
```

See the [runnable example](../../../examples/utils/ProgressBar.py),
[tests](../../../test/utils/test_ProgressBar.py), and
[tutorial](../../../tutorials/utils/Document_Utils_Progress.ipynb).
