# `utils.create_progress_bar`

```python
utils.create_progress_bar(
    total: int | None,
    desc: str,
    unit: str = "file",
    verbose: bool = True,
    colour: str = "cyan",
    unit_scale: bool = False,
    unit_divisor: int = 1024,
) -> ProgressBar
```

Create a configured `ProgressBar` instance through a function factory.

## Contract

Forwards all arguments directly to `utils.ProgressBar(...)` and returns the constructed `ProgressBar` instance. While `ProgressBar` defaults to `unit_scale=True, unit_divisor=1000`, `create_progress_bar()` defaults to `unit_scale=False, unit_divisor=1024` for byte-oriented or explicitly controlled progress counters.

The returned `ProgressBar` supports both context-manager usage (`with utils.create_progress_bar(...) as bar:`) and manual lifecycle management (`bar.update(n)` followed by `bar.close()`).

## Parameters

- `total`: `int | None` — expected number of updates; `None` or `0` uses a fallback display total of `1` when `verbose=True`.
- `desc`: `str` — text displayed before the progress bar.
- `unit`: `str`, default `'file'` — unit label displayed beside the counter.
- `verbose`: `bool`, default `True` — enable creation and display of the underlying `tqdm` progress bar.
- `colour`: `str`, default `'cyan'` — color name forwarded to `tqdm`.
- `unit_scale`: `bool`, default `False` — whether `tqdm` should abbreviate large values automatically.
- `unit_divisor`: `int`, default `1024` — scaling divisor forwarded to `tqdm`.

## Returns

`ProgressBar` — a configured `ProgressBar` instance that should be used as a context manager or closed explicitly via `.close()`.

## Errors and edge cases

- Performs no extra validation beyond `ProgressBar.__init__`; when `verbose=False`, no `tqdm` object is created (`bar is None`) and `update()` / `close()` are safe no-ops.
- Constructor or `tqdm` exceptions when `verbose=True` propagate unchanged.

## AI usage guidance

Prefer `utils.ProgressBar(...)` for standard `with` blocks and `utils.create_progress_bar(...)` when a callable factory or 1024-based divisor default is desired.

## Example

See [`create_progress_bar.py`](../../../examples/utils/create_progress_bar.py) for an executable example.

## Tests

See [`test_create_progress_bar.py`](../../../test/utils/test_create_progress_bar.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.utils.create_progress_bar."""

from klygo import utils

progress = utils.create_progress_bar(
    total=2,
    desc="Saving",
    unit="file",
    verbose=False,
)
progress.update(2)
progress.close()
assert progress.bar is None

with utils.create_progress_bar(4, "Archiving", verbose=False) as ctx_bar:
    ctx_bar.update(4)
assert ctx_bar.bar is None
```
