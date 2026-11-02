# `utils.ProgressBar`

```python
utils.ProgressBar(
    total: int | None,
    desc: str,
    unit: str = "file",
    verbose: bool = True,
    colour: str = "cyan",
    unit_scale: bool = True,
    unit_divisor: int = 1000,
)
```

Create a context-managed progress indicator shared across Klygo modules.

## Contract

Provides a unified progress-reporting interface for both terminal and Jupyter/Colab notebook environments via `tqdm.auto.tqdm`.

- When `verbose=True`, initializes `self.bar` as a `tqdm` instance configured with `total=total or 1` (so `None` or `0` safely falls back to a display total of `1`), `ascii=" █"`, and `leave=True`.
- When `verbose=False`, `self.bar` remains `None`, no `tqdm` renderer is instantiated, and calls to `update(n=1)` and `close()` execute as zero-overhead no-ops.
- Implements the context manager protocol (`__enter__` returns `self`; `__exit__` calls `self.close()`), setting `self.bar = None` upon closing so repeated `close()` calls are idempotent.

## Parameters

- `total`: `int | None` — expected number of updates; `None` or `0` uses a fallback display total of `1`.
- `desc`: `str` — prefix description displayed to the left of the progress bar.
- `unit`: `str`, default `'file'` — unit label such as `'file'`, `'frame'`, `'item'`, or `'B'`.
- `verbose`: `bool`, default `True` — when `True`, creates and renders the underlying `tqdm` bar; when `False`, disables all output.
- `colour`: `str`, default `'cyan'` — bar color forwarded to `tqdm`.
- `unit_scale`: `bool`, default `True` — automatically scale and abbreviate large counts.
- `unit_divisor`: `int`, default `1000` — divisor used when scaling units (`1000` for item counts, `1024` for bytes).

## Returns

`ProgressBar` — a context-managed progress wrapper exposing `.verbose`, `.bar`, `.update(n=1)`, and `.close()`.

## Errors and edge cases

- With `verbose=False`, `tqdm` is never called and `self.bar` is `None`.
- Calling `close()` multiple times or calling `update()` after `close()` is a safe no-op.
- Any exceptions raised by `tqdm` when `verbose=True` (such as invalid parameter types) propagate unchanged.

## AI usage guidance

Prefer `from klygo import utils` followed by `with utils.ProgressBar(...) as progress:`. Always thread a caller's `verbose` flag into `ProgressBar(..., verbose=verbose)` so silent workflows produce no console output.

## Example

See [`ProgressBar.py`](../../../examples/utils/ProgressBar.py) for an executable example.

## Tests

See [`test_ProgressBar.py`](../../../test/utils/test_ProgressBar.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.utils.ProgressBar."""

from klygo import utils

with utils.ProgressBar(
    total=3,
    desc="Processing",
    unit="item",
    verbose=False,
) as progress:
    for _ in range(3):
        progress.update()

assert progress.bar is None
progress.close()
```
