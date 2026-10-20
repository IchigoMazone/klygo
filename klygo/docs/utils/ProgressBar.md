# `utils.ProgressBar`

```python
utils.ProgressBar(total, desc, unit='file', verbose=True, colour='cyan', unit_scale=True, unit_divisor=1000)
```

Create a context-managed progress indicator shared by Klygo modules.

## Contract

When `verbose=True`, the class creates one `tqdm` progress object and forwards all display configuration. When disabled, no renderer is created and `update()` and `close()` are safe no-ops. Leaving a `with` block always closes the renderer.

## Parameters

- `total`: `int | None` — expected number of updates; `None` or `0` uses a display fallback of one.
- `desc`: `str` — text displayed before the progress bar.
- `unit`: `str` — unit label such as `file`, `frame`, or `byte`.
- `verbose`: `bool` — enable creation and display of the underlying progress bar.
- `colour`: `str` — color name forwarded to `tqdm`.
- `unit_scale`: `bool` — allow abbreviation of large values.
- `unit_divisor`: `int` — scaling divisor, normally 1000 for counts or 1024 for bytes.

## Returns

A `ProgressBar` instance. Its `bar` attribute contains the active renderer while enabled and becomes `None` after closing.

## Errors and edge cases

With `verbose=False`, rendering dependencies are not called. Exceptions raised by `tqdm` for unsupported configuration values propagate unchanged. Calling `close()` more than once is safe.

## AI usage guidance

Prefer `from klygo import utils` followed by `utils.ProgressBar(...)`. Use it as a context manager whenever possible. Internal Klygo modules should import this shared implementation instead of defining domain-specific aliases.

## Example

See [`ProgressBar.py`](../../examples/utils/ProgressBar.py) for an executable example.

## Tests

See [`test_ProgressBar.py`](../../test/utils/test_ProgressBar.py) for the executable behavioral contract.

## Complete executable example

```python
'''Executable example for klygo.utils.ProgressBar.'''

from klygo import utils


with utils.ProgressBar(
    total=3,
    desc='Processing',
    unit='item',
    verbose=False,
) as progress:
    for _ in range(3):
        progress.update()

assert progress.bar is None
```
