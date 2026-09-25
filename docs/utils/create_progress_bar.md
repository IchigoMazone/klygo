# `utils.create_progress_bar`

```python
utils.create_progress_bar(total, desc, unit='file', verbose=True, colour='cyan', unit_scale=False, unit_divisor=1024)
```

Create a configured `ProgressBar` through a function-based interface.

## Contract

The factory forwards every argument directly to `ProgressBar` and returns that instance unchanged. Its defaults disable automatic unit scaling and use a 1024 divisor, which is useful for byte-oriented operations.

## Parameters

- `total`: `int | None` — expected number of updates.
- `desc`: `str` — text displayed before the progress bar.
- `unit`: `str` — unit label displayed beside the counter.
- `verbose`: `bool` — enable creation and display of the renderer.
- `colour`: `str` — color name forwarded to `tqdm`.
- `unit_scale`: `bool` — allow abbreviation of large values.
- `unit_divisor`: `int` — divisor used when scaling units.

## Returns

A configured `ProgressBar` instance. The caller should close it or use it as a context manager.

## Errors and edge cases

The factory performs no additional validation. Constructor and renderer exceptions propagate unchanged.

## AI usage guidance

Prefer direct `ProgressBar` construction for context-manager code. Use `utils.create_progress_bar(...)` when a function factory is easier to inject, pass around, or mock in a workflow.

## Example

See [`create_progress_bar.py`](../../examples/utils/create_progress_bar.py) for an executable example.

## Tests

See [`test_create_progress_bar.py`](../../test/utils/test_create_progress_bar.py) for the executable behavioral contract.

## Complete executable example

```python
'''Executable example for klygo.utils.create_progress_bar.'''

from klygo import utils


progress = utils.create_progress_bar(
    total=2,
    desc='Saving',
    unit='file',
    verbose=False,
)
progress.update(2)
progress.close()
assert progress.bar is None
```
