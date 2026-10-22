# `klygo.utils`

These shared helpers manipulate small values, format byte counts, and display
progress in terminals and notebooks. `ProgressBar` and its factory live in
`progress.py`; `exclude_keys`, `invert`, and `human_size` live in
`formatting.py`.

| Submodule | API | Purpose |
|---|---|---|
| `formatting.py` | [`exclude_keys`](api/exclude_keys.md) | Copy a mapping without selected keys |
| `formatting.py` | [`invert`](api/invert.md) | Exchange mapping keys and values |
| `formatting.py` | [`human_size`](api/human_size.md) | Format bytes in binary units |
| `progress.py` | [`ProgressBar`](api/ProgressBar.md) | Manage progress display with `with` |
| `progress.py` | [`create_progress_bar`](api/create_progress_bar.md) | Create a progress wrapper from a function |

Each API has a [runnable example](../../examples/utils) and focused
[tests](../../test/utils). See the
[overview tutorial](../../tutorials/utils/Document_Utils_Overview.ipynb).
