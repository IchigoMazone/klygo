# `klygo.utils` Documentation

`klygo.utils` provides pure mapping transformations, binary byte-size formatting, and unified context-managed progress bars shared across Klygo's filesystem, archive, dataset, media, and model pipelines. Import the public facade with `from klygo import utils`.

Interactive Google Colab Tutorial:
[Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/utils/Document_Utils_Overview.ipynb)

Local Tutorial Notebooks (`tutorials/utils/`, 1-to-1 with `klygo/utils/` submodules):
- [`Document_Utils_Overview.ipynb`](../../tutorials/utils/Document_Utils_Overview.ipynb)
- [`Document_Utils_Formatting.ipynb`](../../tutorials/utils/Document_Utils_Formatting.ipynb) (`klygo/utils/formatting.py`)
- [`Document_Utils_Progress.ipynb`](../../tutorials/utils/Document_Utils_Progress.ipynb) (`klygo/utils/progress.py`)

---

## 1. Core Concepts & Architecture

<details open>
<summary><b>Non-mutating mapping helpers, binary byte formatting, and silent-safe progress reporting</b> <i>(Click to expand or collapse)</i></summary>

`klygo.utils` is organized into two focused submodules:

| Submodule | Public APIs | Key Architectural Guarantees |
| :--- | :--- | :--- |
| **`formatting.py`** | [`exclude_keys()`](api/exclude_keys.md), [`invert()`](api/invert.md), [`human_size()`](api/human_size.md) | **Pure, non-mutating transformations**: `exclude_keys()` and `invert()` accept any `collections.abc.Mapping` and return a new `dict` without modifying the caller's input. `invert()` enforces one-to-one reversibility by rejecting duplicate or unhashable values. `human_size()` formats non-negative integer byte counts using base-1024 binary steps (`B`, `KB`, `MB`, `GB`, `TB`). |
| **`progress.py`** | [`ProgressBar`](api/ProgressBar.md), [`create_progress_bar()`](api/create_progress_bar.md) | **Zero-overhead `verbose=False` mode**: `ProgressBar` wraps `tqdm.auto.tqdm` for terminal and notebook compatibility. When `verbose=False`, no `tqdm` instance is allocated (`bar is None`), and `.update()` and `.close()` become no-ops. Using `with utils.ProgressBar(...) as progress:` guarantees cleanup on block exit. |

Note the intentional default difference between the class constructor and factory function:
- `ProgressBar(...)` defaults to `unit_scale=True, unit_divisor=1000` (suited for general item/frame counts).
- `create_progress_bar(...)` defaults to `unit_scale=False, unit_divisor=1024` (suited for byte-oriented or explicitly scaled counters).

</details>

---

## 2. Task-Oriented Cheatsheet (Common AI / User Questions)

<details open>
<summary><b>Task-oriented quick reference (12 tasks)</b> <i>(Click to expand or collapse)</i></summary>

| I want to... | Recommended API | Key Parameters / Example |
| :--- | :--- | :--- |
| **Copy a dictionary without one specific key** | [`exclude_keys()`](api/exclude_keys.md) | `clean = utils.exclude_keys(cfg, "device")` |
| **Copy a dictionary without multiple keys** | [`exclude_keys()`](api/exclude_keys.md) | `clean = utils.exclude_keys(cfg, ["dtype", "device"])` |
| **Swap dictionary keys and values (`{name: id}` <-> `{id: name}`)** | [`invert()`](api/invert.md) | `id_to_name = utils.invert({"cat": 0, "dog": 1})` |
| **Verify that a mapping is strictly one-to-one before inverting** | [`invert()`](api/invert.md) | `utils.invert(mapping)` (raises `ValueError` on duplicate values) |
| **Format byte counts as human-readable `B`, `KB`, `MB`, `GB`, or `TB`** | [`human_size()`](api/human_size.md) | `utils.human_size(1048576)  # '1.00 MB'` |
| **Control decimal precision when formatting byte sizes** | [`human_size()`](api/human_size.md) | `utils.human_size(1536, decimal_places=1)  # '1.5 KB'` |
| **Display a context-managed progress bar for a loop** | [`ProgressBar`](api/ProgressBar.md) | `with utils.ProgressBar(100, "Processing", unit="img") as bar: bar.update()` |
| **Silence progress output in headless or quiet workflows** | [`ProgressBar`](api/ProgressBar.md) | `with utils.ProgressBar(100, "Silent", verbose=False) as bar: bar.update()` |
| **Create a byte-oriented progress bar via a factory function** | [`create_progress_bar()`](api/create_progress_bar.md) | `bar = utils.create_progress_bar(total_bytes, "Downloading", unit="B", unit_scale=True)` |
| **Advance a progress bar by a batch or chunk size** | [`ProgressBar`](api/ProgressBar.md) | `progress.update(len(batch))` |
| **Handle unknown or zero total counts safely in `tqdm`** | [`ProgressBar`](api/ProgressBar.md) | `utils.ProgressBar(None, "Streaming")` (falls back to display total of `1`) |
| **Manually close a non-context-managed progress bar** | [`create_progress_bar()`](api/create_progress_bar.md) | `bar = utils.create_progress_bar(5, "Task"); bar.update(5); bar.close()` |

</details>

---

## 3. Public APIs Reference (4 Functions, 1 Class across 2 Submodules)

<details open>
<summary><b>Mapping and byte-size formatting — <code>formatting.py</code> (3 functions)</b> <i>(Click to expand)</i></summary>

- [`exclude_keys(values, keys)`](api/exclude_keys.md): Return a shallow dictionary copy omitting one key (`str`) or a list of keys (`list[str]`) without mutating the source mapping ([example](../../examples/utils/exclude_keys.py)).
- [`invert(values)`](api/invert.md): Return a new dictionary with keys and values exchanged, enforcing unique and hashable values ([example](../../examples/utils/invert.py)).
- [`human_size(num_bytes, decimal_places=2)`](api/human_size.md): Format a non-negative integer byte count using base-1024 binary units (`B`, `KB`, `MB`, `GB`, `TB`) ([example](../../examples/utils/human_size.py)).

</details>

<details open>
<summary><b>Progress reporting — <code>progress.py</code> (1 class, 1 function)</b> <i>(Click to expand)</i></summary>

- [`ProgressBar(total, desc, unit='file', verbose=True, colour='cyan', unit_scale=True, unit_divisor=1000)`](api/ProgressBar.md): Context-managed `tqdm` progress wrapper with zero-overhead no-op mode when `verbose=False` ([example](../../examples/utils/ProgressBar.py)).
- [`create_progress_bar(total, desc, unit='file', verbose=True, colour='cyan', unit_scale=False, unit_divisor=1024)`](api/create_progress_bar.md): Factory function returning a configured `ProgressBar` instance ([example](../../examples/utils/create_progress_bar.py)).

</details>
