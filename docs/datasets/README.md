# `klygo.datasets` Documentation

`klygo.datasets` provides 8 focused APIs for loading, inspecting, partitioning, flattening, merging, splitting, and relabeling YOLO-style detection datasets across directories and `.zip` archives.

Interactive Google Colab Tutorial:
[Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/datasets/Document_Datasets_Overview.ipynb)

Local Tutorial Notebooks (`tutorials/datasets/`, 1-to-1 with `klygo/datasets/` submodules):
- [`Document_Datasets_Overview.ipynb`](../../tutorials/datasets/Document_Datasets_Overview.ipynb)
- [`Document_Datasets_Load.ipynb`](../../tutorials/datasets/Document_Datasets_Load.ipynb) (`klygo/datasets/load.py`)
- [`Document_Datasets_Info.ipynb`](../../tutorials/datasets/Document_Datasets_Info.ipynb) (`klygo/datasets/info.py`)
- [`Document_Datasets_Partition.ipynb`](../../tutorials/datasets/Document_Datasets_Partition.ipynb) (`klygo/datasets/partition.py`)
- [`Document_Datasets_Repartition.ipynb`](../../tutorials/datasets/Document_Datasets_Repartition.ipynb) (`klygo/datasets/repartition.py`)
- [`Document_Datasets_Unpartition.ipynb`](../../tutorials/datasets/Document_Datasets_Unpartition.ipynb) (`klygo/datasets/unpartition.py`)
- [`Document_Datasets_Merge.ipynb`](../../tutorials/datasets/Document_Datasets_Merge.ipynb) (`klygo/datasets/merge.py`)
- [`Document_Datasets_Split.ipynb`](../../tutorials/datasets/Document_Datasets_Split.ipynb) (`klygo/datasets/split.py`)
- [`Document_Datasets_Remap_Classes.ipynb`](../../tutorials/datasets/Document_Datasets_Remap_Classes.ipynb) (`klygo/datasets/remap_classes.py`)

---

## 1. Dataset Layout & Archive Interoperability

<details open>
<summary><b>Flat vs. partitioned layouts, <code>data.yaml</code>, and archive I/O</b> <i>(Click to expand or collapse)</i></summary>

`klygo.datasets` operates seamlessly on both directories and `.zip` archives by delegating filesystem operations to `klygo.files`, YAML metadata handling to `klygo.config`, archive extraction/compression to `klygo.archive`, image decoding to `klygo.media`, and detection structures to `klygo.outputs.detect`.

1. **Partitioned vs. Flat Layouts**:
   - **Partitioned (`train`/`val`/`test`)**: Images live under `images/{train,val,test}/` with matching labels under `labels/{train,val,test}/`. Created by [`partition()`](api/partition.md) and [`repartition()`](api/repartition.md).
   - **Flat (`raw`)**: All images live directly inside `images/` and labels inside `labels/` (with relative split paths encoded in filenames when flattened via [`unpartition()`](api/unpartition.md) to prevent basename collisions). Required by [`load()`](api/load.md).
2. **`data.yaml` & Sparse Class IDs**:
   - Nested dataset roots inside archives or subdirectories are discovered automatically by locating `data.yaml` alongside an `images` directory.
   - Contiguous class definitions (`names: ["apple", "banana"]`) and sparse numeric class mappings (`names: {0: "apple", 3: "banana"}`, where `nc` equals the highest class ID plus one) are preserved across [`info()`](api/info.md), [`unpartition()`](api/unpartition.md), [`merge()`](api/merge.md), [`split()`](api/split.md), and [`remap_classes()`](api/remap_classes.md).
3. **Atomic Staging & Portable `.zip` Outputs**:
   - Every transformation stages its output in an isolated temporary workspace and publishes to the target directory or `.zip` archive only after completion.
   - Generated `.zip` archives omit machine-specific absolute `path` entries in `data.yaml` so archives remain portable across machines.

</details>

---

## 2. Task-Oriented Cheatsheet (Common AI / User Questions)

<details open>
<summary><b>Task-oriented quick reference (12 tasks)</b> <i>(Click to expand or collapse)</i></summary>

| I want to... | Recommended API | Key Parameters / Example |
| :--- | :--- | :--- |
| **Inspect classes, image/label counts, and splits** | [`info()`](api/info.md) | `stats = datasets.info("dataset.zip")` |
| **Print dataset summary as a Rich table** | [`info()`](api/info.md) | `datasets.info("dataset.zip", verbose=True)` |
| **Flatten `train`/`val`/`test` splits into one folder** | [`unpartition()`](api/unpartition.md) | `flat = datasets.unpartition("dataset.zip", "flat_data")` |
| **Load a flat dataset into `Detections` for viewing/cropping** | [`load()`](api/load.md) | `dets = datasets.load(flat, format="yolo")` |
| **Partition a raw dataset into `train`/`val`/`test`** | [`partition()`](api/partition.md) | `datasets.partition("raw", "split_ds", (0.8, 0.1, 0.1))` |
| **Partition directly into a portable `.zip` archive** | [`partition()`](api/partition.md) | `datasets.partition("raw", "split_ds.zip", (0.8, 0.2))` |
| **Rebalance split ratios of an already partitioned dataset** | [`repartition()`](api/repartition.md) | `datasets.repartition("old_ds", "new_ds.zip", (0.7, 0.2, 0.1))` |
| **Combine multiple datasets and unify class IDs** | [`merge()`](api/merge.md) | `datasets.merge(["ds1", "ds2.zip"], "merged.zip")` |
| **Export one `.zip` archive per class** | [`split()`](api/split.md) | `datasets.split("dataset.zip", "by_class", by_class=True)` |
| **Export `.zip` archives for custom class groups** | [`split()`](api/split.md) | `datasets.split("ds", "groups", class_groups=[["car", "bus"]])` |
| **Split a dataset into multiple random ratio `.zip` parts** | [`split()`](api/split.md) | `datasets.split("ds", "shards", ratios=(0.5, 0.5))` |
| **Rename class labels or reassign numeric class IDs** | [`remap_classes()`](api/remap_classes.md) | `datasets.remap_classes("ds", "out", {"auto": "car", 1: 3})` |

</details>

---

## 3. Public APIs Reference (8 Functions across 8 Submodules)

| API | Submodule | Return Type | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| [`load(source, format='yolo', verbose=True)`](api/load.md) | `load.py` | `Detections` | Load all images and YOLO annotations from a flat dataset into memory. | [`load.py`](../../examples/datasets/load.py) |
| [`info(source, format='yolo', verbose=False)`](api/info.md) | `info.py` | `dict[str, Any]` | Inspect class names, IDs, image/label counts, and split distribution. | [`info.py`](../../examples/datasets/info.py) |
| [`partition(source, target, ratios, ...)`](api/partition.md) | `partition.py` | `Path` | Create deterministic `train`, `val`, and `test` partitions. | [`partition.py`](../../examples/datasets/partition.py) |
| [`repartition(source, target, ratios, ...)`](api/repartition.md) | `repartition.py` | `Path` | Flatten existing splits and rebuild with new deterministic ratios. | [`repartition.py`](../../examples/datasets/repartition.py) |
| [`unpartition(source, target, ...)`](api/unpartition.md) | `unpartition.py` | `Path` | Flatten `train`/`val`/`test` split folders into a single dataset root. | [`unpartition.py`](../../examples/datasets/unpartition.py) |
| [`merge(sources, output_path, ...)`](api/merge.md) | `merge.py` | `Path` | Combine two or more datasets into a single `.zip` with unified class IDs. | [`merge.py`](../../examples/datasets/merge.py) |
| [`split(source, output_dir, ...)`](api/split.md) | `split.py` | `list[Path]` | Split a dataset into multiple `.zip` archives by class, group, or ratio. | [`split.py`](../../examples/datasets/split.py) |
| [`remap_classes(source, target, class_map, ...)`](api/remap_classes.md) | `remap_classes.py` | `Path` | Rename classes or remap numeric class IDs across `data.yaml` and labels. | [`remap_classes.py`](../../examples/datasets/remap_classes.py) |
