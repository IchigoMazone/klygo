# `datasets.split`

```python
datasets.split(
    source, output_dir, format="yolo", by_class=False,
    class_groups=None, ratios=None, overwrite=False, verbose=True,
    shuffle=True, seed=42, stratify=False,
) -> None
```

Split a YOLO dataset into independent dataset directories or ZIP archives.
Exactly one strategy must be selected: `by_class`, `class_groups`, or `ratios`.

## Output destinations

`output_dir` accepts a directory or a naming pattern. The parameter name is
retained for existing keyword callers. A plain directory produces directories
with default names. One `*` in the final path component is replaced by the
zero-based split index; it is an output placeholder, not an input glob.

| Destination | Created datasets for ratio splitting |
| --- | --- |
| `parts` | `parts/split_part_0/`, `parts/split_part_1/` |
| `parts/*` | `parts/0/`, `parts/1/` |
| `parts/batch_*` | `parts/batch_0/`, `parts/batch_1/` |
| `parts/*_data` | `parts/0_data/`, `parts/1_data/` |
| `parts/*.zip` | `parts/0.zip`, `parts/1.zip` |
| `parts/split_part_*.zip` | `parts/split_part_0.zip`, `parts/split_part_1.zip` |

A generated name ending in `.zip` (case-insensitive) produces a ZIP; other names
produce directories. `**`, multiple placeholders, and placeholders in parent
directories are rejected. A name without `*` is always the receiving directory,
not a custom per-part name.

## Strategies and parameters

- `source`: YOLO dataset directory or supported archive.
- `output_dir`: destination directory or output naming pattern described above.
- `format`: currently only `"yolo"`.
- `by_class=True`: retain only matching images and annotations for each class;
  class IDs are remapped to zero. Default names are `split_<class>`.
- `class_groups=[...]`: filter to each class group and remap its class IDs.
  Single-class groups use `split_<class>`; multi-class groups use `split_group_<i>`.
  Sanitized default-name collisions are disambiguated. Explicit patterns use
  indices for both class strategies. Groups without matching images are omitted;
  their indices are not reassigned.
- `ratios=(...)`: two or more finite ratios between zero and one summing to one.
  Original class names and IDs are preserved. Default names are `split_part_<i>`.
- `shuffle=True`, `seed=42`, `stratify=False`: control reproducible ratio splitting; `stratify=True` preserves per-class label proportions across splits.
- `overwrite=False`: reject existing destinations; `True` replaces each output.
- `verbose=True`: display archive progress.

Each output contains `images`, `labels` when annotations exist, and `data.yaml`.
The function writes outputs and returns `None`; do not assign its result as a path.
The source dataset is unchanged. To create train/val/test partitions within one
dataset, use [`partition`](partition.md).

## Errors

`TypeError` for unsupported argument types. `ValueError` for invalid strategies,
ratios, class groups, format, or output placeholders. `FileNotFoundError` for
missing sources; `FileExistsError` for existing destinations without overwrite.

## Complete executable example

```python
from tempfile import TemporaryDirectory
from PIL import Image
from klygo import datasets, files, media

with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "source"
    for index in range(4):
        media.save(source / "images" / f"img_{index}.jpg", Image.new("RGB", (32, 24)), verbose=False)
        files.save(source / "labels" / f"img_{index}.txt", [f"{index % 2} 0.5 0.5 0.5 0.5"], verbose=False)
    files.save(source / "data.yaml", {"names": ["helmet", "head"]}, verbose=False)

    datasets.split(source, root / "parts", ratios=(0.5, 0.5), verbose=False)
    assert datasets.info(root / "parts" / "split_part_0")["total_images"] == 2
    datasets.split(source, root / "archives" / "batch_*.zip", ratios=(0.5, 0.5), verbose=False)
    assert datasets.info(root / "archives" / "batch_1.zip")["total_images"] == 2
```

[Example](../../../examples/datasets/split.py) ·
[Tests](../../../test/datasets/test_output_destinations.py) ·
[Additional tests](../../../test/datasets/test_api.py) · [Module](../README.md)
