# `datasets.load`

```python
datasets.load(
    source: str | os.PathLike[str],
    format: str = "yolo",
    verbose: bool = True,
    *,
    images_dir: str = "images",
    labels_dir: str | collections.abc.Sequence[str] | collections.abc.Mapping[str, str] = "labels",
    yaml_name: str | collections.abc.Sequence[str] | collections.abc.Mapping[str, str] = "data.yaml",
) -> klygo.outputs.Detections | klygo.outputs.DetectionGroup
```

Load all images and five-column YOLO detection annotations from a flat dataset directory or archive into a `Detections` collection (or a `DetectionGroup` when multiple label sets are passed).

## Contract

Reads a flat YOLO dataset (`<images_dir>/<file>` and optional `<labels_dir>/<file>.txt` alongside `<yaml_name>`), decodes each image into an in-memory RGB PIL image via `klygo.media`, and converts normalized `[class_id, cx, cy, width, height]` rows into pixel XYXY `Box` instances.

When `labels_dir` is a sequence or mapping (such as `["labels", "pseudo_labels"]` or `{"gt": "labels", "pseudo": "pseudo_labels"}`), each image is decoded only once and `load()` returns a `DetectionGroup` with named members. Two sequence entries use `pred`, `truth` in input order;
other counts use `set_1`, `set_2`, and so on. Mapping inputs preserve explicit
names. Use a mapping to assign semantic roles independently of directory order.
The group validates matching frame identities and image sizes and supports
`.cuda()`, `.cpu()`, `.astype()`, and shared label/storage operations directly.
String or integer indexing selects an individual `Detections`; iteration yields
collections. `keys()`, `values()`, and `items()` expose ordered member views.
See [outputs.group](../../outputs/api/group.md).

For a partitioned dataset (`train`/`val`/`test` folders), call [`datasets.unpartition()`](unpartition.md) first and pass the resulting flat directory or `.zip` archive to `datasets.load()`. The input dataset is never modified.

## Parameters

- `source`: `str | os.PathLike[str]` — flat YOLO dataset directory or supported archive containing `data.yaml`, `images/`, and optional `labels/`.
- `format`: `str`, default `"yolo"` — annotation format to decode. Currently only `"yolo"` is supported.
- `verbose`: `bool`, default `True` — display a progress bar while loading images and annotations.
- `images_dir`: `str`, default `"images"` — name of the image subdirectory under `source`.
- `labels_dir`: `str | Sequence[str] | Mapping[str, str]`, default `"labels"` — one label subdirectory name, or multiple label subdirectories to load together into a `DetectionGroup`.
- `yaml_name`: `str | Sequence[str] | Mapping[str, str]`, default `"data.yaml"` — name of the YAML metadata file under `source`, or per-set YAML names matching `labels_dir`.

## Returns

`klygo.outputs.detect.Detections` (when `labels_dir` is a `str`) or `klygo.outputs.DetectionGroup` (when `labels_dir` is a sequence or mapping), with `source_type="images"`. Each `Detection` item contains:
- An in-memory RGB PIL `source_image` (safe to `.show()` or `.crop()` even after temporary workspaces are removed).
- `Box` objects in pixel `[x1, y1, x2, y2]` coordinates with `score=1.0`, `box.metadata["class_id"]` preserving the original numeric ID (including sparse IDs), and `box.metadata["ground_truth"] = True`.
- Detection metadata fields: `dataset_source`, `relative_path` (decoded back to the original split-relative path when flattened by `unpartition()`), `split` (`"train"`, `"val"`, `"test"`, or `"raw"`), and `ground_truth=True`.

## Errors and edge cases

- Raises `TypeError` if `format` is not a `str`.
- Raises `FileNotFoundError` if `source` or its `images` directory is missing.
- Raises `ValueError` if `format` is unsupported, if the dataset is partitioned (must call `datasets.unpartition()` first), if no supported images exist, or if any label line violates the five-column normalized YOLO detection contract (`class_id cx cy width height` with coordinates in `[0, 1]` and positive width/height).
- Raises `OSError` if an image file is corrupt or unreadable.
- Missing or empty `.txt` label files produce `Detection` objects with zero boxes.

## AI usage guidance

Always flatten partitioned datasets with `datasets.unpartition()` before calling `datasets.load()`. Because `load()` materializes RGB pixels in memory for every image, use [`datasets.info()`](info.md) instead when you only need counts or class metadata.

## Example

See [`load.py`](../../../examples/datasets/load.py) for an executable example.

## Tests

See [`test_load.py`](../../../test/datasets/test_load.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.datasets.load."""

from tempfile import TemporaryDirectory
from PIL import Image
from klygo import config, datasets, files, media

with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "source"
    media.save(source / "images" / "train" / "sample_0.jpg", Image.new("RGB", (32, 24), "white"), verbose=False)
    files.save(source / "labels" / "train" / "sample_0.txt", ["0 0.5 0.5 0.5 0.5"], overwrite=True, verbose=False)
    config.save(
        source / "data.yaml",
        {"train": "images/train", "nc": 1, "names": ["apple"]},
        overwrite=True,
        verbose=False,
    )

    flat = datasets.unpartition(source, root / "flat", verbose=False)
    results = datasets.load(flat, format="yolo", verbose=False)
    assert len(results) == 1
    assert results.total_objects == 1
    assert results[0].metadata["split"] == "train"
```

## Multiple label sets

```python
from klygo import datasets, metrics

# Use a dataset containing both label directories.
sets = datasets.load(
    "dataset", labels_dir={"pred": "pseudo_labels", "truth": "labels"},
    verbose=False,
)
sets.astype(32)
report = metrics.evaluate(sets["pred"], sets["truth"])
# With CuPy and CUDA: sets.cuda(0), evaluate again, then sets.cpu().

# Automatic names follow the input order.
# datasets.load("dataset", labels_dir=["pseudo_labels", "labels"]) -> pred, truth
# datasets.load("dataset", labels_dir=["a", "b", "c"]) -> set_1, set_2, set_3
```

Rename automatically generated members after loading with
`sets.rename({"set_1": "pred", "set_2": "truth"})`. This returns the same
group, preserves member order/data/devices, and supports chaining with `.cuda()`
or `.astype()`. See [DetectionGroup](../../outputs/api/DetectionGroup.md).
