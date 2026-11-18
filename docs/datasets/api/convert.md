# `datasets.convert`

```python
convert(source=None, target="", *, src_format="voc", dst_format="yolo", images=None, annotations=None, classes=None, image_ids=None, include_difficult=True, include_crowd=False, voc_1_based=False, overwrite=False, verbose=False)
```

Convert Pascal VOC XML or COCO instances JSON object-detection annotations and source images to a flat YOLO detection dataset containing `images/`, `labels/`, and `data.yaml`.

> [!NOTE]
> For dataset layout flattening with directory trees, use [`datasets.flatten()`](flatten.md). Passing `tree` to `datasets.convert()` raises a `TypeError` explaining the migration.

### Parameters

- `source`: Root directory containing the source dataset.
- `target`: Output directory or `.zip` archive for the flat YOLO dataset.
- `src_format`: Source annotation format (`'voc'` or `'coco'`). Default is `'voc'`.
- `dst_format`: Destination format (`'yolo'`). Default is `'yolo'`.
- `images`: Optional explicit path to directory containing source images.
- `annotations`: Optional explicit path to directory of VOC XML files or path to COCO JSON file.
- `classes`: Optional list or dictionary mapping integer IDs to class names. If omitted, discovered automatically from source annotations.
- `image_ids`: Optional sequence of image stems or identifiers to convert.
- `include_difficult`: Whether to retain objects marked difficult in Pascal VOC (default `True`).
- `include_crowd`: Whether to retain annotations marked `iscrowd=1` in COCO (default `False`).
- `voc_1_based`: Whether source VOC coordinates are 1-based (default `False`).
- `overwrite`: Whether to replace an existing target (default `False`).
- `verbose`: Whether to display progress (default `False`).

### Behavior

1. **Image Byte Preservation**: Original image files are copied directly without re-encoding or compression artifacts.
2. **Negative Samples**: Images without annotations are preserved with empty `.txt` label files, retaining the negative background population.
3. **Bounding Box Normalization**: VOC pixel coordinates `[xmin, ymin, xmax, ymax]` and COCO pixel coordinates `[x, y, w, h]` are converted to normalized YOLO format `[class_id, x_center, y_center, width, height]` with values in `[0.0, 1.0]`.
4. **Sparse Category Remapping**: COCO sparse category IDs are mapped to contiguous `0..N-1` class indices.

### Example

```python
from klygo import datasets

# Convert Pascal VOC annotations
datasets.convert(
    source="voc2007_raw",
    target="voc2007_yolo",
    src_format="voc",
    classes=["dog", "cat", "car"],
)

# Convert COCO annotations
datasets.convert(
    images="coco_raw/val2017",
    annotations="coco_raw/annotations/instances_val2017.json",
    target="val2017_yolo",
    src_format="coco",
)
```

See [`examples/datasets/convert.py`](../../../examples/datasets/convert.py) and [`test/datasets/test_convert.py`](../../../test/datasets/test_convert.py).

### Validation and return value

Returns `None`. Source and destination paths must not overlap, including resolved symlinks. Missing source images, invalid dimensions, non-finite or invalid boxes, and unmapped classes raise errors before publishing output. Boxes crossing image boundaries are clipped; fully outside boxes are rejected. Crowd/difficult filtering remains controlled by the explicit options above.
