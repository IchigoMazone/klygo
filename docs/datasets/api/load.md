# `datasets.load`

## Contract

```python
datasets.load(source, verbose=True) -> Detections
```

Load a flat YOLO detection dataset into `klygo.outputs.detect.Detections`.
For a real train/validation/test dataset, call `unpartition()` first and pass
the created flat dataset to `load()`. The loader reads images through
`klygo.media` and returns one `Detection` per image.

The input is unchanged. Flattening encodes relative paths to keep images with
the same basename in different directories distinct.

## Parameters

- `source`: flat dataset directory or supported archive; accepts strings,
  `pathlib.Path`, and text-returning `os.PathLike` objects. Nested roots are
  found using the existing dataset root discovery.
- `verbose`: display loading progress; defaults to `True`.

## Returns

A `Detections` collection with `source_type="images"`. Each detection contains:

- An RGB PIL image fully loaded into memory.
- Boxes converted from normalized YOLO center/size coordinates to pixel XYXY.
- Ground-truth scores of `1.0`, rather than model confidence estimates.
- `box.metadata["class_id"]` preserving the original numeric class ID, including
  sparse IDs, and `box.metadata["ground_truth"] = True`.
- Detection metadata with `dataset_source`, `relative_path` relative to the
  input dataset's images directory, `split` (`train`, `val`, `test`, or `raw`),
  and `ground_truth = True`.

The output supports the existing `show()`, `crop()`, filtering, and
postprocessing APIs. Images are materialized before return, so images and crops
remain usable even if the source is removed.
`source_path` is unset because the returned images are in memory; use the
metadata fields for provenance.

## Errors and edge cases

- Missing sources or image directories raise `FileNotFoundError`.
- A partitioned dataset raises `ValueError` with guidance to call
  `datasets.unpartition()` first.
- Unsupported source files, invalid class metadata, or datasets without
  supported images raise `ValueError`.
- Missing or empty label files produce detections with zero boxes. Blank
  annotation lines are ignored.
- Labels must contain exactly `class_id cx cy width height`. Unknown class IDs,
  nonnumeric values, nonfinite values, values outside `[0, 1]`, and zero box
  dimensions raise `ValueError` with the label path and line number.
- Coordinates are converted without clipping or rounding. Segmentation,
  pose, and confidence-appended label rows are not accepted.
- Unreadable or corrupt images raise `OSError`.

## AI usage guidance

Use this API after flattening when annotations should be available through the
same output types used by model predictions. Memory use scales with all decoded
image pixels, and flattening requires disk space for a copy of the dataset. Use
[`info`](info.md) when only statistics are needed.

Pass the directory or archive returned by [`unpartition`](unpartition.md)
directly. Encoded filenames preserve their original relative paths and split
metadata.

## Example

```python
from klygo import datasets

flat = datasets.unpartition("dataset.zip", "flat")
results = datasets.load(flat)
results[0].show()
crops = results.crop()
print(results.total_objects, results.label_counts)
```

## Tests

[`test_load.py`](../../../test/datasets/test_load.py) covers all splits,
duplicate filenames, sparse class IDs, empty labels, path-like inputs, nested
archive roots, cleanup, invalid labels, corrupt images, and export/load
round trips.

## Complete executable example

Run the self-contained [`load.py`](../../../examples/datasets/load.py) example:

```bash
python examples/datasets/load.py
```
