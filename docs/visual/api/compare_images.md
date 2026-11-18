# visual.compare_images

```python
compare_images(first, second, *, mode="show", output_path=None,
               titles=("A", "B"), show_scores=True, backend="auto", overwrite=False)
```

Compare two Detection frames directly. Each frame supplies its source image,
boxes, labels, and scores. No separate image or annotation arguments are needed.
Different image sources, dimensions, and label sets are supported. To compare two
label sets on the same image, give both frames that image as their source.

## Parameters

- `first`, `second`: Detection frames. Select one frame from Detections before passing it.
- `mode`: `show` displays (default), `save` writes the comparison.
- `output_path`: required in save mode; prohibited in other modes. Accepts str and os.PathLike.
- `titles`: list or tuple of two strings, displayed above the panels.
- `show_scores`: include confidence scores in annotation captions.
- `backend`: display backend passed to show_image in show mode (auto, matplotlib, or opencv).
- `overwrite`: allow replacing an existing saved file; defaults to False.

## Result and source access

Every mode returns a new RGB PIL image at original image scales, with a 28-pixel
header, 12-pixel gap, and white padding below the shorter panel. Annotations and
source pixels are unchanged. Colors follow draw_bboxes independently per panel.
Detection.image resolves in-memory sources or loads source_path lazily; video
frames use frame_index. Source resolution may cache pixels on the Detection.
The decoded image dimensions must agree with that frame's image_size.

## Errors

TypeError for non-Detection inputs or invalid option types. ValueError for an
unknown mode, invalid title count, missing/unused output_path, or source size
mismatch. RuntimeError when a frame has no source; FileNotFoundError for a missing
source file; FileExistsError when saving over an existing file without overwrite.
Other image conversion and encoding errors propagate from the media APIs.

## Display or save

```python
visual.compare_images(first, second)
visual.compare_images(first, second, mode="save", output_path="artifacts/comparison.jpg")
```

Save mode never opens a viewer. Show mode follows show_image behavior;
desktop auto display can open a viewer and OpenCV waits for a key.

## Complete example

```python
"""Compare two Detection frames and save without opening a viewer."""
from PIL import Image
from klygo import files, outputs, visual

first = outputs.Detection(
    [outputs.Box((40, 50, 140, 170), label="cat", score=0.9)],
    source=Image.new("RGB", (320, 240), "lightgray"),
    source_id="image-a", image_size=(320, 240),
)
second = outputs.Detection(
    [outputs.Box((70, 40, 190, 150), label="dog", score=0.8)],
    source=Image.new("RGB", (280, 200), "lightblue"),
    source_id="image-b", image_size=(280, 200),
)
output = files.path("artifacts/examples/visual/comparison.png")
preview = visual.compare_images(
    first, second, mode="save", output_path=output,
    titles=("Image A", "Image B"), overwrite=True,
)
print(output, preview.size)
```

[Standalone example](../../../examples/visual/compare_images.py) ·
[Tests](../../../test/visual/test_compare_images.py) ·
[Module](../README.md) · [Detection](../../outputs/api/Detection.md)
