# `klygo.preprocessing` Documentation

`klygo.preprocessing` provides 137 lazy, backend-aware image transformations, quality metrics, conditional predicates, and reversible coordinate-trace utilities. Import with `from klygo import preprocessing as p`. Operations applied to `media.LazyImage` remain lazy until materialization.

Interactive Google Colab Tutorial:
[Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/preprocessing/Document_Preprocessing_Overview.ipynb)

Local Tutorial Notebooks (`tutorials/preprocessing/`, 1-to-1 with `klygo/preprocessing/` submodules):
- [`Document_Preprocessing_Overview.ipynb`](../../tutorials/preprocessing/Document_Preprocessing_Overview.ipynb)
- [`Document_Preprocessing_Core.ipynb`](../../tutorials/preprocessing/Document_Preprocessing_Core.ipynb) (`klygo/preprocessing/core.py`)
- [`Document_Preprocessing_Transforms.ipynb`](../../tutorials/preprocessing/Document_Preprocessing_Transforms.ipynb) (`klygo/preprocessing/transforms.py`)
- [`Document_Preprocessing_Metrics.ipynb`](../../tutorials/preprocessing/Document_Preprocessing_Metrics.ipynb) (`klygo/preprocessing/metrics.py`)
- [`Document_Preprocessing_Trace.ipynb`](../../tutorials/preprocessing/Document_Preprocessing_Trace.ipynb) (`klygo/preprocessing/trace.py`)

---

## 1. Lazy Pipelines, Quality Predicates & Automatic Coordinate Restoration (`trace`)

<details open>
<summary><b>How lazy execution, quality predicates, and coordinate restoration work together</b> <i>(Click to expand or collapse)</i></summary>

1. **Lazy Pipelines on `media.LazyImage` (`Operation` & `Compose`)**:
   - Every transform in `transforms.py` returns a serializable [`Operation`](api/Operation.md) and pipelines are built with [`compose()`](api/compose.md) / [`Compose`](api/Compose.md).
   - Applying an `Operation` or `Compose` pipeline to a `media.LazyImage` appends operations to the image's lazy queue without decoding pixels until [`materialize()`](api/materialize.md) (or `.to_pil()` / `.to_numpy()`) is called. Use [`inspect()`](api/inspect.md) to preview queued steps and [`optimize()`](api/optimize.md) to fuse redundant operations.
2. **Quality Scoring & Conditional Predicates (`Metric` & `Predicate`)**:
   - Functions in `metrics.py` (such as [`blur_score()`](api/blur_score.md), [`brightness_score()`](api/brightness_score.md), [`contrast_score()`](api/contrast_score.md), and [`perceptual_hash()`](api/perceptual_hash.md)) return reusable [`Metric`](api/Metric.md) evaluators that compute fast preview statistics.
   - Comparing a `Metric` (e.g., `p.brightness_score() < 80`) or combining conditions with [`all_of()`](api/all_of.md), [`any_of()`](api/any_of.md), [`not_()`](api/not_.md), and [`between()`](api/between.md) produces a [`Predicate`](api/Predicate.md) that gates transforms via `op.when(predicate)`.
3. **Automatic Coordinate Restoration (`trace.py` & `models.Detector`)**:
   - Geometry-changing transforms (`resize`, `letterbox`, `pad`, `crop`, `center_crop`, `flip`, `rotate`, `affine`, `perspective`, etc.) record exact before/after spatial parameters on the image's preprocessing trace.
   - Call [`get_trace()`](api/get_trace.md) and pass the history to [`restore_boxes()`](api/restore_boxes.md), [`restore_points()`](api/restore_points.md), or [`restore_mask()`](api/restore_mask.md) to map predictions from preprocessed coordinates back to the original image space—or pass preprocessed images directly to `models.Detector`, which reads the trace automatically.

</details>

---

## 2. Task-Oriented Cheatsheet (24 Common Preprocessing Tasks)

<details open>
<summary><b>Task-oriented quick reference (24 tasks)</b> <i>(Click to expand or collapse)</i></summary>

| I want to... | Recommended API | Key Parameters / Example |
| :--- | :--- | :--- |
| **Chain multiple transforms into a reusable pipeline** | [`compose()`](api/compose.md) | `pipe = p.compose(p.auto_orient(), p.letterbox((640, 640)))` |
| **Execute a lazy pipeline to PIL / NumPy / tensor** | [`materialize()`](api/materialize.md) | `pil_img = p.materialize(lazy_img, output="pil")` |
| **Inspect or fuse queued operations before decoding** | [`inspect()`](api/inspect.md) / [`optimize()`](api/optimize.md) | `plan = p.inspect(lazy_img); fast_pipe = p.optimize(pipe)` |
| **Run a transform conditionally based on image quality** | [`Predicate`](api/Predicate.md) / [`between()`](api/between.md) | `op = p.clahe().when(p.brightness_score() < 90)` |
| **Combine multiple quality conditions** | [`all_of()`](api/all_of.md) / [`any_of()`](api/any_of.md) | `ok = p.all_of(p.blur_score() > 50, p.between(p.brightness_score(), 40, 220))` |
| **Resize with aspect-ratio preservation & padding** | [`letterbox()`](api/letterbox.md) | `img2 = p.letterbox((640, 640), fill=(114, 114, 114))(img)` |
| **Resize to exact width/height or fit inside bounds** | [`resize()`](api/resize.md) / [`contain()`](api/contain.md) | `p.resize((512, 512))(img)`, `p.contain((512, 512))(img)` |
| **Crop a region or center window** | [`crop()`](api/crop.md) / [`center_crop()`](api/center_crop.md) | `p.crop((x1, y1, x2, y2))(img)`, `p.center_crop((224, 224))(img)` |
| **Slice a high-resolution image into overlapping tiles** | [`tile()`](api/tile.md) | `tiles = p.tile((640, 640), overlap=0.2)(img)` |
| **Fix camera EXIF orientation and strip metadata** | [`auto_orient()`](api/auto_orient.md) / [`strip_exif()`](api/strip_exif.md) | `clean = p.compose(p.auto_orient(), p.strip_exif())(img)` |
| **Rotate, flip, or apply perspective warp** | [`rotate()`](api/rotate.md) / [`flip()`](api/flip.md) / [`perspective()`](api/perspective.md) | `p.rotate(90, expand=True)(img)`, `p.flip("horizontal")(img)` |
| **Convert color spaces or channel ordering** | [`rgb()`](api/rgb.md) / [`bgr()`](api/bgr.md) / [`grayscale()`](api/grayscale.md) | `p.rgb()(img)`, `p.bgr()(img)`, `p.grayscale()(img)` |
| **Boost local contrast on dark or hazy images** | [`clahe()`](api/clahe.md) / [`autocontrast()`](api/autocontrast.md) | `p.clahe(clip_limit=2.0, tile_grid_size=(8, 8))(img)` |
| **Correct color cast and lighting temperature** | [`white_balance()`](api/white_balance.md) / [`gamma()`](api/gamma.md) | `p.white_balance()(img)`, `p.gamma(1.2)(img)` |
| **Denoise or sharpen edges** | [`bilateral_filter()`](api/bilateral_filter.md) / [`unsharp_mask()`](api/unsharp_mask.md) | `p.bilateral_filter()(img)`, `p.unsharp_mask(radius=2.0)(img)` |
| **Binarize documents with adaptive/Otsu thresholding** | [`adaptive_threshold()`](api/adaptive_threshold.md) / [`otsu_threshold()`](api/otsu_threshold.md) | `p.otsu_threshold()(img)` |
| **Extract edges or apply morphological cleanup** | [`canny()`](api/canny.md) / [`opening()`](api/opening.md) / [`closing()`](api/closing.md) | `p.canny(100, 200)(img)`, `p.opening(kernel_size=3)(img)` |
| **Normalize pixels and convert to NCHW model tensor** | [`normalize()`](api/normalize.md) / [`tensor()`](api/tensor.md) | `p.compose(p.rescale(1 / 255.0), p.normalize(), p.tensor())(img)` |
| **Build a pipeline automatically from model input spec** | [`from_input_spec()`](api/from_input_spec.md) | `pipe = p.from_input_spec({"size": (640, 640), "letterbox": True})` |
| **Apply deterministic data augmentations** | [`seed()`](api/seed.md) / [`color_jitter()`](api/color_jitter.md) | `p.compose(p.seed(42), p.random_flip(), p.color_jitter(0.2))(img)` |
| **Measure blur, sharpness, brightness, or noise** | [`blur_score()`](api/blur_score.md) / [`brightness_score()`](api/brightness_score.md) | `score = p.blur_score()(img); lum = p.brightness_score()(img)` |
| **Compute perceptual hash or image similarity** | [`perceptual_hash()`](api/perceptual_hash.md) / [`similarity_score()`](api/similarity_score.md) | `digest = p.perceptual_hash()(img)` |
| **Inspect original vs. processed image dimensions** | [`original_size()`](api/original_size.md) / [`processed_size()`](api/processed_size.md) | `orig_wh = p.original_size(processed_img)` |
| **Restore predicted boxes/points/masks to original image** | [`restore_boxes()`](api/restore_boxes.md) / [`get_trace()`](api/get_trace.md) | `boxes = p.restore_boxes(pred_boxes, p.get_trace(processed_img))` |

</details>

---

## Core

Implemented in `core.py`.

| API | Example |
| --- | --- |
| [`Operation`](api/Operation.md) | [Run](../../examples/preprocessing/Operation.py) |
| [`Compose`](api/Compose.md) | [Run](../../examples/preprocessing/Compose.py) |
| [`Metric`](api/Metric.md) | [Run](../../examples/preprocessing/Metric.py) |
| [`Predicate`](api/Predicate.md) | [Run](../../examples/preprocessing/Predicate.py) |
| [`compose`](api/compose.md) | [Run](../../examples/preprocessing/compose.py) |
| [`apply`](api/apply.md) | [Run](../../examples/preprocessing/apply.py) |
| [`materialize`](api/materialize.md) | [Run](../../examples/preprocessing/materialize.py) |
| [`inspect`](api/inspect.md) | [Run](../../examples/preprocessing/inspect.py) |
| [`optimize`](api/optimize.md) | [Run](../../examples/preprocessing/optimize.py) |
| [`custom`](api/custom.md) | [Run](../../examples/preprocessing/custom.py) |
| [`register`](api/register.md) | [Run](../../examples/preprocessing/register.py) |
| [`execute_pipeline`](api/execute_pipeline.md) | [Run](../../examples/preprocessing/execute_pipeline.py) |
| [`all_of`](api/all_of.md) | [Run](../../examples/preprocessing/all_of.py) |
| [`any_of`](api/any_of.md) | [Run](../../examples/preprocessing/any_of.py) |
| [`not_`](api/not_.md) | [Run](../../examples/preprocessing/not_.py) |
| [`between`](api/between.md) | [Run](../../examples/preprocessing/between.py) |

## Transforms

Implemented in `transforms.py`.

| API | Example |
| --- | --- |
| [`crop`](api/crop.md) | [Run](../../examples/preprocessing/crop.py) |
| [`center_crop`](api/center_crop.md) | [Run](../../examples/preprocessing/center_crop.py) |
| [`resize`](api/resize.md) | [Run](../../examples/preprocessing/resize.py) |
| [`thumbnail`](api/thumbnail.md) | [Run](../../examples/preprocessing/thumbnail.py) |
| [`contain`](api/contain.md) | [Run](../../examples/preprocessing/contain.py) |
| [`cover`](api/cover.md) | [Run](../../examples/preprocessing/cover.py) |
| [`fit`](api/fit.md) | [Run](../../examples/preprocessing/fit.py) |
| [`letterbox`](api/letterbox.md) | [Run](../../examples/preprocessing/letterbox.py) |
| [`pad`](api/pad.md) | [Run](../../examples/preprocessing/pad.py) |
| [`flip`](api/flip.md) | [Run](../../examples/preprocessing/flip.py) |
| [`rotate`](api/rotate.md) | [Run](../../examples/preprocessing/rotate.py) |
| [`transpose`](api/transpose.md) | [Run](../../examples/preprocessing/transpose.py) |
| [`affine`](api/affine.md) | [Run](../../examples/preprocessing/affine.py) |
| [`perspective`](api/perspective.md) | [Run](../../examples/preprocessing/perspective.py) |
| [`warp`](api/warp.md) | [Run](../../examples/preprocessing/warp.py) |
| [`tile`](api/tile.md) | [Run](../../examples/preprocessing/tile.py) |
| [`auto_orient`](api/auto_orient.md) | [Run](../../examples/preprocessing/auto_orient.py) |
| [`exif_transpose`](api/exif_transpose.md) | [Run](../../examples/preprocessing/exif_transpose.py) |
| [`strip_exif`](api/strip_exif.md) | [Run](../../examples/preprocessing/strip_exif.py) |
| [`convert_color`](api/convert_color.md) | [Run](../../examples/preprocessing/convert_color.py) |
| [`grayscale`](api/grayscale.md) | [Run](../../examples/preprocessing/grayscale.py) |
| [`rgb`](api/rgb.md) | [Run](../../examples/preprocessing/rgb.py) |
| [`bgr`](api/bgr.md) | [Run](../../examples/preprocessing/bgr.py) |
| [`rgba`](api/rgba.md) | [Run](../../examples/preprocessing/rgba.py) |
| [`add_alpha`](api/add_alpha.md) | [Run](../../examples/preprocessing/add_alpha.py) |
| [`remove_alpha`](api/remove_alpha.md) | [Run](../../examples/preprocessing/remove_alpha.py) |
| [`select_channels`](api/select_channels.md) | [Run](../../examples/preprocessing/select_channels.py) |
| [`swap_channels`](api/swap_channels.md) | [Run](../../examples/preprocessing/swap_channels.py) |
| [`brightness`](api/brightness.md) | [Run](../../examples/preprocessing/brightness.py) |
| [`contrast`](api/contrast.md) | [Run](../../examples/preprocessing/contrast.py) |
| [`saturation`](api/saturation.md) | [Run](../../examples/preprocessing/saturation.py) |
| [`hue`](api/hue.md) | [Run](../../examples/preprocessing/hue.py) |
| [`gamma`](api/gamma.md) | [Run](../../examples/preprocessing/gamma.py) |
| [`temperature`](api/temperature.md) | [Run](../../examples/preprocessing/temperature.py) |
| [`tint`](api/tint.md) | [Run](../../examples/preprocessing/tint.py) |
| [`white_balance`](api/white_balance.md) | [Run](../../examples/preprocessing/white_balance.py) |
| [`autocontrast`](api/autocontrast.md) | [Run](../../examples/preprocessing/autocontrast.py) |
| [`equalize`](api/equalize.md) | [Run](../../examples/preprocessing/equalize.py) |
| [`clahe`](api/clahe.md) | [Run](../../examples/preprocessing/clahe.py) |
| [`invert`](api/invert.md) | [Run](../../examples/preprocessing/invert.py) |
| [`solarize`](api/solarize.md) | [Run](../../examples/preprocessing/solarize.py) |
| [`posterize`](api/posterize.md) | [Run](../../examples/preprocessing/posterize.py) |
| [`colorize`](api/colorize.md) | [Run](../../examples/preprocessing/colorize.py) |
| [`blur`](api/blur.md) | [Run](../../examples/preprocessing/blur.py) |
| [`box_blur`](api/box_blur.md) | [Run](../../examples/preprocessing/box_blur.py) |
| [`gaussian_blur`](api/gaussian_blur.md) | [Run](../../examples/preprocessing/gaussian_blur.py) |
| [`median_blur`](api/median_blur.md) | [Run](../../examples/preprocessing/median_blur.py) |
| [`bilateral_filter`](api/bilateral_filter.md) | [Run](../../examples/preprocessing/bilateral_filter.py) |
| [`sharpen`](api/sharpen.md) | [Run](../../examples/preprocessing/sharpen.py) |
| [`unsharp_mask`](api/unsharp_mask.md) | [Run](../../examples/preprocessing/unsharp_mask.py) |
| [`denoise`](api/denoise.md) | [Run](../../examples/preprocessing/denoise.py) |
| [`detail`](api/detail.md) | [Run](../../examples/preprocessing/detail.py) |
| [`smooth`](api/smooth.md) | [Run](../../examples/preprocessing/smooth.py) |
| [`emboss`](api/emboss.md) | [Run](../../examples/preprocessing/emboss.py) |
| [`convolve`](api/convolve.md) | [Run](../../examples/preprocessing/convolve.py) |
| [`threshold`](api/threshold.md) | [Run](../../examples/preprocessing/threshold.py) |
| [`adaptive_threshold`](api/adaptive_threshold.md) | [Run](../../examples/preprocessing/adaptive_threshold.py) |
| [`otsu_threshold`](api/otsu_threshold.md) | [Run](../../examples/preprocessing/otsu_threshold.py) |
| [`triangle_threshold`](api/triangle_threshold.md) | [Run](../../examples/preprocessing/triangle_threshold.py) |
| [`morphology`](api/morphology.md) | [Run](../../examples/preprocessing/morphology.py) |
| [`erode`](api/erode.md) | [Run](../../examples/preprocessing/erode.py) |
| [`dilate`](api/dilate.md) | [Run](../../examples/preprocessing/dilate.py) |
| [`opening`](api/opening.md) | [Run](../../examples/preprocessing/opening.py) |
| [`closing`](api/closing.md) | [Run](../../examples/preprocessing/closing.py) |
| [`morphological_gradient`](api/morphological_gradient.md) | [Run](../../examples/preprocessing/morphological_gradient.py) |
| [`tophat`](api/tophat.md) | [Run](../../examples/preprocessing/tophat.py) |
| [`blackhat`](api/blackhat.md) | [Run](../../examples/preprocessing/blackhat.py) |
| [`canny`](api/canny.md) | [Run](../../examples/preprocessing/canny.py) |
| [`sobel`](api/sobel.md) | [Run](../../examples/preprocessing/sobel.py) |
| [`scharr`](api/scharr.md) | [Run](../../examples/preprocessing/scharr.py) |
| [`laplacian`](api/laplacian.md) | [Run](../../examples/preprocessing/laplacian.py) |
| [`find_edges`](api/find_edges.md) | [Run](../../examples/preprocessing/find_edges.py) |
| [`cast`](api/cast.md) | [Run](../../examples/preprocessing/cast.py) |
| [`rescale`](api/rescale.md) | [Run](../../examples/preprocessing/rescale.py) |
| [`normalize`](api/normalize.md) | [Run](../../examples/preprocessing/normalize.py) |
| [`clamp`](api/clamp.md) | [Run](../../examples/preprocessing/clamp.py) |
| [`channels_first`](api/channels_first.md) | [Run](../../examples/preprocessing/channels_first.py) |
| [`channels_last`](api/channels_last.md) | [Run](../../examples/preprocessing/channels_last.py) |
| [`add_batch_dimension`](api/add_batch_dimension.md) | [Run](../../examples/preprocessing/add_batch_dimension.py) |
| [`remove_batch_dimension`](api/remove_batch_dimension.md) | [Run](../../examples/preprocessing/remove_batch_dimension.py) |
| [`tensor`](api/tensor.md) | [Run](../../examples/preprocessing/tensor.py) |
| [`from_input_spec`](api/from_input_spec.md) | [Run](../../examples/preprocessing/from_input_spec.py) |
| [`apply_mask`](api/apply_mask.md) | [Run](../../examples/preprocessing/apply_mask.py) |
| [`composite`](api/composite.md) | [Run](../../examples/preprocessing/composite.py) |
| [`blend`](api/blend.md) | [Run](../../examples/preprocessing/blend.py) |
| [`add`](api/add.md) | [Run](../../examples/preprocessing/add.py) |
| [`subtract`](api/subtract.md) | [Run](../../examples/preprocessing/subtract.py) |
| [`difference`](api/difference.md) | [Run](../../examples/preprocessing/difference.py) |
| [`multiply`](api/multiply.md) | [Run](../../examples/preprocessing/multiply.py) |
| [`alpha_composite`](api/alpha_composite.md) | [Run](../../examples/preprocessing/alpha_composite.py) |
| [`random_apply`](api/random_apply.md) | [Run](../../examples/preprocessing/random_apply.py) |
| [`random_choice`](api/random_choice.md) | [Run](../../examples/preprocessing/random_choice.py) |
| [`random_crop`](api/random_crop.md) | [Run](../../examples/preprocessing/random_crop.py) |
| [`random_resize`](api/random_resize.md) | [Run](../../examples/preprocessing/random_resize.py) |
| [`random_flip`](api/random_flip.md) | [Run](../../examples/preprocessing/random_flip.py) |
| [`random_rotate`](api/random_rotate.md) | [Run](../../examples/preprocessing/random_rotate.py) |
| [`color_jitter`](api/color_jitter.md) | [Run](../../examples/preprocessing/color_jitter.py) |
| [`random_erasing`](api/random_erasing.md) | [Run](../../examples/preprocessing/random_erasing.py) |
| [`seed`](api/seed.md) | [Run](../../examples/preprocessing/seed.py) |

## Metrics

Implemented in `metrics.py`.

| API | Example |
| --- | --- |
| [`brightness_score`](api/brightness_score.md) | [Run](../../examples/preprocessing/brightness_score.py) |
| [`contrast_score`](api/contrast_score.md) | [Run](../../examples/preprocessing/contrast_score.py) |
| [`sharpness_score`](api/sharpness_score.md) | [Run](../../examples/preprocessing/sharpness_score.py) |
| [`blur_score`](api/blur_score.md) | [Run](../../examples/preprocessing/blur_score.py) |
| [`entropy_score`](api/entropy_score.md) | [Run](../../examples/preprocessing/entropy_score.py) |
| [`noise_score`](api/noise_score.md) | [Run](../../examples/preprocessing/noise_score.py) |
| [`exposure_score`](api/exposure_score.md) | [Run](../../examples/preprocessing/exposure_score.py) |
| [`motion_score`](api/motion_score.md) | [Run](../../examples/preprocessing/motion_score.py) |
| [`similarity_score`](api/similarity_score.md) | [Run](../../examples/preprocessing/similarity_score.py) |
| [`perceptual_hash`](api/perceptual_hash.md) | [Run](../../examples/preprocessing/perceptual_hash.py) |
| [`histogram`](api/histogram.md) | [Run](../../examples/preprocessing/histogram.py) |
| [`mean`](api/mean.md) | [Run](../../examples/preprocessing/mean.py) |
| [`std`](api/std.md) | [Run](../../examples/preprocessing/std.py) |
| [`min_max`](api/min_max.md) | [Run](../../examples/preprocessing/min_max.py) |
| [`percentile`](api/percentile.md) | [Run](../../examples/preprocessing/percentile.py) |

## Trace

Implemented in `trace.py`.

| API | Example |
| --- | --- |
| [`get_trace`](api/get_trace.md) | [Run](../../examples/preprocessing/get_trace.py) |
| [`original_size`](api/original_size.md) | [Run](../../examples/preprocessing/original_size.py) |
| [`processed_size`](api/processed_size.md) | [Run](../../examples/preprocessing/processed_size.py) |
| [`restore_boxes`](api/restore_boxes.md) | [Run](../../examples/preprocessing/restore_boxes.py) |
| [`restore_points`](api/restore_points.md) | [Run](../../examples/preprocessing/restore_points.py) |
| [`restore_mask`](api/restore_mask.md) | [Run](../../examples/preprocessing/restore_mask.py) |
| [`clear_trace`](api/clear_trace.md) | [Run](../../examples/preprocessing/clear_trace.py) |

Start with the [overview notebook](../../tutorials/preprocessing/Document_Preprocessing_Overview.ipynb), then open the [Core](../../tutorials/preprocessing/Document_Preprocessing_Core.ipynb), [Transforms](../../tutorials/preprocessing/Document_Preprocessing_Transforms.ipynb), [Metrics](../../tutorials/preprocessing/Document_Preprocessing_Metrics.ipynb), or [Trace](../../tutorials/preprocessing/Document_Preprocessing_Trace.ipynb) guides.
