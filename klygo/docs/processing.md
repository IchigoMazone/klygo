# Klygo Processing

`klygo.processing` cung cấp pipeline xử lý ảnh lazy cho `LazyImage`. Việc gọi
transform chỉ lưu operation; pixel chỉ được đọc tại `materialize()`, `save()` hoặc
khi model thật sự yêu cầu dữ liệu ảnh.

## Cách dùng

```python
from klygo import media
from klygo import processing as P

image = media.load("input.jpg")[0]
pipeline = P.compose([
    P.auto_orient(),
    P.letterbox((640, 640)),
    P.clahe(),
    P.normalize(scale=255),
])

processed = pipeline(image)
assert not processed.is_loaded

array = P.materialize(processed, output="array", cache=False)
```

Với một thư mục hoặc video cần truy cập ngẫu nhiên, áp dụng pipeline trực tiếp
cho cả collection. Kết quả vẫn là `MediaFrames` và vẫn chưa decode pixel:

```python
frames = (
    media.load("dataset/images")
    .where(lambda image: image.path.suffix.lower() == ".jpg")
    .transform(pipeline)
)

print(frames.total_frames, frames.loaded_count)
frames[0].show()
frames.unload()  # bỏ pixel cache, vẫn giữ collection để dùng tiếp
```

## 16 nhóm chức năng

1. Pipeline: `compose`, `apply`, `materialize`, `inspect`, `optimize`, `custom`, `register`.
2. Geometry: crop, resize, contain/cover/fit, letterbox, pad, flip, rotate, affine, perspective, warp, tile.
3. EXIF: auto orient, EXIF transpose và strip EXIF.
4. Color/channel: RGB, BGR, RGBA, grayscale, chuyển color space và chọn/đổi channel.
5. Enhancement: brightness, contrast, saturation, hue, gamma, white balance, equalize và CLAHE.
6. Filters: blur, bilateral, sharpen, denoise, emboss và convolution.
7. Threshold: global, adaptive, Otsu và Triangle.
8. Morphology: erode, dilate, opening, closing, gradient, top-hat và black-hat.
9. Edge: Canny, Sobel, Scharr, Laplacian và PIL edge detection.
10. Model input: cast, rescale, normalize, clamp, layout, batch dimension và tensor.
11. Quality metrics: brightness, contrast, sharpness, blur, entropy, noise và exposure.
12. Predicates: `between`, `all_of`, `any_of`, `not_` và toán tử so sánh metric.
13. Statistics: histogram, mean, standard deviation, min/max và percentile.
14. Multi-image: mask, composite, blend, add/subtract/difference/multiply và alpha composite.
15. Transform trace: phục hồi box, point và mask về hệ tọa độ ảnh gốc.
16. Augmentation: random apply/choice/crop/resize/flip/rotate, color jitter và random erasing.

## Streaming

```python
frames = (
    media.stream("video.mp4", sample_rate=3)
    .where_quality(min_sharpness=80, min_entropy=2.0)
    .where_motion(threshold=0.02)
    .deduplicate(threshold=0.95)
    .transform(pipeline)
)

for frame in frames:
    result = model.predict(frame)
```

Quality metrics dùng preview nhỏ và `cache=False`. Stream chỉ giữ frame hiện tại
hoặc cặp frame cần so sánh, nên bộ nhớ không tăng theo chiều dài video.

## Transform trace

```python
processed = P.letterbox((640, 640))(image)
prediction = model.predict(processed)

original_boxes = P.restore_boxes(
    prediction.boxes,
    P.get_trace(processed),
)
```

`restore_points()` và `restore_mask()` sử dụng cùng trace. Các transform geometry
luôn ghi kích thước trước/sau cùng tham số scale, padding, crop hoặc matrix.

Kết quả detection có thể trở lại ảnh RGB nguồn ở hai không gian:

```python
result = model.predict(processed)[0]

same_geometry = result.restore(target="transformed")
# RGB nguồn + giữ resize/crop/letterbox; box không đổi.

original = result.restore(target="original")
# RGB và kích thước nguyên bản; box được phục hồi tự động.
```

Mặc định `restore()` trả về `Detection` mới. Dùng `inplace=True` khi thực sự
muốn cập nhật trực tiếp kết quả hiện tại.

Với kết quả nhiều ảnh hoặc video, gọi trực tiếp trên collection; stream vẫn
được xử lý tuần tự:

```python
restored_results = results.restore(target="original")
```
