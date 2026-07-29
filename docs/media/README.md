# klygo.media - Bộ công cụ Quản lý I/O Media, Ảnh, Video & Chuyển đổi Tensor

`klygo.media` cung cấp 11 APIs mạnh mẽ giúp đọc, ghi, chuyển đổi định dạng, sao chép, trích xuất khung hình video và chuyển đổi mượt mà giữa **PIL Image**, **NumPy ndarray** và **PyTorch Tensor**.

---

# 1. media.load(source, recursive=False, stream=False, backend='pil', verbose=True)

Đọc 1 file ảnh, file video, hoặc toàn bộ thư mục chứa ảnh.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| source | str \| Path | Bắt buộc | Đường dẫn file ảnh, file video hoặc thư mục chứa ảnh. |
| recursive | bool | False | Duyệt đệ quy qua các thư mục con (khi source là thư mục). |
| stream | bool | False | Trả về Generator đọc đệm từng frame thay vì load toàn bộ vào RAM (dành cho video lớn). |
| backend | str | 'pil' | Thư viện backend nạp ảnh: 'pil' hoặc 'opencv'. |
| verbose | bool | True | Hiển thị thanh tiến trình ProgressBar khi đọc. |

## Giá trị trả về

| Kiểu trả về | Mô tả |
|-------------|-------|
| list[Image.Image \| np.ndarray] \| Generator | Danh sách hoặc Generator các ảnh đã đọc. |

## Ví dụ

```python
import klygo.media as media

imgs = media.load("image.jpg")
frames = media.load("video.mp4")
```

---

# 2. media.save(path, image, overwrite=False, verbose=True)

Lưu một đối tượng ảnh (PIL Image hoặc NumPy array) ra tập tin đĩa.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| path | str \| Path | Bắt buộc | Đường dẫn file ảnh đích cần lưu. |
| image | Image.Image \| np.ndarray | Bắt buộc | Đối tượng dữ liệu ảnh cần ghi. |
| overwrite | bool | False | Cho phép ghi đè nếu file ảnh đã tồn tại. |
| verbose | bool | True | Hiển thị thanh tiến trình ProgressBar khi lưu. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| Path |

## Ví dụ

```python
import klygo.media as media

media.save("output.jpg", img_obj, overwrite=True)
```

---

# 3. media.convert(source, target, overwrite=False, verbose=True)

Chuyển đổi định dạng file ảnh (vd: `.png` sang `.jpg`) hoặc file video (vd: `.avi` sang `.mp4`).

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| source | str \| Path | Bắt buộc | Đường dẫn file media nguồn. |
| target | str \| Path | Bắt buộc | Đường dẫn file media đích cần chuyển đổi. |
| overwrite | bool | False | Cho phép ghi đè nếu file đích đã tồn tại. |
| verbose | bool | True | Hiển thị thanh tiến trình ProgressBar khi chuyển đổi. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| Path |

## Ví dụ

```python
import klygo.media as media

media.convert("image.png", "image.jpg", overwrite=True)
media.convert("video.avi", "video.mp4", overwrite=True)
```

---

# 4. media.copy(source, target, overwrite=False)

Sao chép tập tin ảnh/video hoặc thư mục media sang vị trí mới với kiểm tra tính toàn vẹn media.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| source | str \| Path | Bắt buộc | Đường dẫn file hoặc thư mục media nguồn. |
| target | str \| Path | Bắt buộc | Đường dẫn file hoặc thư mục media đích. |
| overwrite | bool | False | Cho phép ghi đè nếu mục tiêu đã tồn tại. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| Path |

## Ví dụ

```python
import klygo.media as media

media.copy("image.jpg", "backup/image.jpg", overwrite=True)
```

---

# 5. media.save_video(output_path, frames, fps=30.0, fourcc='mp4v', overwrite=False, verbose=True)

Lưu danh sách hoặc Generator các khung hình (frames) thành tập tin video.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| output_path | str \| Path | Bắt buộc | Đường dẫn file video đầu ra. |
| frames | Iterable | Bắt buộc | Danh sách hoặc Generator chứa các khung hình ảnh. |
| fps | float | 30.0 | Số khung hình hiển thị trên mỗi giây. |
| fourcc | str | 'mp4v' | Mã codec video OpenCV (vd: 'mp4v', 'xvid'). |
| overwrite | bool | False | Cho phép ghi đè file nếu đã tồn tại. |
| verbose | bool | True | Hiển thị thanh tiến trình ProgressBar khi đóng gói video. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| Path |

## Ví dụ

```python
import klygo.media as media

media.save_video("output.mp4", frames_list, fps=30, overwrite=True)
```

---

# 6. media.save_images(output_dir, images, prefix='frame', extension='.jpg', overwrite=False, verbose=True)

Lưu chuỗi ảnh/frames vào một thư mục với tên đánh số tăng dần (vd: `frame_000001.jpg`).

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| output_dir | str \| Path | Bắt buộc | Thư mục lưu các file ảnh xuất ra. |
| images | Iterable | Bắt buộc | Danh sách các đối tượng ảnh. |
| prefix | str | 'frame' | Tiền tố tên tập tin ảnh. |
| extension | str | '.jpg' | Đuôi mở rộng file ảnh. |
| overwrite | bool | False | Cho phép ghi đè nếu file ảnh đã tồn tại. |
| verbose | bool | True | Hiển thị thanh tiến trình ProgressBar khi lưu. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| list[Path] |

## Ví dụ

```python
import klygo.media as media

media.save_images("frames_out", frames_list, extension=".jpg")
```

---

# 7. media.iter_frames(source, sample_rate=1, recursive=False, backend='pil', verbose=False)

Generator duyệt từng khung hình (frame) từ file video hoặc thư mục ảnh với tham số bước nhảy (`sample_rate`).

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| source | str \| Path | Bắt buộc | Đường dẫn file video hoặc thư mục ảnh. |
| sample_rate | int | 1 | Bước nhảy duyệt (vd: 1 = duyệt từng frame, 5 = lấy 1 frame mỗi 5 frame). |
| recursive | bool | False | Duyệt đệ quy (nếu source là thư mục). |
| backend | str | 'pil' | Thư viện nạp ảnh: 'pil' hoặc 'opencv'. |
| verbose | bool | False | Hiển thị thanh tiến trình khi duyệt. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| Generator |

## Ví dụ

```python
import klygo.media as media

for frame in media.iter_frames("video.mp4", sample_rate=5):
    process(frame)
```

---

# 8. media.info(path)

Trích xuất thông tin metadata chi tiết của một tập tin ảnh hoặc video.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| path | str \| Path | Bắt buộc | Đường dẫn file ảnh hoặc file video. |

## Giá trị trả về

| Kiểu trả về | Mô tả |
|-------------|-------|
| dict[str, Any] | Metadata (name, path, type, width, height, size, fps, frame_count...). |

## Ví dụ

```python
import klygo.media as media

v_info = media.info("video.mp4")
print(v_info["width"], v_info["height"], v_info["fps"])
```

---

# 9. media.to_array(image)

Chuyển đổi hình ảnh (PIL Image, PyTorch Tensor, hoặc NumPy array) thành mảng NumPy ndarray (`[H, W, C]`).

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| image | Any | Bắt buộc | Đối tượng dữ liệu ảnh. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| np.ndarray |

## Ví dụ

```python
import klygo.media as media

arr = media.to_array(pil_image)
```

---

# 10. media.to_tensor(image, normalize=True)

Chuyển đổi hình ảnh (PIL Image hoặc NumPy array) thành PyTorch Tensor chuẩn mô hình AI (`[C, H, W]`).

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| image | Any | Bắt buộc | Dữ liệu ảnh. |
| normalize | bool | True | Tự động chuẩn hóa giá trị điểm ảnh về dải 0.0 - 1.0. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| torch.Tensor |

## Ví dụ

```python
import klygo.media as media

tensor = media.to_tensor(pil_image)
```

---

# 11. media.to_pil(image)

Chuyển đổi mảng NumPy ndarray hoặc PyTorch Tensor sang đối tượng PIL Image.

## Tham số

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
|----------|--------------|----------|-------|
| image | Any | Bắt buộc | Dữ liệu ảnh. |

## Giá trị trả về

| Kiểu trả về |
|-------------|
| Image.Image |

## Ví dụ

```python
import klygo.media as media

pil_img = media.to_pil(np_arr)
```
