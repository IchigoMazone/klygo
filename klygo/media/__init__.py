"""
Bộ công cụ Quản lý I/O Media, Ảnh, Video & Chuyển đổi Tensor (`klygo.media`).

Danh sách 9 APIs:
  1. load(source, ...) - Đọc 1 ảnh, thư mục ảnh hoặc trích xuất toàn bộ frames của video
  2. save(path, image, ...) - Ghi đối tượng ảnh ra file
  3. save_video(output_path, frames, ...) - Đóng gói danh sách frames thành file video
  4. save_images(output_dir, images, ...) - Lưu hàng loạt danh sách ảnh/frames ra thư mục ảnh
  5. iter_frames(video_path, ...) - Duyệt từng frame video dạng Generator tiết kiệm bộ nhớ RAM
  6. info(source) - Lấy metadata chi tiết của file ảnh hoặc video
  7. to_array(image) - Chuyển đổi PIL Image / PyTorch Tensor sang NumPy ndarray
  8. to_tensor(image) - Chuyển đổi PIL Image / NumPy ndarray sang PyTorch Tensor
  9. to_pil(image) - Chuyển đổi NumPy ndarray / PyTorch Tensor sang PIL Image
"""

from .operations import (
    load,
    save,
    save_video,
    save_images,
    iter_frames,
    info,
    to_array,
    to_tensor,
    to_pil,
)

__all__ = [
    "load",
    "save",
    "save_video",
    "save_images",
    "iter_frames",
    "info",
    "to_array",
    "to_tensor",
    "to_pil",
]
