"""
Bộ công cụ Quản lý I/O Media, Ảnh, Video & Chuyển đổi Tensor (`klygo.media`).

Hướng dẫn sử dụng (Google Colab):
https://colab.research.google.com/drive/1bXduGq2Y4YEfOQlutNMYmz6nYmrJPsmq?usp=sharing

Định dạng hỗ trợ:
  - Ảnh: .png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff
  - Video: .mp4, .avi, .mov, .mkv, .m4v, .webm

Danh sách 11 APIs:
  1.  load(source, ...) - Đọc 1 ảnh, thư mục ảnh hoặc trích xuất toàn bộ frames của video
  2.  save(path, image, ...) - Ghi đối tượng ảnh ra file đĩa
  3.  convert(source, target, ...) - Chuyển đổi định dạng file ảnh (.png -> .jpg) hoặc video (.avi -> .mp4)
  4.  copy(source, target, ...) - Sao chép tập tin ảnh/video hoặc thư mục media
  5.  save_video(output_path, frames, ...) - Đóng gói danh sách frames thành file video
  6.  save_images(output_dir, images, ...) - Lưu hàng loạt danh sách ảnh/frames ra thư mục ảnh
  7.  iter_frames(video_path, ...) - Duyệt từng frame video dạng Generator tiết kiệm bộ nhớ RAM
  8.  info(source) - Lấy thông tin metadata chi tiết của file ảnh hoặc video
  9.  to_array(image) - Chuyển đổi PIL Image / PyTorch Tensor sang NumPy ndarray
  10. to_tensor(image, ...) - Chuyển đổi PIL Image / NumPy ndarray sang PyTorch Tensor
  11. to_pil(image) - Chuyển đổi NumPy ndarray / PyTorch Tensor sang PIL Image
"""

from .operations import (
    load,
    save,
    convert,
    copy,
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
    "convert",
    "copy",
    "save_video",
    "save_images",
    "iter_frames",
    "info",
    "to_array",
    "to_tensor",
    "to_pil",
]
