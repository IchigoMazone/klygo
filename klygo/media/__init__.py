"""
Bộ công cụ Quản lý I/O Media, Ảnh, Video & Chuyển đổi Tensor (`klygo.media`).

Hướng dẫn sử dụng (Google Colab):
https://colab.research.google.com/drive/1bXduGq2Y4YEfOQlutNMYmz6nYmrJPsmq?usp=sharing

Định dạng hỗ trợ:
  - Ảnh: .png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff
  - Video: .mp4, .avi, .mov, .mkv, .m4v, .webm

Danh sách 12 APIs:
  1.  load(source, ...) - Đọc 1 ảnh, thư mục ảnh hoặc trích xuất toàn bộ frames của video
  2.  stream(source, ...) - Duyệt media tuần tự mà không tạo danh sách proxy toàn bộ nguồn
  3.  save(path, image, ...) - Ghi đối tượng ảnh ra file đĩa
  4.  convert(source, target, ...) - Chuyển đổi định dạng file ảnh (.png -> .jpg) hoặc video (.avi -> .mp4)
  5.  copy(source, target, ...) - Sao chép tập tin ảnh/video hoặc thư mục media
  6.  save_video(output_path, frames, ...) - Đóng gói danh sách frames thành file video
  7.  save_images(output_dir, images, ...) - Lưu hàng loạt danh sách ảnh/frames ra thư mục ảnh
  8.  iter_frames(video_path, ...) - Duyệt từng frame video dạng Generator tiết kiệm bộ nhớ RAM
  9.  probe(source) - Lấy thông tin metadata chi tiết của file ảnh hoặc video
  10. to_array(image) - Chuyển đổi PIL Image / PyTorch Tensor sang NumPy ndarray
  11. to_tensor(image, ...) - Chuyển đổi PIL Image / NumPy ndarray sang PyTorch Tensor
  12. to_pil(image) - Chuyển đổi NumPy ndarray / PyTorch Tensor sang PIL Image
"""

from .operations import (
    load,
    stream,
    save,
    convert,
    copy,
    save_video,
    save_images,
    iter_frames,
    probe,
    to_array,
    to_tensor,
    to_pil,
    MediaFrames,
    MediaStream,
    LazyImage,
    VideoReader,
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
)

__all__ = [
    "load",
    "stream",
    "save",
    "convert",
    "copy",
    "save_video",
    "save_images",
    "iter_frames",
    "probe",
    "to_array",
    "to_tensor",
    "to_pil",
    "MediaFrames",
    "MediaStream",
    "LazyImage",
    "VideoReader",
]
