import builtins
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional, Generator, Iterable

import cv2 as cv
import numpy as np
from PIL import Image

from klygo.utils.progress import ProgressBar
from klygo.validators import validate_type
from klygo.files import copy as _files_copy

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".webm",
}


# =========================================================================
# 0A. LazyImage: Zero-RAM URL/Path Proxy (PIL + NumPy Hybrid)
# =========================================================================

class LazyImage(Image.Image):
    """
    Đại diện cho một bức ảnh theo cơ chế nạp lười (Zero-RAM URL/Path Proxy).
    Lưu trữ đường dẫn/URL file ảnh với chi phí RAM = 0.
    Hỗ trợ Zero-RAM Lazy Crop: Cắt vùng ảnh con chỉ bằng cách lưu (URL, bounding_box),
    không decode pixel vào RAM và tính toán size bằng hiệu tọa độ tức thì.
    Kế thừa PIL.Image.Image để giữ 100% khả năng tương thích với isinstance(x, Image.Image).
    Hỗ trợ đầy đủ cú pháp của cả PIL Image (.size, .crop, .show, .save, .convert)
    và NumPy ndarray (.shape, .dtype, np.array).
    """

    def __init__(
        self,
        path: Union[str, Path, "LazyImage"],
        backend: str = "pil",
        crop_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self._im = None
        self._image: Optional[Union[Image.Image, np.ndarray]] = None
        self._cached_size: Optional[Tuple[int, int]] = None

        if isinstance(path, LazyImage):
            self._path = path.path
            self._url = path.url
            self.backend = path.backend
            if path.crop_box is not None and crop_box is not None:
                ox1, oy1, _, _ = path.crop_box
                x1, y1, x2, y2 = crop_box
                self._crop_box = (ox1 + x1, oy1 + y1, ox1 + x2, oy1 + y2)
            else:
                self._crop_box = crop_box or path.crop_box
        else:
            self._path = Path(path).resolve()
            self._url = str(self._path)
            self.backend = backend.lower()
            self._crop_box = tuple(int(x) for x in crop_box) if crop_box is not None else None

    @property
    def path(self) -> Path:
        """Đường dẫn tệp ảnh nguồn."""
        return self._path

    @path.setter
    def path(self, new_path: Union[str, Path]) -> None:
        self._path = Path(new_path).resolve()
        self._url = str(self._path)
        self._image = None
        self._cached_size = None

    @property
    def url(self) -> str:
        """URL hoặc chuỗi đường dẫn ảnh nguồn."""
        return self._url

    @url.setter
    def url(self, new_url: Union[str, Path]) -> None:
        self.path = new_url

    @property
    def crop_box(self) -> Optional[Tuple[int, int, int, int]]:
        """Tọa độ vùng cắt (xmin, ymin, xmax, ymax)."""
        return self._crop_box

    @crop_box.setter
    def crop_box(self, box: Optional[Tuple[int, int, int, int]]) -> None:
        self._crop_box = tuple(int(x) for x in box) if box is not None else None
        self._image = None
        self._cached_size = None

    def load(self) -> Union[Image.Image, np.ndarray]:
        """Thực sự nạp ảnh từ đĩa vào bộ nhớ RAM."""
        if self._image is None:
            if self.backend == "opencv":
                arr = cv.imread(str(self.path), cv.IMREAD_COLOR)
                if arr is None:
                    raise FileNotFoundError(f"Could not read image file: {self.path}")
                if self.crop_box is not None:
                    x1, y1, x2, y2 = self.crop_box
                    arr = arr[max(0, y1):min(arr.shape[0], y2), max(0, x1):min(arr.shape[1], x2)]
                self._image = arr
                self._cached_size = (arr.shape[1], arr.shape[0])
            else:
                with Image.open(self.path) as source_image:
                    source_image = source_image.convert("RGB")
                    if self.crop_box is not None:
                        source_image = source_image.crop(self.crop_box)
                    self._image = source_image
                    self._cached_size = self._image.size
        return self._image

    def unload(self) -> None:
        """Giải phóng bộ nhớ pixel, đưa đối tượng về trạng thái URL thuần túy."""
        self._image = None

    def clear(self) -> None:
        """Alias cho unload()."""
        self.unload()

    @property
    def is_loaded(self) -> bool:
        """Kiểm tra ảnh đã nạp pixel vào RAM hay chưa."""
        return self._image is not None

    @property
    def image(self) -> Union[Image.Image, np.ndarray]:
        """Truy cập hoặc gán lại dữ liệu ảnh (hỗ trợ sửa ảnh trực tiếp)."""
        return self.load()

    @image.setter
    def image(self, value: Union[Image.Image, np.ndarray, "LazyImage"]) -> None:
        if isinstance(value, LazyImage):
            self._image = value.load()
            self.path = value.path
            self.url = value.url
            self._cached_size = value.size
            self.crop_box = value.crop_box
        elif isinstance(value, Image.Image):
            self._image = value
            self._cached_size = value.size
            self.crop_box = None
        elif isinstance(value, np.ndarray):
            self._image = value
            self._cached_size = (value.shape[1], value.shape[0])
            self.crop_box = None
        else:
            raise TypeError(f"Expected PIL.Image.Image, np.ndarray, or LazyImage, got {type(value)}")

    @property
    def im(self):
        return self.to_pil().im

    @property
    def info(self) -> dict:
        return getattr(self.to_pil(), "info", {})

    @property
    def format(self) -> Optional[str]:
        return getattr(self.to_pil(), "format", None)

    @property
    def palette(self):
        return getattr(self.to_pil(), "palette", None)

    @property
    def size(self) -> Tuple[int, int]:
        """Kích thước (width, height) theo chuẩn PIL Image."""
        if self.crop_box is not None:
            return (max(0, self.crop_box[2] - self.crop_box[0]), max(0, self.crop_box[3] - self.crop_box[1]))
        if self._cached_size is not None:
            return self._cached_size
        if self._image is not None:
            if isinstance(self._image, Image.Image):
                self._cached_size = self._image.size
            else:
                self._cached_size = (self._image.shape[1], self._image.shape[0])
            return self._cached_size
        try:
            with Image.open(self.path) as im:
                self._cached_size = im.size
                return self._cached_size
        except Exception:
            return (0, 0)

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Kích thước (height, width, channels) theo chuẩn NumPy ndarray."""
        return (self.height, self.width, 3)

    @property
    def dtype(self) -> np.dtype:
        """Kiểu dữ liệu chuẩn NumPy (uint8)."""
        return np.dtype("uint8")

    @property
    def mode(self) -> str:
        return "RGB"

    def to_pil(self) -> Image.Image:
        """Chuyển thành PIL Image thật."""
        loaded = self.load()
        if isinstance(loaded, Image.Image):
            return loaded
        return Image.fromarray(cv.cvtColor(loaded, cv.COLOR_BGR2RGB))

    def to_array(self) -> np.ndarray:
        """Chuyển thành NumPy array thật."""
        loaded = self.load()
        if isinstance(loaded, np.ndarray):
            return loaded
        return np.array(loaded)

    def lazy_crop(self, box: Tuple[int, int, int, int]) -> "LazyImage":
        """
        Zero-RAM Lazy Cropping: Tạo ảnh con từ vùng cắt mà KHÔNG nạp pixel vào RAM.
        Chỉ lưu tọa độ bounding box và đường dẫn file ảnh gốc.
        """
        return LazyImage(self, backend=self.backend, crop_box=box)

    def crop(self, *args, lazy: bool = False, **kwargs):
        if lazy and args:
            return self.lazy_crop(args[0])
        return self.to_pil().crop(*args, **kwargs)

    def convert(self, *args, **kwargs):
        return self.to_pil().convert(*args, **kwargs)

    def resize(self, *args, **kwargs):
        return self.to_pil().resize(*args, **kwargs)

    def rotate(self, *args, **kwargs):
        return self.to_pil().rotate(*args, **kwargs)

    def transpose(self, *args, **kwargs):
        return self.to_pil().transpose(*args, **kwargs)

    def filter(self, *args, **kwargs):
        return self.to_pil().filter(*args, **kwargs)

    def copy(self):
        return self.to_pil().copy()

    def save(self, *args, **kwargs):
        return self.to_pil().save(*args, **kwargs)

    def show(self, *args, **kwargs):
        return self.to_pil().show(*args, **kwargs)

    def tobytes(self, *args, **kwargs):
        return self.to_pil().tobytes(*args, **kwargs)

    def getpixel(self, *args, **kwargs):
        return self.to_pil().getpixel(*args, **kwargs)

    def putpixel(self, *args, **kwargs):
        return self.to_pil().putpixel(*args, **kwargs)

    def getdata(self, *args, **kwargs):
        return self.to_pil().getdata(*args, **kwargs)

    def getbbox(self):
        return self.to_pil().getbbox()

    def getbands(self):
        return self.to_pil().getbands()

    def getextrema(self):
        return self.to_pil().getextrema()

    def split(self):
        return self.to_pil().split()

    def thumbnail(self, *args, **kwargs):
        return self.to_pil().thumbnail(*args, **kwargs)

    def __array__(self, dtype=None) -> np.ndarray:
        """Hỗ trợ tự động chuyển đổi khi gọi np.array(lazy_img)."""
        arr = self.to_array()
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __getattr__(self, name: str) -> Any:
        """Chuyển tiếp tất cả các method và attribute của ảnh thật khi được gọi."""
        loaded = self.load()
        return getattr(loaded, name)

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        status = "loaded" if self._image is not None else "unloaded"
        w, h = self.size
        crop_info = f" crop={self.crop_box}" if self.crop_box is not None else ""
        return f"<LazyImage [{status}] url='{self.path.name}'{crop_info} size=({w}, {h})>"


# =========================================================================
# 0B. MediaFrames: Unified Media Container (List & Stream)
# =========================================================================

class MediaFrames(list):
    """
    Tập hợp các khung hình media (ảnh / video frames) trả về từ `klygo.media.load`.
    Hỗ trợ đồng nhất cả chế độ In-Memory (danh sách) lẫn Stream (tiết kiệm RAM chống tràn bộ nhớ).
    Kế thừa trực tiếp từ `list` để giữ trọn vẹn 100% tương thích ngược với isinstance(..., list).
    """

    def __init__(
        self,
        items: Optional[Union[Iterable, Generator]] = None,
        *,
        stream: bool = False,
        total_frames: Optional[int] = None,
        fps: float = 30.0,
        source_path: Optional[Union[str, Path]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        source_type: str = "video",
        cap: Optional[Any] = None,
    ) -> None:
        self.is_stream = bool(stream)
        self.fps = float(fps) if fps else 30.0
        self.source_path = str(source_path) if source_path is not None else None
        self.width = width
        self.height = height
        self.source_type = source_type
        self.total_frames = total_frames
        self._cap = cap
        self._cache: List[Any] = []

        if self.is_stream:
            super().__init__()
            self._stream_gen = iter(items) if items is not None else iter(())
        else:
            super().__init__(items if items is not None else [])
            if self.total_frames is None:
                self.total_frames = super().__len__()

    def __iter__(self):
        if self.is_stream:
            while self._cache:
                yield self._cache.pop(0)
            yield from self._stream_gen
        else:
            yield from super().__iter__()

    def __len__(self) -> int:
        if self.is_stream:
            return self.total_frames if self.total_frames is not None else len(self._cache)
        return super().__len__()

    def __getitem__(self, index: Union[int, slice]) -> Any:
        if self.is_stream:
            if isinstance(index, int):
                if index < 0:
                    raise IndexError("Negative indexing is not supported in streaming MediaFrames.")
                while len(self._cache) <= index:
                    try:
                        self._cache.append(next(self._stream_gen))
                    except StopIteration:
                        raise IndexError("MediaFrames stream index out of range")
                return self._cache[index]
            elif isinstance(index, slice):
                raise TypeError("Slicing is not supported on streaming MediaFrames. Use list(frames)[slice] instead.")
            raise TypeError(f"Invalid index type: {type(index)}")
        return super().__getitem__(index)

    def to_list(self) -> List[Any]:
        """Chuyển đổi toàn bộ frame thành list chuẩn trong bộ nhớ."""
        if self.is_stream:
            items = list(self)
            self.clear()
            self.extend(items)
            self.is_stream = False
            return items
        return list(self)

    def save_video(self, output_path: Union[str, Path], fps: Optional[float] = None, **kwargs) -> Path:
        """Đóng gói toàn bộ frames thành file video mp4/avi."""
        target_fps = fps if fps is not None else self.fps
        return globals()["save_video"](output_path, self, fps=target_fps, **kwargs)

    def save_images(self, output_dir: Union[str, Path], prefix: str = "frame", **kwargs) -> Path:
        """Lưu toàn bộ frames ra thư mục ảnh."""
        return globals()["save_images"](output_dir, self, prefix=prefix, **kwargs)

    def close(self) -> None:
        """Giải phóng tài nguyên (OpenCV VideoCapture) nếu có."""
        if getattr(self, "_cap", None) is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "MediaFrames":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        if self.is_stream:
            cnt_str = f"{self.total_frames} frames" if self.total_frames is not None else "streaming"
            return f"<MediaFrames (Stream): {cnt_str}, type='{self.source_type}', fps={self.fps}>"
        return f"<MediaFrames: {len(self)} frames, type='{self.source_type}', fps={self.fps}>"


# =========================================================================
# 1. Media Load / Save / Convert / Copy / Info
# =========================================================================

def _read_video_frames(
    path: Path,
    stream: bool = False,
    backend: str = "pil",
    verbose: bool = True,
) -> MediaFrames:
    cap = cv.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {path}")

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)

    def _frame_generator():
        try:
            with ProgressBar(total=total_frames if total_frames > 0 else None, desc=f"Reading video {path.name}", unit="frame", verbose=verbose, colour="cyan") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if backend == "pil":
                        img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                    else:
                        img = frame
                    pbar.update(1)
                    yield img
        finally:
            cap.release()

    if stream:
        return MediaFrames(
            _frame_generator(),
            stream=True,
            total_frames=total_frames,
            fps=fps,
            source_path=path,
            width=width,
            height=height,
            source_type="video",
            cap=cap,
        )
    else:
        return MediaFrames(
            list(_frame_generator()),
            stream=False,
            total_frames=total_frames,
            fps=fps,
            source_path=path,
            width=width,
            height=height,
            source_type="video",
        )


def load(
    source: Union[str, Path],
    recursive: bool = False,
    stream: bool = False,
    backend: str = "pil",
    verbose: bool = True,
) -> MediaFrames:
    """
    Tác dụng:
    - Đọc 1 file ảnh, file video, hoặc toàn bộ thư mục chứa ảnh.
    - Luôn trả về đối tượng `MediaFrames` (kế thừa list) đồng nhất cho cả stream=False và stream=True.

    Định dạng tương thích:
    - Ảnh: .png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff
    - Video: .mp4, .avi, .mov, .mkv, .m4v, .webm

    Đầu vào:
    - source [str | Path]: Đường dẫn file ảnh, file video hoặc thư mục chứa ảnh.
    - recursive [bool]: Duyệt đệ quy qua các thư mục con (khi source là thư mục). Mặc định: False.
    - stream [bool]: Nếu True, trả về MediaFrames đọc đệm từng frame (dùng cho video lớn). Mặc định: False.
    - backend [str]: 'pil' (mặc định) hoặc 'opencv'.
    - verbose [bool]: Hiển thị thanh tiến trình ProgressBar khi đọc. Mặc định: True.

    Đầu ra:
    - [MediaFrames]: Danh sách frames kế thừa list, hỗ trợ cả stream chống tràn RAM, lấy [0], len(frames).

    Ví dụ:
    >>> import klygo.media as media
    >>> imgs = media.load("image.jpg")
    >>> frames = media.load("video.mp4", stream=True)
    >>> len(frames)        # Biết ngay tổng số frame của video!
    >>> first = frames[0]  # Lấy frame đầu tiên mà không làm hỏng luồng stream!
    """
    validate_type(source, (str, Path), "source")
    validate_type(recursive, bool, "recursive")
    validate_type(stream, bool, "stream")
    validate_type(backend, str, "backend")
    validate_type(verbose, bool, "verbose")

    backend = backend.lower()
    if backend not in ("pil", "opencv"):
        raise ValueError("backend must be 'pil' or 'opencv'")

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"source does not exist: {p}")

    if p.is_file():
        suf = p.suffix.lower()
        if suf in VIDEO_SUFFIXES:
            return _read_video_frames(p, stream=stream, backend=backend, verbose=verbose)
        elif suf in IMAGE_SUFFIXES:
            image_paths = [p]
        else:
            raise ValueError(f"source is not a supported image or video file: {p}")
    elif p.is_dir():
        iterator = p.rglob("*") if recursive else p.glob("*")
        image_paths = sorted(
            (
                f
                for f in iterator
                if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda f: str(f).lower(),
        )
        if not image_paths:
            raise ValueError(f"No supported image files found in directory: {p}")
    else:
        raise ValueError(f"source must be a file or directory: {p}")

    # Zero-RAM Lazy Loading: wrap each path in a LazyImage (URL/Path Proxy)
    lazy_frames = [LazyImage(f, backend=backend) for f in image_paths]

    if stream:
        def _images_stream_generator():
            with ProgressBar(total=len(lazy_frames), desc=f"Streaming images from {p.name}", unit="file", verbose=verbose, colour="cyan") as pbar:
                for lz in lazy_frames:
                    yield lz
                    pbar.update(1)

        return MediaFrames(
            _images_stream_generator(),
            stream=True,
            total_frames=len(lazy_frames),
            source_path=p,
            source_type="folder" if p.is_dir() else "image",
        )

    return MediaFrames(
        lazy_frames,
        stream=False,
        total_frames=len(lazy_frames),
        source_path=p,
        source_type="folder" if p.is_dir() else "image",
    )


def save(
    path: Union[str, Path],
    image: Union[Image.Image, np.ndarray],
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Tác dụng:
    - Lưu một đối tượng ảnh (PIL Image, NumPy array, PyTorch Tensor) ra tập tin đĩa.

    Định dạng tương thích:
    - .png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff

    Đầu vào:
    - path [str | Path]: Đường dẫn file ảnh đích cần lưu.
    - image [Image.Image | np.ndarray]: Đối tượng dữ liệu ảnh cần ghi.
    - overwrite [bool]: Ghi đè nếu file đã tồn tại. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình ProgressBar khi lưu. Mặc định: True.

    Đầu ra:
    - [Path]: Đường dẫn file ảnh đã lưu.

    Ví dụ:
    >>> import klygo.media as media
    >>> media.save("output.jpg", img_obj, overwrite=True)
    """
    validate_type(path, (str, Path), "path")
    validate_type(overwrite, bool, "overwrite")
    validate_type(verbose, bool, "verbose")

    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {p}. Use overwrite=True to replace it.")

    p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, LazyImage):
        image = image.load()

    with ProgressBar(total=1, desc=f"Saving image {p.name}", unit="file", verbose=verbose, colour="cyan") as pbar:
        if isinstance(image, Image.Image):
            image.save(p)
        elif isinstance(image, np.ndarray):
            res = cv.imwrite(str(p), image)
            if not res:
                raise ValueError(f"Failed to save image to {p}")
        else:
            raise TypeError("image must be PIL.Image.Image, numpy.ndarray, or LazyImage")
        pbar.update(1)

    return p


def convert(
    source: Union[str, Path],
    target: Union[str, Path],
    codec: Optional[str] = None,
    crf: int = 23,
    fps: Optional[float] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Tác dụng:
    - Chuyển đổi định dạng file ảnh (.png -> .jpg) hoặc file video (.avi -> .mp4, .mkv -> .webm...).
    - Tự động nhận diện nhóm chuyển đổi Web-ready, Storage, Modern Web, hoặc Demo.
    """
    validate_type(source, (str, Path), "source")
    validate_type(target, (str, Path), "target")
    validate_type(overwrite, bool, "overwrite")
    validate_type(verbose, bool, "verbose")

    src_p = Path(source)
    tgt_p = Path(target)

    if not src_p.exists():
        raise FileNotFoundError(f"source file does not exist: {src_p}")

    src_suf = src_p.suffix.lower()
    tgt_suf = tgt_p.suffix.lower()

    if src_suf in IMAGE_SUFFIXES and tgt_suf in IMAGE_SUFFIXES:
        imgs = load(src_p, verbose=False)
        return save(tgt_p, imgs[0], overwrite=overwrite, verbose=verbose)
    elif src_suf in VIDEO_SUFFIXES and tgt_suf in VIDEO_SUFFIXES:
        import shutil
        import subprocess

        fourcc = "mp4v"
        ffmpeg_vcodec = "libx264"
        if codec:
            codec_lower = codec.lower()
            if codec_lower in ("h264", "avc1"):
                fourcc, ffmpeg_vcodec = "avc1", "libx264"
            elif codec_lower in ("h265", "hevc"):
                fourcc, ffmpeg_vcodec = "hevc", "libx265"
            elif codec_lower in ("vp9", "webm"):
                fourcc, ffmpeg_vcodec = "vp09", "libvpx-vp9"
            elif codec_lower == "mjpeg":
                fourcc, ffmpeg_vcodec = "MJPG", "mjpeg"
            else:
                fourcc, ffmpeg_vcodec = codec, codec
        else:
            if tgt_suf == ".mp4":
                fourcc, ffmpeg_vcodec = "avc1", "libx264"
            elif tgt_suf == ".webm":
                fourcc, ffmpeg_vcodec = "vp09", "libvpx-vp9"
            elif tgt_suf == ".avi":
                fourcc, ffmpeg_vcodec = "MJPG", "mjpeg"

        # 1. Try FFmpeg first (preserves audio, handles AV1, much faster)
        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg", "-y" if overwrite else "-n",
                "-err_detect", "ignore_err",
                "-i", str(src_p),
                "-c:v", ffmpeg_vcodec,
                "-pix_fmt", "yuv420p"
            ]
            if fps:
                cmd.extend(["-r", str(fps)])
            cmd.append(str(tgt_p))
            
            try:
                if verbose:
                    print(f"Converting video using FFmpeg: {src_p.name} -> {tgt_p.name}")
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    return tgt_p
                elif verbose:
                    print(f"FFmpeg conversion failed (code {result.returncode}), falling back to OpenCV...")
            except Exception as e:
                if verbose:
                    print(f"FFmpeg execution failed: {e}, falling back to OpenCV...")

        # 2. Fallback to OpenCV (frame-by-frame)
        frames = load(src_p, stream=True, verbose=False)
        v_info = probe(src_p)
        target_fps = fps if fps is not None else v_info.get("fps", 30.0)
        
        try:
            return save_video(tgt_p, frames, fps=target_fps, fourcc=fourcc, overwrite=overwrite, verbose=verbose)
        except ValueError as e:
            if "iterable is empty" in str(e):
                raise RuntimeError(
                    f"OpenCV failed to decode any frames from '{src_p}'. "
                    "The file might use an unsupported codec (e.g., AV1) or is corrupted. "
                    "Please install FFmpeg to handle this format."
                ) from None
            raise
    else:
        raise ValueError(f"Cannot convert from {src_suf} to {tgt_suf}. Both files must be images or both must be videos.")


def copy(
    source: Union[str, Path],
    target: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """
    Tác dụng:
    - Sao chép tập tin ảnh/video hoặc thư mục media sang vị trí mới với kiểm tra tính toàn vẹn media.

    Đầu vào:
    - source [str | Path]: Đường dẫn file hoặc thư mục media nguồn.
    - target [str | Path]: Đường dẫn file hoặc thư mục media đích.
    - overwrite [bool]: Ghi đè nếu mục tiêu đã tồn tại. Mặc định: False.

    Đầu ra:
    - [Path]: Đường dẫn vị trí mới sau khi copy.

    Ví dụ:
    >>> import klygo.media as media
    >>> media.copy("image.jpg", "backup/image.jpg", overwrite=True)
    """
    validate_type(source, (str, Path), "source")
    validate_type(target, (str, Path), "target")
    validate_type(overwrite, bool, "overwrite")

    src_p = Path(source)
    if not src_p.exists():
        raise FileNotFoundError(f"Media source does not exist: {src_p}")

    return _files_copy(src_p, target, overwrite=overwrite)


def save_video(
    output_path: Union[str, Path],
    frames: Iterable[Union[Image.Image, np.ndarray]],
    fps: float = 30.0,
    fourcc: str = "mp4v",
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Tác dụng:
    - Lưu danh sách hoặc Generator các khung hình (frames) thành tập tin video (.mp4, .avi, .mkv...).

    Đầu vào:
    - output_path [str | Path]: Đường dẫn file video đầu ra.
    - frames [Iterable]: Danh sách hoặc Generator các ảnh/frames.
    - fps [float]: Số khung hình trên giây. Mặc định: 30.0.
    - fourcc [str]: Mã codec video OpenCV (vd: 'mp4v', 'xvid'). Mặc định: 'mp4v'.
    - overwrite [bool]: Ghi đè file nếu đã tồn tại. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình ProgressBar khi đóng gói video. Mặc định: True.

    Đầu ra:
    - [Path]: Đường dẫn file video đã lưu.

    Ví dụ:
    >>> import klygo.media as media
    >>> media.save_video("output.mp4", frames_list, fps=30, overwrite=True)
    """
    validate_type(output_path, (str, Path), "output_path")
    validate_type(overwrite, bool, "overwrite")
    validate_type(verbose, bool, "verbose")

    p = Path(output_path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"Video file already exists: {p}. Use overwrite=True to replace it.")

    p.parent.mkdir(parents=True, exist_ok=True)

    frame_iter = iter(frames)
    try:
        first_frame = next(frame_iter)
    except StopIteration:
        raise ValueError("frames iterable is empty")

    arr = to_array(first_frame)
    height, width = arr.shape[:2]

    fourcc_code = cv.VideoWriter_fourcc(*fourcc)
    writer = cv.VideoWriter(str(p), fourcc_code, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {p} with fourcc {fourcc!r}")

    def _write_frame(f):
        f_arr = to_array(f)
        if isinstance(f, (Image.Image, LazyImage)) or (f_arr.ndim == 3 and f_arr.shape[2] == 3):
            f_bgr = cv.cvtColor(f_arr, cv.COLOR_RGB2BGR)
        else:
            f_bgr = f_arr
        writer.write(f_bgr)

    try:
        with ProgressBar(total=None, desc=f"Encoding video {p.name}", unit="frame", verbose=verbose, colour="cyan") as pbar:
            _write_frame(first_frame)
            pbar.update(1)
            for frame in frame_iter:
                _write_frame(frame)
                pbar.update(1)
    finally:
        writer.release()

    return p


def save_images(
    output_dir: Union[str, Path],
    images: Iterable[Union[Image.Image, np.ndarray]],
    prefix: str = "frame",
    extension: str = ".jpg",
    overwrite: bool = False,
    verbose: bool = True,
) -> List[Path]:
    """
    Tác dụng:
    - Lưu chuỗi ảnh/frames vào một thư mục với tên tăng dần (vd: frame_000001.jpg).

    Đầu vào:
    - output_dir [str | Path]: Thư mục xuất các file ảnh.
    - images [Iterable]: Danh sách ảnh hoặc frames.
    - prefix [str]: Tiền tố tên file. Mặc định: 'frame'.
    - extension [str]: Đuôi file ảnh (.jpg, .png, .webp...). Mặc định: '.jpg'.
    - overwrite [bool]: Ghi đè nếu file ảnh đã tồn tại. Mặc định: False.
    - verbose [bool]: Hiển thị thanh tiến trình ProgressBar khi lưu. Mặc định: True.

    Đầu ra:
    - [List[Path]]: Danh sách đường dẫn tới từng file ảnh đã được lưu.

    Ví dụ:
    >>> import klygo.media as media
    >>> media.save_images("extracted_frames", frames_list, extension=".jpg")
    """
    validate_type(output_dir, (str, Path), "output_dir")
    validate_type(overwrite, bool, "overwrite")
    validate_type(verbose, bool, "verbose")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if not extension.startswith("."):
        extension = f".{extension}"

    saved_paths: List[Path] = []
    img_list = builtins.list(images) if not isinstance(images, (builtins.list, tuple)) else images
    with ProgressBar(total=len(img_list), desc=f"Saving image batch to {out_p.name}", unit="file", verbose=verbose, colour="cyan") as pbar:
        for idx, img in enumerate(img_list, start=1):
            file_path = out_p / f"{prefix}_{idx:06d}{extension}"
            save(file_path, img, overwrite=overwrite, verbose=False)
            saved_paths.append(file_path)
            pbar.update(1)

    return saved_paths


def iter_frames(
    source: Union[str, Path],
    sample_rate: int = 1,
    recursive: bool = False,
    backend: str = "pil",
    verbose: bool = False,
) -> Generator[Union[Image.Image, np.ndarray], None, None]:
    """
    Tác dụng:
    - Generator duyệt từng khung hình (frame) từ file video hoặc thư mục ảnh với tham số bước nhảy (sample_rate).

    Đầu vào:
    - source [str | Path]: Đường dẫn file video hoặc thư mục ảnh.
    - sample_rate [int]: Bước nhảy duyệt (vd: 1 = duyệt từng frame, 5 = lấy 1 frame mỗi 5 frame). Mặc định: 1.
    - recursive [bool]: Duyệt đệ quy (nếu source là thư mục). Mặc định: False.
    - backend [str]: 'pil' (mặc định) hoặc 'opencv'.
    - verbose [bool]: Hiển thị thanh tiến trình. Mặc định: False.

    Đầu ra:
    - [Generator]: Generator phát ra từng khung hình dạng PIL Image hoặc NumPy array.

    Ví dụ:
    >>> import klygo.media as media
    >>> for frame in media.iter_frames("video.mp4", sample_rate=5):
    ...     process(frame)
    """
    if sample_rate < 1:
        raise ValueError("sample_rate must be an integer >= 1")

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"source does not exist: {p}")

    backend = backend.lower()
    if backend not in ("pil", "opencv"):
        raise ValueError("backend must be 'pil' or 'opencv'")

    if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
        cap = cv.VideoCapture(str(p))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {p}")

        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        try:
            with ProgressBar(total=total_frames if total_frames > 0 else None, desc=f"Iterating video {p.name}", unit="frame", verbose=verbose, colour="cyan") as pbar:
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % sample_rate == 0:
                        if backend == "pil":
                            yield Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                        else:
                            yield frame
                    frame_idx += 1
                    pbar.update(1)
        finally:
            cap.release()
    else:
        imgs = load(source, recursive=recursive, stream=False, backend=backend, verbose=verbose)
        for idx, img in enumerate(imgs):
            if idx % sample_rate == 0:
                yield img


def probe(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Tác dụng:
    - Trích xuất thông tin metadata chi tiết của một tập tin ảnh hoặc video (soi định dạng).

    Đầu vào:
    - path [str | Path]: Đường dẫn file ảnh hoặc file video.

    Đầu ra:
    - [Dict[str, Any]]: Dictionary chứa metadata (name, path, format, width, height, size, fps...).

    Ví dụ:
    >>> import klygo.media as media
    >>> v_info = media.probe("video.mp4")
    >>> print(v_info['fps'], v_info['frame_count'])
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    if not p.is_file():
        raise ValueError(f"Path must be a file: {p}")

    suf = p.suffix.lower()

    if suf in VIDEO_SUFFIXES:
        cap = cv.VideoCapture(str(p))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {p}")
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv.CAP_PROP_FPS))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0

        fourcc_int = int(cap.get(cv.CAP_PROP_FOURCC))
        codec_raw = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]).strip().lower()
        cap.release()

        # Map to friendly codec names
        codec_map = {
            "avc1": "h264",
            "h264": "h264",
            "x264": "h264",
            "hev1": "h265",
            "hvc1": "h265",
            "hevc": "h265",
            "x265": "h265",
            "mp4v": "mp4v",
            "mjpg": "mjpeg",
            "vp09": "vp9",
            "vp08": "vp8",
            "av01": "av1",
        }
        codec = codec_map.get(codec_raw, codec_raw if codec_raw else "unknown")
        web_ready = codec in ("h264", "vp8", "vp9", "av1") and suf in (".mp4", ".webm")

        return {
            "name": p.name,
            "path": p,
            "type": "video",
            "format": suf.lstrip("."),
            "codec": codec,
            "web_ready": web_ready,
            "width": width,
            "height": height,
            "size": (width, height),
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": round(duration, 2),
        }

    with Image.open(p) as img:
        width, height = img.size
        return {
            "name": p.name,
            "path": p,
            "type": "image",
            "format": img.format,
            "mode": img.mode,
            "width": width,
            "height": height,
            "size": (width, height),
        }


# =========================================================================
# 2. Format Conversions (to_array, to_tensor, to_pil)
# =========================================================================

def to_array(image: Any) -> np.ndarray:
    """
    Tác dụng:
    - Chuyển đổi linh hoạt hình ảnh từ PIL Image, PyTorch Tensor, NumPy array hoặc LazyImage sang mảng NumPy ndarray (dạng [H, W, C]).

    Đầu vào:
    - image [Image.Image | torch.Tensor | np.ndarray | LazyImage]: Đối tượng dữ liệu ảnh.

    Đầu ra:
    - [np.ndarray]: Mảng NumPy ndarray.

    Ví dụ:
    >>> import klygo.media as media
    >>> arr = media.to_array(pil_img)
    """
    if isinstance(image, LazyImage):
        return image.to_array()

    if isinstance(image, np.ndarray):
        return image.copy()

    if isinstance(image, Image.Image):
        return np.array(image)

    if _HAS_TORCH and isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.ndim == 4:
            t = t.squeeze(0)
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):
            t = t.permute(1, 2, 0)
        arr = t.numpy()
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return arr

    raise TypeError(f"Unsupported image type for to_array: {type(image)}")


def to_tensor(image: Any, normalize: bool = True) -> Any:
    """
    Tác dụng:
    - Chuyển đổi hình ảnh (PIL Image, NumPy array hoặc LazyImage) sang PyTorch Tensor dạng chuẩn mô hình AI [C, H, W].

    Đầu vào:
    - image [Image.Image | np.ndarray | torch.Tensor | LazyImage]: Dữ liệu ảnh.
    - normalize [bool]: Tự động chuẩn hóa giá trị về dải 0.0 - 1.0 (float32). Mặc định: True.

    Đầu ra:
    - [torch.Tensor]: Đối tượng PyTorch Tensor.

    Ví dụ:
    >>> import klygo.media as media
    >>> tensor = media.to_tensor(pil_img)
    """
    if isinstance(image, LazyImage):
        image = image.to_pil()

    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is not installed in current environment.")

    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone()
    elif isinstance(image, Image.Image):
        arr = np.array(image)
        tensor = torch.from_numpy(arr)
        if arr.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        tensor = tensor.permute(2, 0, 1)
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
    else:
        raise TypeError(f"Unsupported image type for to_tensor: {type(image)}")

    if normalize and tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32) / 255.0

    return tensor


def to_pil(image: Any) -> Image.Image:
    """
    Tác dụng:
    - Chuyển đổi mảng NumPy ndarray, PyTorch Tensor hoặc LazyImage sang đối tượng PIL Image.

    Đầu vào:
    - image [np.ndarray | torch.Tensor | Image.Image | LazyImage]: Dữ liệu ảnh.

    Đầu ra:
    - [Image.Image]: Đối tượng PIL Image.

    Ví dụ:
    >>> import klygo.media as media
    >>> pil_img = media.to_pil(np_array)
    """
    if isinstance(image, LazyImage):
        return image.to_pil()

    if isinstance(image, Image.Image):
        return image.copy()

    arr = to_array(image)
    if arr.ndim == 2:
        return Image.fromarray(arr)
    elif arr.ndim == 3:
        if arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")
        elif arr.shape[2] == 4:
            return Image.fromarray(arr, mode="RGBA")
        elif arr.shape[2] == 1:
            return Image.fromarray(arr.squeeze(2))

    raise ValueError(f"Unsupported array shape for to_pil: {arr.shape}")
