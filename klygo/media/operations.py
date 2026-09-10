import builtins
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional, Generator, Iterable, Callable

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
# 0A. VideoReader: Smart Sequential & Random Access Video Seeker
# =========================================================================

class VideoReader:
    """
    Trình quản lý đọc video thông minh, tối ưu tốc độ đọc tuần tự và nhảy frame.
    Duy trì kết nối cv.VideoCapture duy nhất, tự động theo dõi con trỏ vị trí frame `_current_pos`.
    - Đọc tuần tự (frame sau = frame trước + 1): Gọi trực tiếp `cap.read()` đạt tốc độ gốc OpenCV
      (300-500+ FPS), hoàn toàn loại bỏ độ trễ seek của `cap.set()`.
    - Truy cập ngẫu nhiên (vd: frames[500] hoặc frames[-1]): Tự động gọi `cap.set(CAP_PROP_POS_FRAMES, index)`
      rồi tiếp tục đọc tuần tự từ vị trí đó.
    - Zero-RAM: Khởi tạo tức thì không decode bất kỳ frame pixel nào vào bộ nhớ RAM.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Could not find video file: {self.path}")

        # Thăm dò nhanh metadata của video 1 lần duy nhất lúc mở
        cap = cv.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {self.path}")

        self.total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(cap.get(cv.CAP_PROP_FPS) or 30.0)
        self.width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        self._cap: Optional[cv.VideoCapture] = None
        self._current_pos: int = 0

    @property
    def cap(self) -> cv.VideoCapture:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv.VideoCapture(str(self.path))
            self._current_pos = 0
        return self._cap

    def read_frame(self, index: int) -> Optional[np.ndarray]:
        """
        Đọc frame pixel tại vị trí `index` với cơ chế Seeker thông minh.
        """
        if index < 0 or (self.total_frames > 0 and index >= self.total_frames):
            return None

        c = self.cap
        if self._current_pos != index:
            c.set(cv.CAP_PROP_POS_FRAMES, index)
            self._current_pos = index

        ret, frame = c.read()
        if ret:
            self._current_pos += 1
            return frame

        # Thử seek lại một lần nếu luồng bị mất đồng bộ
        c.set(cv.CAP_PROP_POS_FRAMES, index)
        ret, frame = c.read()
        if ret:
            self._current_pos = index + 1
            return frame

        return None

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def close(self) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()


# =========================================================================
# 0B. LazyImage: Zero-RAM URL/Path/Frame Proxy (PIL + NumPy Hybrid)
# =========================================================================

class LazyImage(Image.Image):
    """
    Đại diện cho một bức ảnh hoặc frame video theo cơ chế nạp lười (Zero-RAM URL/Path Proxy).
    Lưu trữ đường dẫn/URL file ảnh hoặc vị trí frame video với chi phí RAM = 0.
    Hỗ trợ Zero-RAM Lazy Crop mặc định: Cắt vùng ảnh con chỉ bằng cách lưu (URL/Frame, bounding_box),
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
        frame_index: Optional[int] = None,
        reader: Optional[VideoReader] = None,
    ) -> None:
        self._im = None
        self._image: Optional[Union[Image.Image, np.ndarray]] = None
        self._cached_size: Optional[Tuple[int, int]] = None
        self._cached_mode: str = "RGB"
        self._instructions: List[Tuple[str, tuple, dict]] = []

        if isinstance(path, LazyImage):
            self._path = path.path
            self._url = path.url
            self.backend = path.backend
            self.frame_index = path.frame_index if frame_index is None else frame_index
            self.reader = path.reader if reader is None else reader
            self._instructions = list(path._instructions)
            self._cached_mode = path._cached_mode
            self._cached_size = path._cached_size
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
            self.frame_index = frame_index
            self.reader = reader
            self._crop_box = tuple(int(x) for x in crop_box) if crop_box is not None else None

    @property
    def path(self) -> Path:
        """Đường dẫn tệp ảnh hoặc video nguồn."""
        return self._path

    @path.setter
    def path(self, new_path: Union[str, Path]) -> None:
        self._path = Path(new_path).resolve()
        self._url = str(self._path)
        self._image = None
        self._cached_size = None

    @property
    def url(self) -> str:
        """URL hoặc chuỗi đường dẫn ảnh/video nguồn."""
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

    def load(self, cache: bool = True) -> Union[Image.Image, np.ndarray]:
        """Thực sự nạp ảnh hoặc frame video từ đĩa vào bộ nhớ RAM."""
        if self._image is not None:
            return self._image
        
        result = None
        if self.frame_index is not None:
            # Video frame decoding via reader
            if self.reader is None:
                self.reader = VideoReader(self.path)
            frame = self.reader.read_frame(self.frame_index)
            if frame is None:
                raise IndexError(f"Could not read frame {self.frame_index} from video {self.path}")

            if self.crop_box is not None:
                x1, y1, x2, y2 = self.crop_box
                h_max, w_max = frame.shape[:2]
                frame = frame[max(0, y1):min(h_max, y2), max(0, x1):min(w_max, x2)]

            if self.backend == "opencv":
                result = frame
            else:
                result = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        elif self.backend == "opencv":
            arr = cv.imread(str(self.path), cv.IMREAD_COLOR)
            if arr is None:
                raise FileNotFoundError(f"Could not read image file: {self.path}")
            if self.crop_box is not None:
                x1, y1, x2, y2 = self.crop_box
                arr = arr[max(0, y1):min(arr.shape[0], y2), max(0, x1):min(arr.shape[1], x2)]
            result = arr
        else:
            with Image.open(self.path) as source_image:
                source_image = source_image.convert("RGB")
                if self.crop_box is not None:
                    source_image = source_image.crop(self.crop_box)
                result = source_image
                
        if cache:
            self._image = result
            if isinstance(result, np.ndarray):
                self._cached_size = (result.shape[1], result.shape[0])
            else:
                self._cached_size = result.size
        return result

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
            self.frame_index = value.frame_index
            self.reader = value.reader
            self._cached_size = value.size
            self.crop_box = value.crop_box
        elif isinstance(value, Image.Image):
            self._image = value
            self._cached_size = value.size
            self._crop_box = None
        elif isinstance(value, np.ndarray):
            self._image = value
            self._cached_size = (value.shape[1], value.shape[0])
            self._crop_box = None
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
        """Kích thước (width, height) theo chuẩn PIL Image (Zero-RAM calculation)."""
        if self._cached_size is not None:
            return self._cached_size
        if self.crop_box is not None:
            return (max(0, self.crop_box[2] - self.crop_box[0]), max(0, self.crop_box[3] - self.crop_box[1]))
        if self._image is not None:
            if isinstance(self._image, Image.Image):
                self._cached_size = self._image.size
            else:
                self._cached_size = (self._image.shape[1], self._image.shape[0])
            return self._cached_size
        if self.frame_index is not None:
            if self.reader is not None:
                self._cached_size = (self.reader.width, self.reader.height)
                return self._cached_size
            try:
                self.reader = VideoReader(self.path)
                self._cached_size = (self.reader.width, self.reader.height)
                return self._cached_size
            except Exception:
                return (0, 0)
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
    def shape(self) -> Tuple[int, ...]:
        """Kích thước (height, width, channels) theo chuẩn NumPy ndarray."""
        if self.mode in ("L", "1", "I", "F"):
            return (self.height, self.width)
        elif self.mode == "RGBA":
            return (self.height, self.width, 4)
        return (self.height, self.width, 3)

    @property
    def dtype(self) -> np.dtype:
        """Kiểu dữ liệu chuẩn NumPy (uint8)."""
        return np.dtype("uint8")

    @property
    def mode(self) -> str:
        return self._cached_mode

    def to_pil(self, cache: bool = False) -> Image.Image:
        """Chuyển thành PIL Image thật (thực thi Lazy Pipeline)."""
        loaded = self.load(cache=cache)
        if isinstance(loaded, Image.Image):
            img = loaded
        else:
            img = Image.fromarray(cv.cvtColor(loaded, cv.COLOR_BGR2RGB))
            
        for op, args, kwargs in self._instructions:
            img = getattr(img, op)(*args, **kwargs)
            
        return img

    def to_array(self, cache: bool = False) -> np.ndarray:
        """Chuyển thành NumPy array thật."""
        if len(self._instructions) > 0:
            return np.array(self.to_pil(cache=cache))
            
        loaded = self.load(cache=cache)
        if isinstance(loaded, np.ndarray):
            return loaded
        return np.array(loaded)

    def lazy_crop(self, box: Tuple[int, int, int, int]) -> "LazyImage":
        """
        Zero-RAM Lazy Cropping: Tạo ảnh con từ vùng cắt mà KHÔNG nạp pixel vào RAM.
        Chỉ lưu tọa độ bounding box và đường dẫn file ảnh gốc/frame index.
        """
        return LazyImage(
            self,
            backend=self.backend,
            crop_box=box,
            frame_index=self.frame_index,
            reader=self.reader,
        )

    def crop(self, *args, lazy: Optional[bool] = None, **kwargs) -> Any:
        """
        Cắt ảnh. Mặc định hoạt động theo cơ chế Zero-RAM Lazy Crop:
        Trả về một LazyImage mới giữ nguyên URL/path/frame_index và tính toán size tức thì mà không nạp RAM.
        Nếu truyền lazy=False rõ ràng, trả về đối tượng PIL Image thật.
        """
        if lazy is False:
            return self.to_pil().crop(*args, **kwargs)

        box = None
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            box = tuple(int(x) for x in args[0])
        elif len(args) == 4:
            box = tuple(int(x) for x in args)
        elif "box" in kwargs:
            box = tuple(int(x) for x in kwargs["box"])

        if box is not None and len(box) == 4:
            return self.lazy_crop(box)

        return self.to_pil().crop(*args, **kwargs)

    def __array__(self, dtype=None) -> np.ndarray:
        """Hỗ trợ tự động chuyển đổi khi gọi np.array(lazy_img)."""
        arr = self.to_array()
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    @property
    def __array_interface__(self):
        return self.to_pil(cache=False).__array_interface__

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
        frame_info = f" frame={self.frame_index}" if self.frame_index is not None else ""
        return f"<LazyImage [{status}] url='{self.path.name}'{frame_info}{crop_info} size=({w}, {h})>"


# =========================================================================
# 0C. MediaFrames: Unified Zero-RAM List-like Media Container
# =========================================================================

class MediaFrames(list):
    """
    Tập hợp các khung hình media (ảnh / video frames) trả về từ `klygo.media.load`.
    Kế thừa trực tiếp từ `list` của Python, hoạt động 100% như danh sách chuẩn:
    - Zero-RAM: Toàn bộ frames là LazyImage, chi phí RAM = 0 khi load (dù 100 hay 100,000 frames).
    - Indexing tự do: frames[0], frames[-1], frames[100]...
    - Slicing mượt mà: frames[10:50], frames[::2] trả về MediaFrames con với Zero-RAM.
    - List mutations: append, extend, insert, pop, __setitem__, __delitem__...
    - Fast sequential decoding: Đọc tuần tự với tốc độ tối đa của OpenCV (300-500+ FPS).
    """

    def __init__(
        self,
        items: Optional[Iterable] = None,
        *,
        stream: bool = False,
        total_frames: Optional[int] = None,
        fps: float = 30.0,
        source_path: Optional[Union[str, Path]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        source_type: str = "video",
        reader: Optional[VideoReader] = None,
        cap: Optional[Any] = None,
    ) -> None:
        super().__init__(items if items is not None else [])
        self.is_stream = bool(stream)
        self.fps = float(fps) if fps else 30.0
        self.source_path = str(source_path) if source_path is not None else None
        self.width = width
        self.height = height
        self.source_type = source_type
        self.total_frames = total_frames if total_frames is not None else super().__len__()
        self.reader = reader
        self._cap = cap

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, index: Union[int, slice]) -> Any:
        if isinstance(index, slice):
            sub_items = super().__getitem__(index)
            step = index.step or 1
            new_fps = self.fps / abs(step) if abs(step) > 1 else self.fps
            return MediaFrames(
                sub_items,
                stream=self.is_stream,
                total_frames=len(sub_items),
                fps=new_fps,
                source_path=self.source_path,
                width=self.width,
                height=self.height,
                source_type=self.source_type,
                reader=self.reader,
                cap=self._cap,
            )
        return super().__getitem__(index)

    def map(self, func: Callable[[Any], Any]) -> "MediaFrames":
        """
        Áp dụng hàm tiền xử lý (crop, resize, normalize, đổi màu...) lên từng frame.
        """
        mapped = [func(f) for f in self]
        return MediaFrames(
            mapped,
            stream=self.is_stream,
            total_frames=len(mapped),
            fps=self.fps,
            source_path=self.source_path,
            width=self.width,
            height=self.height,
            source_type=self.source_type,
            reader=self.reader,
            cap=self._cap,
        )

    def filter(self, func: Callable[[Any], bool]) -> "MediaFrames":
        """
        Lọc loại bỏ các frame không thoả mãn điều kiện func(frame) == True.
        """
        filtered = [f for f in self if func(f)]
        return MediaFrames(
            filtered,
            stream=self.is_stream,
            total_frames=len(filtered),
            fps=self.fps,
            source_path=self.source_path,
            width=self.width,
            height=self.height,
            source_type=self.source_type,
            reader=self.reader,
            cap=self._cap,
        )

    def append(self, item: Any) -> None:
        super().append(item)
        self.total_frames = super().__len__()

    def extend(self, other: Iterable[Any]) -> None:
        super().extend(other)
        self.total_frames = super().__len__()

    def insert(self, index: int, item: Any) -> None:
        super().insert(index, item)
        self.total_frames = super().__len__()

    def pop(self, index: int = -1) -> Any:
        res = super().pop(index)
        self.total_frames = super().__len__()
        return res

    def to_list(self) -> List[Any]:
        """Chuyển đổi thành standard list."""
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
        if self.reader is not None:
            try:
                self.reader.release()
            except Exception:
                pass
            self.reader = None
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
        return f"<MediaFrames: {len(self)} frames, type='{self.source_type}', fps={self.fps}>"



# =========================================================================
# 1. Media Load / Save / Convert / Copy / Info
# =========================================================================

def _read_video_frames(
    path: Path,
    stream: bool = False,
    backend: str = "pil",
    verbose: bool = False,
) -> MediaFrames:
    reader = VideoReader(path)
    total_frames = reader.total_frames
    fps = reader.fps
    width = reader.width
    height = reader.height

    lazy_frames = [
        LazyImage(
            path,
            backend=backend,
            frame_index=i,
            reader=reader,
        )
        for i in range(total_frames)
    ]

    return MediaFrames(
        lazy_frames,
        stream=stream,
        total_frames=total_frames,
        fps=fps,
        source_path=path,
        width=width,
        height=height,
        source_type="video",
        reader=reader,
    )


def load(
    source: Union[str, Path],
    recursive: bool = False,
    stream: bool = False,
    backend: str = "pil",
    verbose: bool = False,
) -> MediaFrames:
    """
    Tác dụng:
    - Đọc 1 file ảnh, file video, hoặc toàn bộ thư mục chứa ảnh.
    - Luôn trả về đối tượng `MediaFrames` (kế thừa list) chuẩn Zero-RAM.

    Định dạng tương thích:
    - Ảnh: .png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff
    - Video: .mp4, .avi, .mov, .mkv, .m4v, .webm

    Đầu vào:
    - source [str | Path]: Đường dẫn file ảnh, file video hoặc thư mục chứa ảnh.
    - recursive [bool]: Duyệt đệ quy qua các thư mục con (khi source là thư mục). Mặc định: False.
    - stream [bool]: Tham số giữ tương thích ngược (toàn bộ khung hình đã là Zero-RAM LazyImage). Mặc định: False.
    - backend [str]: 'pil' (mặc định) hoặc 'opencv'.
    - verbose [bool]: Hiển thị thanh tiến trình khi đọc. Mặc định: False.

    Đầu ra:
    - [MediaFrames]: Danh sách khung hình kế thừa Python list, Zero-RAM, hỗ trợ indexing [0], [-1], slicing [10:20], len(frames).

    Ví dụ:
    >>> import klygo.media as media
    >>> imgs = media.load("image.jpg")
    >>> frames = media.load("video.mp4")
    >>> len(frames)        # Tổng số frame của video tức thì (Zero-RAM)!
    >>> first = frames[0]  # Lấy frame đầu tiên, f.size, f.crop(...) hoàn toàn Zero-RAM!
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

    return MediaFrames(
        lazy_frames,
        stream=stream,
        total_frames=len(lazy_frames),
        source_path=p,
        source_type="folder" if p.is_dir() else "image",
    )


def save(
    path: Union[str, Path],
    image: Union[Image.Image, np.ndarray],
    overwrite: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Tác dụng:
    - Lưu một đối tượng ảnh (PIL Image, NumPy array, PyTorch Tensor) ra tập tin đĩa.
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
    gpu: Optional[bool] = None,
    preset: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Tác dụng:
    - Chuyển đổi định dạng file ảnh (.png -> .jpg) hoặc file video (.avi -> .mp4, .mkv -> .webm...).
    - Tự động nhận diện nhóm chuyển đổi Web-ready, Storage, Modern Web, hoặc Demo.
    - Tự động phát hiện và tận dụng GPU (NVIDIA NVENC) nếu máy/Colab có sẵn (`gpu=None`), hoặc chỉ định rõ `gpu=True`/`gpu=False`.
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
        with ProgressBar(total=1, desc=f"Converting image {src_p.name} -> {tgt_p.name}", unit="file", verbose=verbose, colour="cyan") as pbar:
            imgs = load(src_p, verbose=False)
            res = save(tgt_p, imgs[0], overwrite=overwrite, verbose=False)
            pbar.update(1)
            return res
    elif src_suf in VIDEO_SUFFIXES and tgt_suf in VIDEO_SUFFIXES:
        import shutil
        import subprocess

        # Tự động nhận diện GPU nếu gpu=None
        use_gpu = gpu
        if use_gpu is None:
            if _HAS_TORCH and torch.cuda.is_available():
                use_gpu = True
            elif shutil.which("nvidia-smi"):
                use_gpu = True
            else:
                use_gpu = False

        fourcc = "mp4v"
        ffmpeg_vcodec = "h264_nvenc" if use_gpu else "libx264"
        default_preset = "p4" if use_gpu else "fast"
        active_preset = preset or default_preset

        if codec:
            codec_lower = codec.lower()
            if codec_lower in ("h264", "avc1"):
                fourcc = "avc1"
                ffmpeg_vcodec = "h264_nvenc" if use_gpu else "libx264"
            elif codec_lower in ("h265", "hevc"):
                fourcc = "hevc"
                ffmpeg_vcodec = "hevc_nvenc" if use_gpu else "libx265"
            elif codec_lower in ("vp9", "webm"):
                fourcc, ffmpeg_vcodec = "vp09", "libvpx-vp9"
            elif codec_lower == "mjpeg":
                fourcc, ffmpeg_vcodec = "MJPG", "mjpeg"
            else:
                fourcc, ffmpeg_vcodec = codec, codec
        else:
            if tgt_suf == ".mp4":
                fourcc = "avc1"
                ffmpeg_vcodec = "h264_nvenc" if use_gpu else "libx264"
            elif tgt_suf == ".webm":
                fourcc, ffmpeg_vcodec = "vp09", "libvpx-vp9"
            elif tgt_suf == ".avi":
                fourcc, ffmpeg_vcodec = "MJPG", "mjpeg"

        # 1. Try FFmpeg first (preserves audio, handles AV1, much faster)
        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg", "-y" if overwrite else "-n",
                "-threads", "0",
                "-err_detect", "ignore_err",
                "-i", str(src_p),
                "-c:v", ffmpeg_vcodec,
                "-pix_fmt", "yuv420p",
            ]
            if not gpu:
                cmd.extend(["-crf", str(crf)])
            else:
                # NVENC uses -cq (constant quality) or -preset
                cmd.extend(["-cq", str(crf)])

            if active_preset:
                cmd.extend(["-preset", str(active_preset)])

            # Keep audio intact without re-encoding
            cmd.extend(["-c:a", "copy"])

            if fps:
                cmd.extend(["-r", str(fps)])
            cmd.append(str(tgt_p))
            
            try:
                v_info = probe(src_p)
                total_frames = v_info.get("frame_count") or 1
                with ProgressBar(total=total_frames, desc=f"Converting {src_p.name} (FFmpeg)", unit="frame", verbose=verbose, colour="cyan") as pbar:
                    # Run FFmpeg with progress monitoring if verbose, or silent if not
                    if not verbose:
                        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        # Monitor FFmpeg progress output
                        proc_cmd = cmd[:-1] + ["-progress", "pipe:1", str(tgt_p)]
                        proc = subprocess.Popen(proc_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                        last_frame = 0
                        for line in proc.stdout:
                            if line.startswith("frame="):
                                try:
                                    cur_frame = int(line.split("=")[1].strip())
                                    delta = cur_frame - last_frame
                                    if delta > 0:
                                        pbar.update(delta)
                                        last_frame = cur_frame
                                except Exception:
                                    pass
                        proc.wait()
                        result = subprocess.CompletedProcess(cmd, proc.returncode)

                    if result.returncode == 0:
                        if verbose and last_frame < total_frames:
                            pbar.update(total_frames - last_frame)
                        return tgt_p
                    elif use_gpu and gpu is None:
                        # Fallback to CPU FFmpeg if GPU NVENC was auto-detected but failed (e.g. ffmpeg build lacks nvenc)
                        cpu_cmd = [
                            "ffmpeg", "-y" if overwrite else "-n",
                            "-threads", "0",
                            "-err_detect", "ignore_err",
                            "-i", str(src_p),
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-crf", str(crf),
                            "-preset", preset or "fast",
                            "-c:a", "copy",
                        ]
                        if fps:
                            cpu_cmd.extend(["-r", str(fps)])
                        cpu_cmd.append(str(tgt_p))
                        cpu_res = subprocess.run(cpu_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        if cpu_res.returncode == 0:
                            return tgt_p
            except Exception:
                pass

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
    verbose: bool = False,
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
    verbose: bool = False,
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

# =========================================================================
# Lazy Pipeline Auto-Binding (Dynamic Proxy for PIL.Image.Image)
# =========================================================================
LAZY_OPS = {"resize", "convert", "rotate", "transpose", "filter", "point", "quantize", "transform"}

for attr in dir(Image.Image):
    if attr.startswith("_") or not callable(getattr(Image.Image, attr)): 
        continue
    if attr in ["crop", "show", "save", "load", "copy", "thumbnail", "tobytes"]: 
        continue

    def _make_forwarder(name):
        def forwarder(self, *args, **kwargs):
            if name in LAZY_OPS:
                # Trả về một LazyImage mới, ghi nhận chỉ thị để không tốn RAM
                new_lazy = LazyImage(self)
                new_lazy._instructions = list(self._instructions) + [(name, args, kwargs)]
                
                # Cập nhật size và mode tức thì
                if name == "resize":
                    new_lazy._cached_size = kwargs.get("size", args[0] if args else self._cached_size)
                elif name == "convert":
                    new_lazy._cached_mode = kwargs.get("mode", args[0] if args else self._cached_mode)
                return new_lazy
            else:
                # Hàm thực thi cuối (ví dụ: getpixel, histogram), tự động nhả RAM sau khi chạy
                return getattr(self.to_pil(cache=False), name)(*args, **kwargs)
        return forwarder

    setattr(LazyImage, attr, _make_forwarder(attr))

# Override specific terminal methods for consistency
def _lazy_show(self, *args, **kwargs):
    return self.to_pil(cache=False).show(*args, **kwargs)

def _lazy_save(self, *args, **kwargs):
    return self.to_pil(cache=False).save(*args, **kwargs)
    
def _lazy_copy(self):
    return LazyImage(self)

setattr(LazyImage, "show", _lazy_show)
setattr(LazyImage, "save", _lazy_save)
setattr(LazyImage, "copy", _lazy_copy)

