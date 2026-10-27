import builtins
import importlib
import math
import os
import uuid
import weakref
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional, Generator, Iterable, Callable, Iterator

import cv2 as cv
import numpy as np
from PIL import Image

from klygo.utils.progress import ProgressBar
from klygo.validators import validate_type, validate_count
from klygo import files

__all__ = [
    "load", "save", "convert", "copy", "save_video", "save_images",
    "iter_frames", "probe", "to_array", "to_tensor", "to_pil",
    "MediaFrames", "LazyImage", "VideoReader",
    "compose",
]


def _optional_torch(required: bool = False):
    """Load PyTorch only for media operations that explicitly need tensors."""
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        if required:
            raise ImportError(
                "PyTorch tensor conversion requires the 'torch' extra. "
                "Install it using 'pip install \"klygo[torch]\"'."
            ) from exc
        return None

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


def _operation_size(operation: Any, size: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Return lazy output dimensions for a known preprocessing operation."""
    if size is None or not getattr(operation, "changes_geometry", False):
        return size
    width, height = size
    params = operation.params
    if operation.name == "crop":
        left, top, right, bottom = params["box"]
        return max(0, right - left), max(0, bottom - top)
    if operation.name in ("resize", "thumbnail", "contain") and params.get("keep_ratio", operation.name != "resize"):
        target_width, target_height = params["size"]
        scale = min(target_width / max(1, width), target_height / max(1, height))
        return max(1, round(width * scale)), max(1, round(height * scale))
    if operation.name in ("resize", "fit", "cover", "letterbox", "center_crop"):
        return tuple(params["size"])
    if operation.name == "pad":
        padding = params["padding"]
        if isinstance(padding, int):
            left = top = right = bottom = padding
        elif len(padding) == 2:
            left = right = padding[0]
            top = bottom = padding[1]
        else:
            left, top, right, bottom = padding
        return width + left + right, height + top + bottom
    if operation.name in ("affine", "perspective", "warp") and params.get("size"):
        return tuple(params["size"])
    if operation.name == "transpose" and params.get("method") in (
        "rotate_90", "rotate_270", "transpose", "transverse",
    ):
        return height, width
    return None


# =========================================================================
# 0A. VideoReader: Smart Sequential & Random Access Video Seeker
# =========================================================================

class VideoReader:
    """Read video frames through one reusable OpenCV capture.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing video path.

    Returns
    -------
    VideoReader
        Open reader; call close() to release the capture.

    Raises
    ------
    FileNotFoundError
        The video is missing or unreadable."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = files.resolve(path)
        if not files.exists(self.path):
            raise FileNotFoundError(f"Could not find video file: {self.path}")

        # Probe video metadata once during initialization
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
        Read frame pixels at `index` using accurate frame seeking.
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

        # Retry seek once if stream synchronization is lost
        c.set(cv.CAP_PROP_POS_FRAMES, index)
        ret, frame = c.read()
        if ret:
            self._current_pos = index + 1
            return frame

        return None

    def release(self) -> None:
        if getattr(self, "_cap", None) is not None:
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
    """Represent source pixels without decoding them until requested.

    Parameters
    ----------
    source : str, pathlib.Path, or LazyImage
        Source file or another lazy frame reference.

    Returns
    -------
    LazyImage
        Lazy image proxy with conversion and crop methods."""

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
        self._processed_cache: Optional[Any] = None
        self._cached_size: Optional[Tuple[int, int]] = None
        self._cached_mode: str = "RGB"
        self._instructions: List[Tuple[str, tuple, dict]] = []
        self._preprocessing_ops: List[Any] = []
        self._preprocessing_trace: List[Dict[str, Any]] = []
        self._preprocessing_original_size: Optional[Tuple[int, int]] = None

        if isinstance(path, LazyImage):
            self._path = path.path
            self._url = path.url
            self.backend = path.backend
            self.frame_index = path.frame_index if frame_index is None else frame_index
            self.reader = path.reader if reader is None else reader
            self._instructions = list(path._instructions)
            self._preprocessing_ops = list(path._preprocessing_ops)
            self._preprocessing_trace = [dict(item) for item in path._preprocessing_trace]
            self._preprocessing_original_size = path._preprocessing_original_size
            self._cached_mode = path._cached_mode
            self._cached_size = path._cached_size
            if path.crop_box is not None and crop_box is not None:
                ox1, oy1, _, _ = path.crop_box
                x1, y1, x2, y2 = crop_box
                self._crop_box = (ox1 + x1, oy1 + y1, ox1 + x2, oy1 + y2)
            else:
                self._crop_box = crop_box or path.crop_box
        else:
            self._path = files.resolve(path)
            self._url = str(self._path)
            self.backend = backend.lower()
            self.frame_index = frame_index
            self.reader = reader
            self._crop_box = tuple(int(x) for x in crop_box) if crop_box is not None else None

    @property
    def path(self) -> Path:
        """Path to source image or video file."""
        return self._path

    @path.setter
    def path(self, new_path: Union[str, Path]) -> None:
        self._path = files.resolve(new_path)
        self._url = str(self._path)
        self._image = None
        self._processed_cache = None
        self._cached_size = None

    @property
    def url(self) -> str:
        """Source image or video URL/path."""
        return self._url

    @url.setter
    def url(self, new_url: Union[str, Path]) -> None:
        self.path = new_url

    @property
    def crop_box(self) -> Optional[Tuple[int, int, int, int]]:
        """Bounding box crop coordinates (xmin, ymin, xmax, ymax)."""
        return self._crop_box

    @crop_box.setter
    def crop_box(self, box: Optional[Tuple[int, int, int, int]]) -> None:
        self._crop_box = tuple(int(x) for x in box) if box is not None else None
        self._image = None
        self._processed_cache = None
        self._cached_size = None

    def load(self, cache: bool = True) -> Union[Image.Image, np.ndarray]:
        """Materialize image or video frame pixels from disk into RAM."""
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
            if not self._preprocessing_ops:
                if isinstance(result, np.ndarray):
                    self._cached_size = (result.shape[1], result.shape[0])
                else:
                    self._cached_size = result.size
        return result

    def unload(self) -> None:
        """Release pixel memory and revert proxy to unloaded state."""
        self._image = None
        self._processed_cache = None

    def clear(self) -> None:
        """Alias for :meth:`unload`."""
        self.unload()

    @property
    def is_loaded(self) -> bool:
        """Check whether pixel data is currently loaded in memory."""
        return self._image is not None

    @property
    def image(self) -> Union[Image.Image, np.ndarray]:
        """Access or assign underlying pixel image data."""
        return self.load()

    @image.setter
    def image(self, value: Union[Image.Image, np.ndarray, "LazyImage"]) -> None:
        self._processed_cache = None
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
        """Dimensions (width, height) following PIL convention without decoding pixels."""
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
        """Dimensions (height, width, channels) following NumPy ndarray convention."""
        if self.mode in ("L", "1", "I", "F"):
            return (self.height, self.width)
        elif self.mode == "RGBA":
            return (self.height, self.width, 4)
        return (self.height, self.width, 3)

    @property
    def dtype(self) -> np.dtype:
        """NumPy data type (uint8)."""
        return np.dtype("uint8")

    @property
    def mode(self) -> str:
        return self._cached_mode

    def to_pil(self, cache: bool = False) -> Image.Image:
        """Materialize the pipeline as a PIL image; optionally cache its result."""
        from klygo.preprocessing.core import _to_pil

        return _to_pil(self._execute_preprocessing(cache=cache))

    def to_array(self, cache: bool = False) -> np.ndarray:
        """Materialize the pipeline as a NumPy array; optionally cache its result."""
        from klygo.preprocessing.core import _to_array

        return _to_array(self._execute_preprocessing(cache=cache))

    def to_tensor(self, cache: bool = False) -> Any:
        """Materialize LazyImage into a PyTorch Tensor."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for LazyImage.to_tensor()") from exc

        result = self._execute_preprocessing(cache=cache)
        if isinstance(result, torch.Tensor):
            return result
        from klygo.preprocessing.core import _to_array

        array = np.array(_to_array(result), copy=True, order="C")
        tensor = torch.from_numpy(array)
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        return tensor

    def _execute_preprocessing(self, cache: bool = False) -> Any:
        """Decode and apply queued operations, reusing the result when cached."""
        if cache and self._processed_cache is not None:
            return self._processed_cache
        loaded = self.load(cache=cache)
        if isinstance(loaded, Image.Image):
            result: Any = loaded
        else:
            result = Image.fromarray(cv.cvtColor(loaded, cv.COLOR_BGR2RGB))

        for op, args, kwargs in self._instructions:
            result = getattr(result, op)(*args, **kwargs)

        if self._preprocessing_ops:
            from klygo.preprocessing.core import execute_pipeline

            trace: List[Dict[str, Any]] = []
            result, trace = execute_pipeline(result, self._preprocessing_ops, trace)
            self._preprocessing_trace = trace
            if isinstance(result, Image.Image):
                self._cached_size = result.size
                self._cached_mode = result.mode
                try:
                    result._preprocessing_trace = list(self._preprocessing_trace)
                except Exception:
                    pass
            elif isinstance(result, np.ndarray) and result.ndim >= 2:
                self._cached_size = (int(result.shape[1]), int(result.shape[0]))
        else:
            self._preprocessing_trace = []
        if cache:
            self._processed_cache = result
        return result

    def process(self, operation_or_pipeline: Any) -> "LazyImage":
        """Attach a preprocessing transform without decoding pixel data."""
        from klygo.preprocessing import apply

        return apply(self, operation_or_pipeline)

    def transform(self, operation_or_pipeline: Any) -> "LazyImage":
        """Apply a callable transform or pipeline to this image lazily."""
        if callable(operation_or_pipeline):
            return operation_or_pipeline(self)
        from klygo.preprocessing import apply

        return apply(self, operation_or_pipeline)

    @property
    def trace(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._preprocessing_trace]

    def lazy_crop(self, box: Tuple[int, int, int, int]) -> "LazyImage":
        """
        Zero-RAM Lazy Cropping: Create a cropped sub-image proxy without allocating pixels.
        Only crop coordinates and source image references are recorded.
        """
        if self._preprocessing_ops:
            from klygo.preprocessing import crop

            return crop(box)(self)
        return LazyImage(
            self,
            backend=self.backend,
            crop_box=box,
            frame_index=self.frame_index,
            reader=self.reader,
        )

    def crop(self, *args, lazy: Optional[bool] = None, **kwargs) -> Any:
        """
        Crop image.

        By default operates via Zero-RAM lazy cropping: returns a new LazyImage
        maintaining URL/path/frame_index without decoding pixels into RAM.
        When lazy=False is explicitly specified, materializes and returns a PIL Image.
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
            if self._preprocessing_ops:
                from klygo.preprocessing import crop

                return crop(box)(self)
            return self.lazy_crop(box)

        return self.to_pil().crop(*args, **kwargs)

    def __array__(self, dtype=None) -> np.ndarray:
        """Support automatic conversion via np.array(lazy_img)."""
        arr = self.to_array()
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    @property
    def __array_interface__(self):
        return self.to_pil(cache=False).__array_interface__

    def __getattr__(self, name: str) -> Any:
        """Forward attribute and method lookups to the materialized image."""
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
        return f"<LazyImage [{status}] url='{files.name(self.path)}'{frame_info}{crop_info} size=({w}, {h})>"


# =========================================================================
# 0C. MediaFrames: Unified Streaming & Materialized Container
# =========================================================================

class MediaFrames(list):
    """Represent media as indexed lazy frames or a bounded-memory stream.

    Parameters
    ----------
    source : str or os.PathLike[str], default=""
        Image, video, or image-directory source.
    sample_rate : int, default=1
        Yield every nth frame.
    max_frames : int or None, default=None
        Maximum frames to yield.
    recursive : bool, default=False
        Traverse subdirectories for images.
    backend : str, default='pil'
        Pixel backend, either 'pil' or 'opencv'.
    verbose : bool, default=False
        Show progress output.
    stream : bool, default=True
        Whether to stream sequentially (True) or load materialized frames (False).

    Returns
    -------
    MediaFrames
        Unified media container with materialized and streaming modes.

    Raises
    ------
    FileNotFoundError
        The source does not exist.
    ValueError
        Invalid backend or source format.
    """

    is_stream: bool = True

    def __new__(cls, *args: Any, stream: bool = False, **kwargs: Any) -> Any:
        if cls is MediaFrames and not stream:
            return load(*args, stream=False, **kwargs)
        return list.__new__(cls)

    def __init__(
        self,
        source: Union[str, os.PathLike[str]] = "",
        *,
        sample_rate: int = 1,
        max_frames: Optional[int] = None,
        recursive: bool = False,
        backend: str = "pil",
        verbose: bool = False,
        stream: bool = False,
        indexed: bool = False,
        **kwargs: Any,
    ) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        list.__init__(self)
        validate_type(source, (str, os.PathLike), "source")
        validate_count(sample_rate, "sample_rate")
        validate_type(recursive, bool, "recursive")
        validate_type(backend, str, "backend")
        validate_type(verbose, bool, "verbose")
        validate_type(indexed, bool, "indexed")
        if indexed:
            raise ValueError("stream and indexed modes are mutually exclusive")
        if max_frames is not None:
            validate_count(max_frames, "max_frames")

        backend = backend.lower()
        if backend not in ("pil", "opencv"):
            raise ValueError("backend must be 'pil' or 'opencv'")

        path = files.path(source)
        if not files.exists(path):
            raise FileNotFoundError(f"source does not exist: {path}")

        self.source_path = str(path)
        self.path = path
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.recursive = recursive
        self.backend = backend
        self.verbose = verbose
        self.is_stream = True
        self.is_indexed = False
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.fps = 30.0
        self._active_readers: List[VideoReader] = []
        self._preprocessing_ops: List[Any] = []
        self._predicates: List[Callable[[Any], bool]] = []
        self._motion_threshold: Optional[float] = None
        self._deduplicate_threshold: Optional[float] = None

        if files.is_file(path):
            suffix = files.extension(path).lower()
            if suffix in VIDEO_SUFFIXES:
                self.source_type = "video"
                reader = VideoReader(path)
                try:
                    source_total = reader.total_frames
                    self.width = reader.width
                    self.height = reader.height
                    self.fps = reader.fps / sample_rate
                finally:
                    reader.release()
            elif suffix in IMAGE_SUFFIXES:
                self.source_type = "image"
                source_total = 1
                try:
                    with Image.open(path) as image:
                        self.width, self.height = image.size
                except Exception:
                    pass
            else:
                raise ValueError(f"source is not a supported image or video file: {path}")
        elif files.is_dir(path):
            self.source_type = "folder"
            self._cached_paths = tuple(self._iter_image_paths())
            source_total = len(self._cached_paths)
            if source_total == 0:
                raise ValueError(f"No supported image files found in directory: {path}")
        else:
            raise ValueError(f"source must be a file or directory: {path}")

        if self.source_type == "video" and source_total <= 0:
            self._total_frames = None
        else:
            sampled_total = (source_total + sample_rate - 1) // sample_rate
            self._total_frames = min(sampled_total, max_frames) if max_frames is not None else sampled_total

    @classmethod
    def _from_items(
        cls,
        items: Iterable[Any],
        *,
        fps: float = 30.0,
        source_path: Optional[Union[str, Path]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        source_type: str = "video",
        reader: Optional[VideoReader] = None,
    ) -> "MediaFrames":
        """Build materialized storage without reloading the source."""
        instance = list.__new__(cls)
        list.__init__(instance, items)
        instance._initialized = True
        instance.fps = float(fps) if fps else 30.0
        instance.source_path = str(source_path) if source_path is not None else None
        instance.path = files.path(source_path) if source_path is not None else None
        instance.width = width
        instance.height = height
        instance.source_type = source_type
        instance.reader = reader
        instance.is_stream = False
        instance.is_indexed = False
        instance.sample_rate = 1
        instance.max_frames = None
        instance._active_readers = []
        instance._preprocessing_ops = []
        instance._predicates = []
        instance._motion_threshold = None
        instance._deduplicate_threshold = None
        return instance

    @classmethod
    def _from_index(
        cls,
        entries: Iterable[Any],
        *,
        fps: float = 30.0,
        source_path: Optional[Union[str, Path]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        source_type: str = "folder",
        backend: str = "pil",
        reader: Optional[VideoReader] = None,
    ) -> "MediaFrames":
        """Build path-backed indexed storage without creating frame proxies."""
        instance = list.__new__(cls)
        list.__init__(instance)
        instance._initialized = True
        instance.fps = float(fps) if fps else 30.0
        instance.source_path = str(source_path) if source_path is not None else None
        instance.path = files.path(source_path) if source_path is not None else None
        instance.width = width
        instance.height = height
        instance.source_type = source_type
        instance.backend = backend
        instance.reader = reader
        instance.is_stream = False
        instance.is_indexed = True
        instance.sample_rate = 1
        instance.max_frames = None
        instance.recursive = False
        instance.verbose = False
        instance._active_readers = []
        instance._preprocessing_ops = []
        instance._predicates = []
        instance._motion_threshold = None
        instance._deduplicate_threshold = None
        instance._indexed_entries = list(entries)
        instance._indexed_cache = weakref.WeakValueDictionary()
        instance._indexed_generation = 0
        return instance

    def _indexed_proxy(self, index: int) -> Any:
        """Return one cached indexed item, creating its lazy proxy on demand."""
        length = len(self._indexed_entries)
        position = index + length if index < 0 else index
        if position < 0 or position >= length:
            raise IndexError("list index out of range")
        entry = self._indexed_entries[position]
        if not isinstance(entry, (str, os.PathLike, int)) or isinstance(entry, bool):
            return entry
        key = (self._indexed_generation, position)
        cached = self._indexed_cache.get(key)
        if cached is not None:
            return cached
        if self.source_type == "video":
            proxy = LazyImage(
                self.path,
                backend=self.backend,
                frame_index=int(entry),
                reader=self.reader,
            )
        else:
            proxy = LazyImage(entry, backend=self.backend)
        for operation in self._preprocessing_ops:
            proxy = operation(proxy)
        self._indexed_cache[key] = proxy
        return proxy

    def _invalidate_index_cache(self) -> None:
        """Invalidate positional proxy keys after structural edits."""
        self._indexed_generation += 1
        self._indexed_cache = weakref.WeakValueDictionary()

    def _new(self, items: Iterable[Any], *, fps: Optional[float] = None) -> "MediaFrames":
        """Create a materialized container while preserving source metadata."""
        materialized = list(items)
        width, height = self.width, self.height
        if materialized:
            sizes = set()
            for item in materialized:
                operations = getattr(item, "_preprocessing_ops", ())
                if any(
                    getattr(operation, "changes_geometry", False)
                    and _operation_size(operation, (1, 1)) is None
                    for operation in operations
                ):
                    sizes.clear()
                    break
                item_size = getattr(item, "size", None)
                if isinstance(item_size, (tuple, list)) and len(item_size) == 2:
                    sizes.add((int(item_size[0]), int(item_size[1])))
                    continue
                shape = getattr(item, "shape", ())
                if len(shape) >= 2:
                    if len(shape) == 3 and shape[0] in (1, 3, 4):
                        sizes.add((int(shape[2]), int(shape[1])))
                    else:
                        sizes.add((int(shape[1]), int(shape[0])))
            width, height = next(iter(sizes)) if len(sizes) == 1 else (None, None)
        return type(self)._from_items(
            materialized,
            fps=self.fps if fps is None else fps,
            source_path=self.source_path,
            width=width,
            height=height,
            source_type=self.source_type,
            reader=self.reader,
        )

    def _require_materialized(self, operation: str) -> None:
        """Ensure random-access operation is only performed on materialized frames."""
        if getattr(self, "is_stream", False):
            raise TypeError(f"{operation} is unavailable for a stream; load with stream=False")

    @property
    def total_frames(self) -> Optional[int]:
        """Estimated or exact total frames in collection."""
        if not getattr(self, "is_stream", False):
            if getattr(self, "is_indexed", False):
                return len(self._indexed_entries)
            return list.__len__(self)
        return getattr(self, "_total_frames", None)

    @total_frames.setter
    def total_frames(self, value: Optional[int]) -> None:
        self._total_frames = value

    @property
    def source(self) -> Optional[str]:
        """Short alias for ``source_path``."""
        return self.source_path

    @property
    def loaded_count(self) -> int:
        """Number of frames currently holding decoded pixels in RAM."""
        self._require_materialized("loaded_count")
        if getattr(self, "is_indexed", False):
            return sum(
                bool(getattr(item, "is_loaded", False))
                for item in tuple(self._indexed_cache.values())
            )
        return sum(bool(getattr(item, "is_loaded", False)) for item in self)

    def unload(self) -> None:
        """Release cached pixels across all items while preserving the collection."""
        self._require_materialized("unload()")
        items = tuple(self._indexed_cache.values()) if getattr(self, "is_indexed", False) else self
        for item in items:
            unload = getattr(item, "unload", None)
            if callable(unload):
                unload()

    def to_list(self) -> List[Any]:
        """Convert items to a standard list."""
        self._require_materialized("to_list()")
        return list(self)

    def __getitem__(self, index: Union[int, slice]) -> Any:
        self._require_materialized("indexing and slicing")
        if getattr(self, "is_indexed", False):
            if isinstance(index, slice):
                indices = range(*index.indices(len(self._indexed_entries)))
                entries = [self._indexed_entries[position] for position in indices]
                step = index.step or 1
                result = type(self)._from_index(
                    entries,
                    fps=self.fps / abs(step) if abs(step) > 1 else self.fps,
                    source_path=self.source_path,
                    width=self.width,
                    height=self.height,
                    source_type=self.source_type,
                    backend=self.backend,
                    reader=self.reader,
                )
                result._preprocessing_ops = list(self._preprocessing_ops)
                return result
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("frame index must be an integer or slice")
            return self._indexed_proxy(index)
        if isinstance(index, slice):
            sub_items = list.__getitem__(self, index)
            step = index.step or 1
            new_fps = self.fps / abs(step) if abs(step) > 1 else self.fps
            return self._new(sub_items, fps=new_fps)
        return list.__getitem__(self, index)

    def __len__(self) -> int:
        self._require_materialized("len()")
        if getattr(self, "is_indexed", False):
            return len(self._indexed_entries)
        return list.__len__(self)

    def __contains__(self, item: Any) -> bool:
        self._require_materialized("membership testing")
        return any(candidate == item for candidate in self)

    def __eq__(self, other: Any) -> bool:
        if getattr(self, "is_stream", False):
            return self is other
        try:
            return list(self) == list(other)
        except TypeError:
            return False

    def __add__(self, other: Iterable[Any]) -> list[Any]:
        self._require_materialized("concatenation")
        return list(self) + list(other)

    def __iadd__(self, other: Iterable[Any]) -> "MediaFrames":
        self.extend(other)
        return self

    def __reversed__(self) -> Iterator[Any]:
        self._require_materialized("reversed iteration")
        return (self[index] for index in range(len(self) - 1, -1, -1))

    def index(self, item: Any, start: int = 0, stop: Optional[int] = None) -> int:
        """Return the first matching item position."""
        self._require_materialized("index()")
        limit = len(self) if stop is None else min(stop, len(self))
        for position in range(*slice(start, limit).indices(len(self))):
            if self[position] == item:
                return position
        raise ValueError(f"{item!r} is not in MediaFrames")

    def count(self, item: Any) -> int:
        """Return the number of matching items."""
        self._require_materialized("count()")
        return sum(candidate == item for candidate in self)

    def reverse(self) -> None:
        """Reverse frames in place."""
        self._require_materialized("reverse()")
        if getattr(self, "is_indexed", False):
            self._indexed_entries.reverse()
            self._invalidate_index_cache()
            return
        list.reverse(self)

    def sort(self, *, key: Optional[Callable[[Any], Any]] = None, reverse: bool = False) -> None:
        """Sort frames in place using list-compatible semantics."""
        self._require_materialized("sort()")
        if getattr(self, "is_indexed", False):
            decorated = [(self[index], self._indexed_entries[index]) for index in range(len(self))]
            decorated.sort(key=(lambda pair: key(pair[0])) if key is not None else (lambda pair: pair[0]), reverse=reverse)
            self._indexed_entries = [entry for _item, entry in decorated]
            self._invalidate_index_cache()
            return
        list.sort(self, key=key, reverse=reverse)

    def append(self, item: Any) -> None:
        self._require_materialized("append()")
        if getattr(self, "is_indexed", False):
            self._indexed_entries.append(item)
            return
        list.append(self, item)

    def extend(self, items: Iterable[Any]) -> None:
        self._require_materialized("extend()")
        if getattr(self, "is_indexed", False):
            self._indexed_entries.extend(items)
            return
        list.extend(self, items)

    def insert(self, index: int, item: Any) -> None:
        self._require_materialized("insert()")
        if getattr(self, "is_indexed", False):
            self._indexed_entries.insert(index, item)
            self._invalidate_index_cache()
            return
        list.insert(self, index, item)

    def pop(self, index: int = -1) -> Any:
        self._require_materialized("pop()")
        if getattr(self, "is_indexed", False):
            item = self[index]
            self._indexed_entries.pop(index)
            self._invalidate_index_cache()
            return item
        return list.pop(self, index)

    def remove(self, item: Any) -> None:
        self._require_materialized("remove()")
        if getattr(self, "is_indexed", False):
            for index, candidate in enumerate(self):
                if candidate == item:
                    self._indexed_entries.pop(index)
                    self._invalidate_index_cache()
                    return
            raise ValueError("MediaFrames.remove(x): x not in collection")
        list.remove(self, item)

    def clear(self) -> None:
        self._require_materialized("clear()")
        if getattr(self, "is_indexed", False):
            self._indexed_entries.clear()
            self._invalidate_index_cache()
            return
        list.clear(self)

    def __setitem__(self, index: Any, value: Any) -> None:
        self._require_materialized("item assignment")
        if getattr(self, "is_indexed", False):
            if isinstance(index, slice):
                self._indexed_entries[index] = list(value)
            else:
                self._indexed_entries[index] = value
            self._invalidate_index_cache()
            return
        list.__setitem__(self, index, value)

    def __delitem__(self, index: Any) -> None:
        self._require_materialized("item deletion")
        if getattr(self, "is_indexed", False):
            del self._indexed_entries[index]
            self._invalidate_index_cache()
            return
        list.__delitem__(self, index)

    def _iter_image_paths(self) -> Iterator[Path]:
        iterator = files.iter_find(self.path, recursive=self.recursive)
        paths = (p for p in iterator if files.is_file(p) and files.extension(p).lower() in IMAGE_SUFFIXES)
        yield from sorted(paths, key=lambda p: str(p).lower())

    def _clone(self) -> "MediaFrames":
        # Reuse validated source metadata instead of scanning a large folder or
        # probing a video again whenever a lazy stream operation is attached.
        cloned = list.__new__(MediaFrames)
        cloned.__dict__.update(self.__dict__)
        cloned._active_readers = []
        cloned._preprocessing_ops = list(self._preprocessing_ops)
        cloned._predicates = list(self._predicates)
        cloned._motion_threshold = self._motion_threshold
        cloned._deduplicate_threshold = self._deduplicate_threshold
        cloned._cached_paths = getattr(self, "_cached_paths", None)
        if getattr(self, "is_indexed", False):
            cloned._indexed_entries = list(self._indexed_entries)
            cloned._indexed_cache = weakref.WeakValueDictionary()
            cloned._indexed_generation = self._indexed_generation
        return cloned

    def transform(self, operation_or_pipeline: Any) -> Any:
        """Apply a preprocessing transform, media pipeline, or callable function."""
        from klygo.preprocessing import Compose, Operation

        if type(operation_or_pipeline).__name__ == "MediaCompose" or isinstance(operation_or_pipeline, globals().get("MediaCompose", tuple)):
            return operation_or_pipeline(self)

        if isinstance(operation_or_pipeline, Operation):
            operations = [operation_or_pipeline]
        elif isinstance(operation_or_pipeline, Compose):
            operations = list(operation_or_pipeline.operations)
        elif callable(operation_or_pipeline):
            return operation_or_pipeline(self)
        else:
            raise TypeError("transform expects preprocessing.Operation, preprocessing.Compose, media.compose, or callable")

        if self.is_stream or getattr(self, "is_indexed", False):
            cloned = self._clone()
            cloned._preprocessing_ops.extend(operations)
            size = None if cloned.width is None or cloned.height is None else (cloned.width, cloned.height)
            for operation in operations:
                size = _operation_size(operation, size)
            cloned.width, cloned.height = size if size is not None else (None, None)
            return cloned
        return self.map(operation_or_pipeline)

    def map(self, func: Callable[[Any], Any]) -> "MediaFrames":
        """Attach a custom frame-level callable transform lazily."""
        if not callable(func):
            raise TypeError("func must be callable")
        if not self.is_stream:
            return self._new(func(frame) for frame in self)
        cloned = self._clone()
        cloned._preprocessing_ops.append(func)
        return cloned

    def where(self, predicate: Callable[[Any], bool]) -> Any:
        """Retain items satisfying predicate evaluated sequentially."""
        if not callable(predicate):
            raise TypeError("predicate must be callable")
        if self.is_stream:
            cloned = self._clone()
            cloned._predicates.append(predicate)
            cloned.total_frames = None
            return cloned
        return self._new(frame for frame in self if predicate(frame))

    def filter(self, func: Callable[[Any], bool]) -> Any:
        """Compatibility alias for :meth:`where`."""
        return self.where(func)

    def where_quality(
        self,
        *,
        min_brightness: Optional[float] = None,
        max_brightness: Optional[float] = None,
        min_contrast: Optional[float] = None,
        min_sharpness: Optional[float] = None,
        max_blur: Optional[float] = None,
        min_entropy: Optional[float] = None,
        preview_size: Tuple[int, int] = (160, 160),
    ) -> Any:
        """Filter frames based on image quality metrics."""
        from klygo import preprocessing as p

        conditions = []
        if min_brightness is not None:
            conditions.append(p.brightness_score(preview_size) >= min_brightness)
        if max_brightness is not None:
            conditions.append(p.brightness_score(preview_size) <= max_brightness)
        if min_contrast is not None:
            conditions.append(p.contrast_score(preview_size) >= min_contrast)
        if min_sharpness is not None:
            conditions.append(p.sharpness_score(preview_size) >= min_sharpness)
        if max_blur is not None:
            conditions.append(p.blur_score(preview_size=preview_size) <= max_blur)
        if min_entropy is not None:
            conditions.append(p.entropy_score(preview_size) >= min_entropy)
        return self.where(p.all_of(*conditions)) if conditions else (self._clone() if self.is_stream else self.copy())

    def where_motion(self, threshold: float = 0.02) -> Any:
        """Filter frames that show motion difference above threshold."""
        if self.is_stream:
            cloned = self._clone()
            cloned._motion_threshold = float(threshold)
            cloned.total_frames = None
            return cloned
        from klygo import preprocessing as p
        filtered = []
        previous = None
        for frame in self:
            if previous is not None and p.motion_score(previous)(frame) < threshold:
                continue
            previous = frame
            filtered.append(frame)
        return self._new(filtered)

    def deduplicate(self, threshold: float = 0.95) -> Any:
        """Filter consecutive frames that have similarity above threshold."""
        if self.is_stream:
            cloned = self._clone()
            cloned._deduplicate_threshold = float(threshold)
            cloned.total_frames = None
            return cloned
        from klygo import preprocessing as p
        filtered = []
        previous = None
        for frame in self:
            was_loaded = getattr(frame, "is_loaded", False)
            if previous is not None and p.similarity_score(previous)(frame) >= threshold:
                if not was_loaded:
                    frame.unload()
                continue
            if previous is not None and not getattr(previous, "_initially_loaded", False):
                previous.unload()
            frame._initially_loaded = was_loaded
            previous = frame
            filtered.append(frame)
        if previous is not None and not getattr(previous, "_initially_loaded", False):
            previous.unload()
        for f in filtered:
            if hasattr(f, "_initially_loaded"):
                delattr(f, "_initially_loaded")
        return self._new(filtered)

    def sample(self, sample_rate: int) -> "MediaFrames":
        """Sample every nth frame."""
        validate_count(sample_rate, "sample_rate")
        if not self.is_stream:
            return self[::sample_rate]
        cloned = self._clone()
        cloned.sample_rate = self.sample_rate * sample_rate
        cloned.fps = self.fps / sample_rate
        if cloned.total_frames is not None:
            cloned.total_frames = (cloned.total_frames + sample_rate - 1) // sample_rate
        return cloned

    def limit(self, max_frames: Optional[int]) -> "MediaFrames":
        """Limit maximum frames yielded or retained."""
        if max_frames is not None:
            validate_count(max_frames, "max_frames")
        if not self.is_stream:
            return self.copy() if max_frames is None else self[:max_frames]
        cloned = self._clone()
        if max_frames is None:
            return cloned
        cloned.max_frames = min(cloned.max_frames, max_frames) if cloned.max_frames is not None else max_frames
        if cloned.total_frames is not None and max_frames is not None:
            cloned.total_frames = min(cloned.total_frames, max_frames)
        return cloned

    def save_video(self, output_path: Union[str, Path], fps: Optional[float] = None, **kwargs: Any) -> Path:
        """Encode and save all frames into an MP4/AVI video file."""
        target_fps = fps if fps is not None else getattr(self, "fps", 30.0)
        return globals()["save_video"](output_path, self, fps=target_fps, **kwargs)

    def save_images(self, output_dir: Union[str, Path], prefix: str = "frame", **kwargs: Any) -> Path:
        """Save all frames into an image directory."""
        return globals()["save_images"](output_dir, self, prefix=prefix, **kwargs)

    def copy(self) -> Any:
        """Copy collection without decoding pixels or losing metadata."""
        if self.is_stream or getattr(self, "is_indexed", False):
            return self._clone()
        return self._new(self)

    def __iter__(self) -> Iterator[LazyImage]:
        if not self.is_stream:
            if getattr(self, "is_indexed", False):
                for index in range(len(self._indexed_entries)):
                    yield self._indexed_proxy(index)
                return
            yield from list.__iter__(self)
            return
        previous_frame: Optional[LazyImage] = None

        def prepare(frame: LazyImage) -> Optional[LazyImage]:
            nonlocal previous_frame
            from klygo import preprocessing as p

            if any(not predicate(frame) for predicate in self._predicates):
                return None
            if previous_frame is not None and self._motion_threshold is not None:
                if p.motion_score(previous_frame)(frame) < self._motion_threshold:
                    return None
            if previous_frame is not None and self._deduplicate_threshold is not None:
                if p.similarity_score(previous_frame)(frame) >= self._deduplicate_threshold:
                    return None
            previous_frame = frame
            if self._preprocessing_ops:
                res = frame
                for op in self._preprocessing_ops:
                    res = op(res)
                return res
            return frame

        if self.source_type == "video":
            reader = VideoReader(self.path)
            self._active_readers.append(reader)
            emitted = 0
            frame_index = 0
            try:
                while reader.total_frames <= 0 or frame_index < reader.total_frames:
                    if self.max_frames is not None and emitted >= self.max_frames:
                        break

                    lazy_image = LazyImage(
                        self.path,
                        backend=self.backend,
                        frame_index=frame_index,
                        reader=reader,
                    )

                    # Some streams lack frame counts in container headers.
                    # Read ahead one frame to detect EOF while bounding memory.
                    if reader.total_frames <= 0:
                        frame = reader.read_frame(frame_index)
                        if frame is None:
                            break
                        if self.backend == "opencv":
                            lazy_image._image = frame
                        else:
                            lazy_image._image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                        lazy_image._cached_size = (frame.shape[1], frame.shape[0])

                    prepared = prepare(lazy_image)
                    if prepared is not None:
                        yield prepared
                        emitted += 1
                    if reader.total_frames <= 0:
                        lazy_image.unload()
                    frame_index += self.sample_rate
            finally:
                reader.release()
                if reader in self._active_readers:
                    self._active_readers.remove(reader)
            return

        paths: Iterable[Path]
        if self.source_type == "image":
            paths = (self.path,)
        elif getattr(self, "_cached_paths", None) is not None:
            paths = self._cached_paths
        else:
            paths = self._iter_image_paths()

        emitted = 0
        for index, path in enumerate(paths):
            if index % self.sample_rate != 0:
                continue
            if self.max_frames is not None and emitted >= self.max_frames:
                break
            prepared = prepare(LazyImage(path, backend=self.backend))
            if prepared is not None:
                yield prepared
                emitted += 1

    def close(self) -> None:
        """Release underlying video captures, readers, and cached pixels."""
        for reader in tuple(getattr(self, "_active_readers", ())):
            reader.release()
        if hasattr(self, "_active_readers"):
            self._active_readers.clear()
        if hasattr(self, "reader") and self.reader is not None:
            try:
                self.reader.release()
            except Exception:
                pass
            self.reader = None
        if not getattr(self, "is_stream", False):
            items = (
                tuple(self._indexed_cache.values())
                if getattr(self, "is_indexed", False)
                else getattr(self, "__iter__", lambda: ())()
            )
            for item in items:
                unload = getattr(item, "unload", None)
                if callable(unload):
                    unload()

    def __enter__(self) -> "MediaFrames":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        if self.is_stream:
            return (
                f"<MediaFrames: stream=True, type='{self.source_type}', total_frames={self.total_frames}, "
                f"fps={self.fps}>"
            )
        return (
            f"<MediaFrames: {len(self)} frames, loaded={self.loaded_count}, "
            f"indexed={getattr(self, 'is_indexed', False)}, "
            f"type='{self.source_type}', fps={self.fps}>"
        )


# =========================================================================
# 0D. MediaCompose: Pipeline Composition for Media Operations
# =========================================================================

class MediaCompose:
    """Chain media stream or frames filtering, sampling, and transform steps.

    Parameters
    ----------
    steps : iterable or varargs of callables
        Sequence of filtering steps, transform pipelines, or container functions.

    Returns
    -------
    MediaCompose
        Callable pipeline that executes steps sequentially on media containers.
    """

    def __init__(self, *steps: Any) -> None:
        if len(steps) == 1 and isinstance(steps[0], (list, tuple)):
            steps = steps[0]
        self.steps = tuple(steps)

    def __call__(self, target: Any) -> Any:
        from klygo import preprocessing as p

        if isinstance(target, (str, os.PathLike)):
            target = globals()["load"](target)

        result = target
        for step in self.steps:
            if isinstance(step, (p.Operation, p.Compose)):
                result = result.transform(step)
            elif callable(step):
                result = step(result)
            else:
                raise TypeError(f"Expected callable step or preprocessing operation, got {type(step).__name__}")
        return result

    def __repr__(self) -> str:
        return f"<MediaCompose: {len(self.steps)} steps>"


class _ComposeStep:
    """Hold a named lazy filtering step for media.compose."""

    def __init__(self, func: Callable[[Any], Any], name: str) -> None:
        self.func = func
        self.name = name

    def __call__(self, target: Any) -> Any:
        return self.func(target)

    def __repr__(self) -> str:
        return f"<ComposeStep: {self.name}>"


def compose(*steps: Any) -> MediaCompose:
    """Create a composite filtering and transformation pipeline for media.

    Parameters
    ----------
    *steps : callable, preprocessing.Compose, or list thereof
        Sequential filtering steps or transforms. Accepts varargs or a list.

    Returns
    -------
    MediaCompose
        Callable pipeline applicable to ``MediaFrames`` or file paths.

    Examples
    --------
    >>> from klygo import media, preprocessing as p
    >>> pipeline = media.compose(
    ...     media.compose.quality(min_brightness=0.3),
    ...     media.compose.sample(2),
    ...     p.resize((640, 640)),
    ... )
    >>> stream = media.load("video.mp4", stream=True)
    >>> processed = pipeline(stream)
    """
    return MediaCompose(*steps)


compose.where = lambda predicate: _ComposeStep(lambda c: c.where(predicate), f"where({getattr(predicate, '__name__', 'predicate')})")
compose.filter = compose.where
compose.quality = lambda **kwargs: _ComposeStep(lambda c: c.where_quality(**kwargs), f"quality({kwargs})")
compose.where_quality = compose.quality
compose.motion = lambda threshold=0.02: _ComposeStep(lambda c: c.where_motion(threshold), f"motion({threshold})")
compose.where_motion = compose.motion
compose.deduplicate = lambda threshold=0.95: _ComposeStep(lambda c: c.deduplicate(threshold), f"deduplicate({threshold})")
compose.sample = lambda sample_rate=1: _ComposeStep(lambda c: c.sample(sample_rate), f"sample({sample_rate})")
compose.limit = lambda max_frames=None: _ComposeStep(lambda c: c.limit(max_frames), f"limit({max_frames})")

# Top-level step builders for media.compose
quality = compose.quality
where_quality = compose.where_quality
motion = compose.motion
where_motion = compose.where_motion
deduplicate = compose.deduplicate
sample = compose.sample
limit = compose.limit
where = compose.where


# =========================================================================
# 1. Media Load / Save / Convert / Copy / Info
# =========================================================================

def _read_video_frames(
    path: Path,
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

    return MediaFrames._from_items(
        lazy_frames,
        fps=fps,
        source_path=path,
        width=width,
        height=height,
        source_type="video",
        reader=reader,
    )


def load(
    source: Union[str, os.PathLike[str]],
    recursive: bool = False,
    backend: str = "pil",
    verbose: bool = False,
    *,
    stream: bool = False,
    indexed: bool = False,
    sample_rate: int = 1,
    max_frames: Optional[int] = None,
) -> MediaFrames:
    """Load an image, video, or image directory as lazy media frames.

    By default, return a list-like collection for indexing and slicing. With
    ``stream=True``, return a bounded-memory iterator over the source.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Image file, video file, or directory of images.
    recursive : bool, default=False
        Search subdirectories when ``source`` is a directory.
    backend : str, default='pil'
        Pixel backend, either ``'pil'`` or ``'opencv'``.
    verbose : bool, default=False
        Enable progress output where supported.
    stream : bool, default=False
        Yield frames sequentially instead of storing all frame proxies.
    indexed : bool, default=False
        Store only paths or video frame numbers and create ``LazyImage``
        proxies on demand. This mode preserves random access without eagerly
        allocating one proxy per frame.
    sample_rate : int, default=1
        Use every nth frame when ``stream=True``.
    max_frames : int or None, default=None
        Maximum number of yielded frames when ``stream=True``.

    Returns
    -------
    MediaFrames
        Unified container in materialized-proxy mode by default, path-indexed
        mode with ``indexed=True``, or streaming mode with ``stream=True``.

    Raises
    ------
    TypeError
        If an argument has an unsupported type.
    FileNotFoundError
        If ``source`` does not exist.
    ValueError
        If the source or backend is unsupported, or streaming options are
        invalid or supplied without ``stream=True``.

    Examples
    --------
    >>> import klygo.media as media
    >>> frames = media.load("video.mp4")
    >>> first = frames[0]
    >>> streamed = media.load("video.mp4", stream=True, sample_rate=2)
    >>> first_sampled = next(iter(streamed))
    """
    validate_type(source, (str, os.PathLike), "source")
    validate_type(recursive, bool, "recursive")
    validate_type(backend, str, "backend")
    validate_type(verbose, bool, "verbose")
    validate_type(stream, bool, "stream")
    validate_type(indexed, bool, "indexed")

    if stream and indexed:
        raise ValueError("stream and indexed modes are mutually exclusive")

    if stream:
        return MediaFrames(
            source,
            stream=True,
            sample_rate=sample_rate,
            max_frames=max_frames,
            recursive=recursive,
            backend=backend,
            verbose=verbose,
        )
    if sample_rate != 1 or max_frames is not None:
        raise ValueError("sample_rate and max_frames require stream=True")

    backend = backend.lower()
    if backend not in ("pil", "opencv"):
        raise ValueError("backend must be 'pil' or 'opencv'")

    p = files.path(source)
    if not files.exists(p):
        raise FileNotFoundError(f"source does not exist: {p}")

    if files.is_file(p):
        suf = files.extension(p).lower()
        if suf in VIDEO_SUFFIXES:
            if indexed:
                reader = VideoReader(p)
                total_frames = reader.total_frames
                if total_frames <= 0:
                    reader.release()
                    raise ValueError("indexed video loading requires a known frame count")
                return MediaFrames._from_index(
                    range(total_frames),
                    fps=reader.fps,
                    source_path=p,
                    width=reader.width,
                    height=reader.height,
                    source_type="video",
                    backend=backend,
                    reader=reader,
                )
            return _read_video_frames(p, backend=backend, verbose=verbose)
        elif suf in IMAGE_SUFFIXES:
            image_paths = [p]
        else:
            raise ValueError(f"source is not a supported image or video file: {p}")
    elif files.is_dir(p):
        iterator = files.iter_find(p, recursive=recursive)
        image_paths = sorted(
            (
                f
                for f in iterator
                if files.is_file(f) and files.extension(f).lower() in IMAGE_SUFFIXES
            ),
            key=lambda f: str(f).lower(),
        )
        if not image_paths:
            raise ValueError(f"No supported image files found in directory: {p}")
    else:
        raise ValueError(f"source must be a file or directory: {p}")

    if indexed:
        return MediaFrames._from_index(
            image_paths,
            source_path=p,
            source_type="folder" if files.is_dir(p) else "image",
            backend=backend,
        )

    # Lazy pixel loading: proxies are materialized while pixels remain deferred.
    lazy_frames = [LazyImage(f, backend=backend) for f in image_paths]

    return MediaFrames._from_items(
        lazy_frames,
        source_path=p,
        source_type="folder" if files.is_dir(p) else "image",
    )


def save(
    path: Union[str, Path],
    image: Union[Image.Image, np.ndarray],
    overwrite: bool = False,
    verbose: bool = False,
) -> Path:
    """Save one image to the requested path.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination image file.
image : PIL.Image.Image, numpy.ndarray, or LazyImage
        Pixels to write.
overwrite : bool
        Replace an existing file when true.
verbose : bool
        Show progress when true.

    Returns
    -------
    pathlib.Path
        Destination path.

    Raises
    ------
    FileExistsError
        Destination exists without overwrite.
TypeError
        Image type is unsupported."""
    validate_type(path, (str, os.PathLike), "path")
    validate_type(overwrite, bool, "overwrite")
    validate_type(verbose, bool, "verbose")

    p = files.path(path)
    if files.exists(p) and not overwrite:
        raise FileExistsError(f"File already exists: {p}. Use overwrite=True to replace it.")

    files.mkdir(files.parent(p))

    if isinstance(image, LazyImage):
        image = image.to_pil(cache=False)

    with ProgressBar(total=1, desc=f"Saving image {files.name(p)}", unit="file", verbose=verbose, colour="cyan") as pbar:
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
    """Convert an image format or transcode a video file.

    Parameters
    ----------
    source, target : str or pathlib.Path
        Existing input and requested output.
codec : str, optional
        Video codec override.
crf : int
        Video quality control.
fps : float, optional
        Output frame rate.
gpu : bool, optional
        Select or autodetect GPU encoding.
preset : str, optional
        Video encoder preset.
overwrite, verbose : bool
        Replacement and progress options.

    Returns
    -------
    pathlib.Path
        Converted output path.

    Raises
    ------
    FileNotFoundError
        Source is missing.
ValueError
        Input and output are incompatible media kinds."""
    validate_type(source, (str, os.PathLike), "source")
    validate_type(target, (str, os.PathLike), "target")
    validate_type(overwrite, bool, "overwrite")
    validate_type(verbose, bool, "verbose")

    src_p = files.path(source)
    tgt_p = files.path(target)

    if not files.exists(src_p):
        raise FileNotFoundError(f"source file does not exist: {src_p}")

    src_suf = files.extension(src_p).lower()
    tgt_suf = files.extension(tgt_p).lower()

    if src_suf in IMAGE_SUFFIXES and tgt_suf in IMAGE_SUFFIXES:
        with ProgressBar(total=1, desc=f"Converting image {files.name(src_p)} -> {files.name(tgt_p)}", unit="file", verbose=verbose, colour="cyan") as pbar:
            imgs = load(src_p, verbose=False)
            res = save(tgt_p, imgs[0], overwrite=overwrite, verbose=False)
            pbar.update(1)
            return res
    elif src_suf in VIDEO_SUFFIXES and tgt_suf in VIDEO_SUFFIXES:
        import shutil
        import subprocess

        # Auto-detect GPU if gpu=None
        use_gpu = gpu
        if use_gpu is None:
            torch = _optional_torch()
            if torch is not None and torch.cuda.is_available():
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
                with ProgressBar(total=total_frames, desc=f"Converting {files.name(src_p)} (FFmpeg)", unit="frame", verbose=verbose, colour="cyan") as pbar:
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
    """Copy a media file or directory to another location.

    Parameters
    ----------
    source, target : str or pathlib.Path
        Existing input and requested output.
overwrite : bool
        Allow replacement of an existing target.

    Returns
    -------
    pathlib.Path
        Copied destination.

    Raises
    ------
    FileNotFoundError
        Source does not exist.
FileExistsError
        Destination exists without overwrite."""
    validate_type(source, (str, os.PathLike), "source")
    validate_type(target, (str, os.PathLike), "target")
    validate_type(overwrite, bool, "overwrite")

    src_p = files.path(source)
    if not files.exists(src_p):
        raise FileNotFoundError(f"Media source does not exist: {src_p}")

    return files.copy(src_p, target, overwrite=overwrite)


def save_video(
    output_path: Union[str, Path],
    frames: Iterable[Union[Image.Image, np.ndarray]],
    fps: float = 30.0,
    fourcc: str = "mp4v",
    overwrite: bool = False,
    verbose: bool = False,
) -> Path:
    """Encode an iterable of image frames as a video.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Output video path.
frames : iterable
        Frames with matching dimensions.
fps : float
        Playback frame rate.
fourcc : str
        Four-character OpenCV codec.
overwrite, verbose : bool
        Replacement and progress options.

    Returns
    -------
    pathlib.Path
        Encoded video path.

    Raises
    ------
    ValueError
        Frames iterable is empty.
RuntimeError
        Video writer cannot be opened."""
    validate_type(output_path, (str, os.PathLike), "output_path")
    validate_type(overwrite, bool, "overwrite")
    validate_type(verbose, bool, "verbose")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise TypeError("fps must be a number")
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("fps must be a finite positive number")
    if not isinstance(fourcc, str):
        raise TypeError("fourcc must be a string")
    if len(fourcc) != 4:
        raise ValueError("fourcc must contain exactly four characters")

    p = files.path(output_path)
    if files.exists(p) and not overwrite:
        raise FileExistsError(f"Video file already exists: {p}. Use overwrite=True to replace it.")

    files.mkdir(files.parent(p))

    frame_iter = iter(frames)
    try:
        first_frame = next(frame_iter)
    except StopIteration:
        raise ValueError("frames iterable is empty")

    def _video_frame(value: Any, expected: Optional[Tuple[int, int]] = None) -> np.ndarray:
        array = np.asarray(to_array(value))
        if array.ndim == 2:
            array = cv.cvtColor(array, cv.COLOR_GRAY2RGB)
        elif array.ndim == 3 and array.shape[2] == 1:
            array = cv.cvtColor(array[..., 0], cv.COLOR_GRAY2RGB)
        elif array.ndim == 3 and array.shape[2] == 4:
            array = cv.cvtColor(array, cv.COLOR_RGBA2RGB)
        elif array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("video frames must have one, three, or four channels")
        if array.dtype != np.uint8:
            raise TypeError("video frames must use uint8 pixels")
        current = int(array.shape[1]), int(array.shape[0])
        if expected is not None and current != expected:
            raise ValueError(f"all video frames must have size {expected}; received {current}")
        return np.ascontiguousarray(cv.cvtColor(array, cv.COLOR_RGB2BGR))

    first_bgr = _video_frame(first_frame)
    height, width = first_bgr.shape[:2]
    frame_size = (width, height)
    extension = files.extension(p) or ".mp4"
    temporary = files.join(
        files.parent(p), f".{files.stem(p)}.{uuid.uuid4().hex}.tmp{extension}"
    )

    fourcc_code = cv.VideoWriter_fourcc(*fourcc)
    writer = cv.VideoWriter(str(temporary), fourcc_code, float(fps), frame_size)
    if not writer.isOpened():
        files.remove(temporary, missing_ok=True)
        raise RuntimeError(f"Could not open VideoWriter for {p} with fourcc {fourcc!r}")

    try:
        with ProgressBar(total=None, desc=f"Encoding video {files.name(p)}", unit="frame", verbose=verbose, colour="cyan") as pbar:
            writer.write(first_bgr)
            pbar.update(1)
            for frame in frame_iter:
                writer.write(_video_frame(frame, frame_size))
                pbar.update(1)
    except Exception:
        writer.release()
        files.remove(temporary, missing_ok=True)
        raise
    finally:
        writer.release()

    files.move(temporary, p, overwrite=overwrite)
    return p


def save_images(
    output_dir: Union[str, Path],
    images: Iterable[Union[Image.Image, np.ndarray]],
    prefix: str = "frame",
    extension: str = ".jpg",
    overwrite: bool = False,
    verbose: bool = False,
) -> List[Path]:
    """Save an iterable of images to numbered files.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Output directory.
images : iterable
        Images to write.
prefix, extension : str
        Generated filename components.
overwrite, verbose : bool
        Replacement and progress options.

    Returns
    -------
    list of pathlib.Path
        Paths written in input order.

    Raises
    ------
    FileExistsError
        A generated file already exists without overwrite."""
    validate_type(output_dir, (str, os.PathLike), "output_dir")
    validate_type(overwrite, bool, "overwrite")
    validate_type(verbose, bool, "verbose")

    out_p = files.path(output_dir)
    files.mkdir(out_p)

    if not extension.startswith("."):
        extension = f".{extension}"

    saved_paths: List[Path] = []
    is_sequential = bool(getattr(images, "is_stream", False))
    total = len(images) if isinstance(images, (builtins.list, tuple)) and not is_sequential else getattr(images, "total_frames", None)
    with ProgressBar(total=total, desc=f"Saving image batch to {files.name(out_p)}", unit="file", verbose=verbose, colour="cyan") as pbar:
        for idx, img in enumerate(images, start=1):
            file_path = files.join(out_p, f"{prefix}_{idx:06d}{extension}")
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
    """Yield sampled frames from a video or image directory.

    Parameters
    ----------
    source : str or pathlib.Path
        Existing media source.
sample_rate : int
        Yield every nth frame.
recursive : bool
        Traverse image subdirectories.
backend : str
        PIL or OpenCV pixel backend.
verbose : bool
        Show progress when true.

    Returns
    -------
    generator
        Frames decoded as they are requested.

    Raises
    ------
    ValueError
        Sampling rate is below one.
FileNotFoundError
        Source is missing."""
    validate_count(sample_rate, "sample_rate")

    p = files.path(source)
    if not files.exists(p):
        raise FileNotFoundError(f"source does not exist: {p}")

    backend = backend.lower()
    if backend not in ("pil", "opencv"):
        raise ValueError("backend must be 'pil' or 'opencv'")

    if files.is_file(p) and files.extension(p).lower() in VIDEO_SUFFIXES:
        cap = cv.VideoCapture(str(p))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {p}")

        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        try:
            with ProgressBar(total=total_frames if total_frames > 0 else None, desc=f"Iterating video {files.name(p)}", unit="frame", verbose=verbose, colour="cyan") as pbar:
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
        imgs = load(source, recursive=recursive, backend=backend, verbose=verbose)
        for idx, img in enumerate(imgs):
            if idx % sample_rate == 0:
                yield img


def probe(path: Union[str, Path], *, verbose: bool = False) -> Dict[str, Any]:
    """Read image or video metadata from a media file.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing media file.
    verbose : bool, default=False
        Print the metadata as a Rich table while returning the dictionary.

    Returns
    -------
    dict
        Format, dimensions, and video timing where available.

    Raises
    ------
    FileNotFoundError
        Path does not exist.
    ValueError
        Path is a directory.
    """
    p = files.path(path)
    if not files.exists(p):
        raise FileNotFoundError(f"Path does not exist: {p}")
    if not files.is_file(p):
        raise ValueError(f"Path must be a file: {p}")

    suf = files.extension(p).lower()

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

        result = {
            "name": files.name(p),
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
    else:
        with Image.open(p) as img:
            width, height = img.size
            result = {
                "name": files.name(p),
                "path": p,
                "type": "image",
                "format": img.format,
                "mode": img.mode,
                "width": width,
                "height": height,
                "size": (width, height),
            }

    if verbose:
        from rich.console import Console
        from rich.table import Table

        table = Table(title=f"Media Information: {files.name(p)}")
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        for key, val in result.items():
            label = str(key).replace("_", " ").title()
            table.add_row(label, str(val))
        Console().print(table)

    return result

# =========================================================================
# 2. Format Conversions (to_array, to_tensor, to_pil)
# =========================================================================

def to_array(image: Any) -> np.ndarray:
    """Convert supported image data to a NumPy array.

    Parameters
    ----------
    image : PIL image, numpy.ndarray, LazyImage, or torch.Tensor
        Input image representation.

    Returns
    -------
    numpy.ndarray
        Independent pixel array.

    Raises
    ------
    TypeError
        Image representation is unsupported."""
    if isinstance(image, LazyImage):
        return image.to_array()

    if isinstance(image, np.ndarray):
        return image.copy()

    if isinstance(image, Image.Image):
        return np.array(image)

    torch = _optional_torch()
    if torch is not None and isinstance(image, torch.Tensor):
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
    """Convert supported image data to a CHW PyTorch tensor.

    Parameters
    ----------
    image : PIL image, numpy.ndarray, LazyImage, or torch.Tensor
        Input image representation.
    normalize : bool
        Scale uint8 pixels to floating-point values in 0-1.

    Returns
    -------
    torch.Tensor
        CHW tensor.

    Raises
    ------
    ImportError
        PyTorch is unavailable.
TypeError
        Image representation is unsupported."""
    torch = _optional_torch(required=True)

    if isinstance(image, LazyImage):
        tensor = image.to_tensor(cache=False)
        if normalize and tensor.dtype == torch.uint8:
            tensor = tensor.to(torch.float32) / 255.0
        return tensor

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
    """Convert supported image data to a PIL image.

    Parameters
    ----------
    image : PIL image, numpy.ndarray, LazyImage, or torch.Tensor
        Input image representation.

    Returns
    -------
    PIL.Image.Image
        Independent PIL image.

    Raises
    ------
    TypeError
        Image representation is unsupported.
ValueError
        Array shape is unsupported."""
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
                # Return a new LazyImage proxy without allocating RAM
                new_lazy = LazyImage(self)
                if self._preprocessing_ops:
                    from klygo.preprocessing.core import Operation

                    new_lazy._preprocessing_ops = list(self._preprocessing_ops) + [
                        Operation("pil_method", {"method": name, "args": args, "kwargs": kwargs}, "pil")
                    ]
                else:
                    new_lazy._instructions = list(self._instructions) + [(name, args, kwargs)]

                # Update size and mode immediately
                if name == "resize":
                    new_lazy._cached_size = kwargs.get("size", args[0] if args else self._cached_size)
                elif name == "convert":
                    new_lazy._cached_mode = kwargs.get("mode", args[0] if args else self._cached_mode)
                return new_lazy
            else:
                # Terminal executor (e.g., getpixel, histogram); releases RAM after execution
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

