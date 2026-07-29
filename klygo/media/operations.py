import builtins
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional, Generator, Iterable

import cv2 as cv
import numpy as np
from PIL import Image

from klygo.utils.progress import ProgressBar
from klygo.validators import validate_type

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
# 1. Load / Save / Info
# =========================================================================

def _read_video_frames(
    path: Path,
    stream: bool = False,
    backend: str = "pil",
    verbose: bool = True,
) -> Union[List[Union[Image.Image, np.ndarray]], Generator[Union[Image.Image, np.ndarray], None, None]]:
    cap = cv.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {path}")

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

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
        return _frame_generator()
    else:
        return list(_frame_generator())


def load(
    source: Union[str, Path],
    recursive: bool = False,
    stream: bool = False,
    backend: str = "pil",
    verbose: bool = True,
) -> Union[List[Union[Image.Image, np.ndarray]], Generator[Union[Image.Image, np.ndarray], None, None]]:
    """
    Đọc 1 file ảnh, file video, hoặc thư mục chứa ảnh.
    - Hỗ trợ file ảnh: .png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff
    - Hỗ trợ file video: .mp4, .avi, .mov, .mkv, .m4v, .webm
    - Hỗ trợ thư mục chứa ảnh (recursive=True/False)
    - Tham số verbose (mặc định True) hiển thị thanh tiến trình ProgressBar.
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

    images: List[Union[Image.Image, np.ndarray]] = []
    with ProgressBar(total=len(image_paths), desc=f"Loading images from {p.name}", unit="file", verbose=verbose, colour="cyan") as pbar:
        for img_path in image_paths:
            if backend == "pil":
                with Image.open(img_path) as source_image:
                    images.append(source_image.convert("RGB"))
            else:
                image = cv.imread(str(img_path), cv.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not read image: {img_path}")
                images.append(image)
            pbar.update(1)

    return images


def save(
    path: Union[str, Path],
    image: Union[Image.Image, np.ndarray],
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Lưu ảnh (PIL Image hoặc NumPy array) ra đường dẫn file."""
    validate_type(path, (str, Path), "path")
    validate_type(overwrite, bool, "overwrite")
    validate_type(verbose, bool, "verbose")

    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {p}. Use overwrite=True to replace it.")

    p.parent.mkdir(parents=True, exist_ok=True)

    with ProgressBar(total=1, desc=f"Saving image {p.name}", unit="file", verbose=verbose, colour="cyan") as pbar:
        if isinstance(image, Image.Image):
            image.save(p)
        elif isinstance(image, np.ndarray):
            res = cv.imwrite(str(p), image)
            if not res:
                raise ValueError(f"Failed to save image to {p}")
        else:
            raise TypeError("image must be PIL.Image.Image or numpy.ndarray")
        pbar.update(1)

    return p


def save_video(
    output_path: Union[str, Path],
    frames: Iterable[Union[Image.Image, np.ndarray]],
    fps: float = 30.0,
    fourcc: str = "mp4v",
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Lưu danh sách hoặc generator các khung hình (frames) thành file video (.mp4, .avi...).
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
        if isinstance(f, Image.Image) or (f_arr.ndim == 3 and f_arr.shape[2] == 3):
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
    Lưu chuỗi ảnh/frames vào một thư mục với tên tăng dần (vd: frame_000001.jpg).
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
    Generator duyệt từng khung hình từ file video hoặc thư mục ảnh với tham số bước nhảy (sample_rate).
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


def info(path: Union[str, Path]) -> Dict[str, Any]:
    """Lấy thông tin chi tiết của 1 file ảnh hoặc video."""
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
        cap.release()

        return {
            "name": p.name,
            "path": p,
            "type": "video",
            "format": suf.lstrip("."),
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
    """Chuyển đổi hình ảnh (PIL Image, PyTorch Tensor, hoặc NumPy array) thành NumPy ndarray."""
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
    """Chuyển đổi hình ảnh (PIL Image hoặc NumPy array) thành PyTorch Tensor dạng (C, H, W)."""
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
    """Chuyển đổi hình ảnh (NumPy array hoặc PyTorch Tensor) thành PIL Image."""
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



