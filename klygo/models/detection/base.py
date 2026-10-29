"""Specialized base class for object detection tasks (`klygo.models.detection.base`).

LAYER 2: Task engine interface and implementation managing multi-source resolution
via media.load, batching loop, progress bar tracking, external timing, hardware dispatch,
export, and benchmark workflows.
"""

from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy, deepcopy
from collections import deque
import itertools
import math
import os
import statistics
from threading import RLock
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import PIL.Image

from klygo.models.base import BaseModel
from klygo.models import utils
from klygo import utils as klygo_utils
from klygo.models.backend import common
from klygo.outputs.detect import Box, Detection, Detections
from klygo.utils.progress import ProgressBar
from klygo import validators

__all__ = ["Detector"]

_UNSET = object()
_LOADING_OPTIONS = {
    "token", "use_auth_token", "trust_remote_code", "cache_dir", "revision",
    "local_files_only", "low_cpu_mem_usage", "device_map", "torch_dtype", "dtype",
}


def _materialize_model_image(image: Any, backend: Optional[str] = None) -> Any:
    """Decode one model input without replacing its lazy source proxy."""
    from klygo import media

    if isinstance(image, media.LazyImage):
        if (
            backend == "Ultralytics"
            and image.backend == "opencv"
            and not image._instructions
            and not image._preprocessing_ops
        ):
            return image.load(cache=False)
        image = image.to_pil(cache=False)
        if image.mode != "RGB":
            image = image.convert("RGB")
    return image


def _materialize_model_images(images: Sequence[Any], backend: Optional[str] = None) -> List[Any]:
    """Give model backends real RGB images without replacing lazy source proxies."""
    return [_materialize_model_image(image, backend=backend) for image in images]


def _prepared_batches(
    images: Any,
    batch: int,
    workers: int,
    prefetch: int,
    backend: Optional[str] = None,
):
    """Yield ordered source batches and decoded pixels with bounded read-ahead."""
    iterator = iter(images)

    def materialize(image: Any) -> Any:
        if backend == "Ultralytics":
            return _materialize_model_image(image, backend=backend)
        return _materialize_model_image(image)

    def parts():
        while True:
            part = list(itertools.islice(iterator, batch))
            if not part:
                return
            yield part

    batches = iter(parts())
    if workers == 0 or prefetch == 0:
        for part in batches:
            yield part, [materialize(image) for image in part]
        return

    first = next(batches, None)
    if first is None:
        return
    second = next(batches, None)
    if second is None:
        yield first, [materialize(image) for image in first]
        return
    batches = itertools.chain((first, second), batches)

    pending: deque[tuple[List[Any], List[Future[Any]]]] = deque()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="klygo-decode") as executor:
        def submit(part: List[Any]) -> tuple[List[Any], List[Future[Any]]]:
            return part, [executor.submit(materialize, image) for image in part]

        try:
            for _ in range(prefetch):
                part = next(batches, None)
                if part is None:
                    break
                pending.append(submit(part))
            while pending:
                part, futures = pending.popleft()
                prepared = [future.result() for future in futures]
                following = next(batches, None)
                if following is not None:
                    pending.append(submit(following))
                yield part, prepared
        finally:
            for _part, futures in pending:
                for future in futures:
                    future.cancel()


def _values_match(left: Any, right: Any) -> bool:
    """Compare configuration values without leaking array truth-value errors."""
    try:
        comparison = left == right
    except Exception:
        return False
    if isinstance(comparison, bool):
        return comparison
    try:
        reducer = getattr(comparison, "all", None)
        if callable(reducer):
            comparison = reducer()
        return bool(comparison)
    except (TypeError, ValueError, RuntimeError):
        return False


class Detector(BaseModel):
    """Backend-neutral detector wrapper for image and video prediction.

    Subclasses implement ``forward`` with shared option resolution, backend
    inference, and output-building helpers while inheriting the prediction lifecycle.
    """

    def __init__(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        backend: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            metadata=metadata,
            backend=backend,
            model_id=model_id,
            **kwargs,
        )
        self.model: Any = None
        self.names: Dict[int, Optional[str]] = {}
        self._predict_config = {"prompt": None}
        self._prediction_lock = RLock()

    # =========================================================================
    # 1. CORE INFERENCE API (predict, __call__, forward)
    # =========================================================================

    def predict(
        self,
        source: Any,
        prompt: Any = _UNSET,
        batch: int = 1,
        workers: int = 1,
        prefetch: int = 1,
        vid_stride: int = 1,
        max_frames: Optional[int] = None,
        verbose: bool = True,
        stream: bool = False,
        native: bool = False,
        packed: bool = False,
        **kwargs,
    ) -> Detections:
        """Predict with remembered inference options and independent result metadata.

        Omitted prompt reuses the previous prompt; explicit None clears it.
        Execution controls are never remembered. Lazy streams capture their
        configuration at invocation, including across interleaved predictions.
        """
        validators.validate_count(batch, "batch")
        validators.validate_count(workers, "workers", minimum=0)
        validators.validate_count(prefetch, "prefetch", minimum=0)
        validators.validate_count(vid_stride, "vid_stride")
        if max_frames is not None:
            validators.validate_count(max_frames, "max_frames")
        validators.validate_type(verbose, bool, "verbose")
        validators.validate_type(stream, bool, "stream")
        validators.validate_type(native, bool, "native")
        validators.validate_type(packed, bool, "packed")
        stream = stream or bool(getattr(source, "is_stream", False))
        if stream and packed:
            raise ValueError("stream and packed result storage are mutually exclusive")

        if native:
            if stream:
                raise ValueError(
                    "stream is unavailable in native mode; use native=False for Klygo managed stream, "
                    "or model-specific parameters (such as model_stream) if supported by the backend"
                )
            return self._predict_native(
                source,
                prompt=prompt,
                batch=batch,
                workers=workers,
                prefetch=prefetch,
                vid_stride=vid_stride,
                max_frames=max_frames,
                verbose=verbose,
                stream=stream,
                packed=packed,
                kwargs=kwargs,
            )

        with self._prediction_lock:
            metadata, prompts = self._resolve_prediction(prompt, kwargs)
            pixel_backend = "opencv" if self.backend == "Ultralytics" else "pil"
            images, is_single = utils.resolve_images(
                source,
                step=vid_stride,
                max_frames=max_frames,
                stream=stream,
                pixel_backend=pixel_backend,
            )
            source_type = getattr(images, "source_type", "image" if is_single else "frames")
            previous = self._snapshot_prediction_state()
            try:
                with self._use_metadata(metadata):
                    self._prepare_prediction(prompts)
                if not stream:
                    frames = []
                    with ProgressBar(total=len(images), desc="Predict", unit="frame", verbose=verbose, colour="cyan") as progress:
                        offset = 0
                        for part, prepared in _prepared_batches(
                            images, batch, workers, prefetch, backend=self.backend
                        ):
                            frames.extend(self._infer_batch(
                                part, prompts, metadata, offset * vid_stride,
                                vid_stride, prepared_images=prepared,
                            ))
                            progress.update(len(part))
                            offset += len(part)
                    result = Detections(
                        frames,
                        metadata={"source_type": source_type, "fps": getattr(images, "fps", 30.0)},
                        names=self.names,
                        packed=packed,
                    )
            except Exception:
                self._restore_prediction_state(previous)
                raise

            self._remember_prediction(metadata, prompts)
            if not stream:
                return result

            names = deepcopy(self.names)
            total = getattr(images, "total_frames", None)
            if total is None and hasattr(images, "__len__"):
                total = len(images)

            isolate_state = self._stream_state_isolation_required()
            prediction_snapshot = self._config_snapshot(metadata, prompts)

        def generate():
            offset = 0
            try:
                for part, prepared in _prepared_batches(
                    images, batch, workers, prefetch, backend=self.backend
                ):
                    def infer() -> List[Detection]:
                        if self.state == "UNLOADED":
                            from klygo.models.errors import InvalidStateError
                            raise InvalidStateError("Cannot consume a stream after model unload")
                        return self._infer_batch(
                            part, prompts, metadata, offset * vid_stride,
                            vid_stride, prepared_images=prepared,
                            prediction_snapshot=prediction_snapshot,
                        )

                    if isolate_state:
                        with self._prediction_lock:
                            current = self._snapshot_prediction_state()
                            try:
                                frames = infer()
                            finally:
                                self._restore_prediction_state(current)
                    else:
                        frames = infer()
                    yield from frames
                    offset += len(part)
            finally:
                close = getattr(images, "close", None)
                if callable(close):
                    close()

        return Detections(
            generate(),
            metadata={
                "source_type": source_type,
                "fps": getattr(images, "fps", 30.0),
                "requested_total_frames": total,
            },
            names=names,
            stream=True,
        )

    def _native_runtime_options(
        self,
        metadata: Dict[str, Any],
        prompt: Any,
    ) -> Dict[str, Any]:
        """Build backend-native runtime options for one prediction."""
        if self.backend == "Ultralytics":
            model_config = dict(metadata.get("model", {}))
            post_config = dict(metadata.get("post", {}))
            classes = model_config.pop("classes", None)
            if prompt and not hasattr(self, "original_names"):
                requested = [prompt] if isinstance(prompt, str) else list(prompt)
                unknown = [label for label in requested if label not in self.names.values()]
                if unknown:
                    raise ValueError(f"Unknown YOLO classes: {unknown}; use labels from model.names")
                class_ids = klygo_utils.invert(self.names)
                classes = [class_ids[name] for name in requested]

            post_model_opts = {
                k: v for k, v in post_config.items()
                if k in {"agnostic_nms", "max_det", "retina_masks", "nms"} and k not in ("conf", "iou", "threshold")
            }
            runtime = {**model_config, **post_model_opts}
            runtime["conf"] = float(post_config.get("threshold", 0.25))
            runtime["iou"] = float(post_config.get("iou", 0.7))
            if classes is not None:
                runtime["classes"] = classes
            return runtime
        return {}

    def _native_image(self, raw: Any, frame_index: int) -> Any:
        """Return source media used to attach identity to one native result."""
        from klygo import files, media

        path = getattr(raw, "path", None)
        if path is not None and files.exists(path):
            suffix = files.extension(path).lower()
            video_index = frame_index if suffix in media.VIDEO_SUFFIXES else None
            source = media.LazyImage(path, frame_index=video_index)
            shape = getattr(raw, "orig_shape", None)
            if shape is not None and len(shape) >= 2:
                source._cached_size = (int(shape[1]), int(shape[0]))
            return source
        image = getattr(raw, "orig_img", None)
        if image is None:
            raise ValueError("native backend result does not provide source image data")
        return media.to_pil(image)

    def _pack_native_frame(
        self,
        raw: Any,
        frame_index: int,
        metadata: Dict[str, Any],
        prompt: Any,
        prediction_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Detection:
        """Convert one backend-native result into a Klygo Detection."""
        formatted = self.format_results([raw])
        if len(formatted) != 1:
            raise ValueError("native backend must produce exactly one result per frame")
        speed = getattr(raw, "speed", {}) or {}
        latency = float(sum(float(value) for value in speed.values())) if isinstance(speed, Mapping) else 0.0
        frame = self._pack_detection(
            formatted[0],
            self._native_image(raw, frame_index),
            frame_index,
            latency,
            1000.0 / max(latency, 0.001),
        )
        frame.metadata.update({
            "native": True,
            "prompt": list(prompt) if prompt is not None else None,
            "predict_config": (
                deepcopy(prediction_snapshot)
                if prediction_snapshot is not None
                else self._config_snapshot(metadata, prompt)
            ),
            "model_id": self.model_id,
            "backend": self.backend,
            "source_path": frame.source_path,
            "frame_index": frame.frame_index,
            "image_frame_index": frame.frame_index,
        })
        return frame

    def _predict_native(
        self,
        source: Any,
        *,
        prompt: Any,
        batch: int,
        workers: int,
        prefetch: int,
        vid_stride: int,
        max_frames: Optional[int],
        verbose: bool,
        stream: bool,
        packed: bool,
        kwargs: Dict[str, Any],
    ) -> Detections:
        """Run a backend-owned input pipeline and normalize its output."""
        from klygo import files, media

        if stream:
            raise ValueError(
                "stream is unavailable in native mode; use native=False for Klygo managed stream, "
                "or model-specific parameters (such as model_stream) if supported by the backend"
            )

        if workers != 1 or prefetch != 1:
            raise ValueError("workers and prefetch are unavailable in native mode")

        locked = []
        if batch != 1:
            locked.append("batch")
        if vid_stride != 1:
            locked.append("vid_stride")
        if max_frames is not None:
            locked.append("max_frames")
        if locked:
            names = ", ".join(locked)
            raise ValueError(
                f"{names} are unavailable in native mode; use registered model parameters"
            )

        if self.backend != "Ultralytics":
            with self._prediction_lock:
                metadata, prompts = self._resolve_prediction(prompt, kwargs, runtime=False)
                _ = self._native_runtime_options(metadata, prompts)
                images, is_single = common._native_inputs(source)
                source_type = "image" if is_single else "frames"
                previous = self._snapshot_prediction_state()
                try:
                    with self._use_metadata(metadata):
                        self._prepare_prediction(prompts)
                        frames_list = self._infer_batch(
                            images,
                            prompts,
                            metadata,
                            prepared_images=images,
                        )
                        for frame in frames_list:
                            frame.metadata["native"] = True
                        result = Detections(
                            frames_list,
                            metadata={
                                "source_type": source_type,
                                "fps": getattr(images, "fps", 30.0),
                                "native": True,
                            },
                            names=self.names,
                            packed=packed,
                        )
                except Exception:
                    self._restore_prediction_state(previous)
                    raise
                self._remember_prediction(metadata, prompts)
                return result

        if isinstance(source, media.MediaFrames):
            raise TypeError("native prediction requires a backend-supported source, not MediaFrames")

        with self._prediction_lock:
            metadata, prompts = self._resolve_prediction(prompt, kwargs, runtime=False)
            previous = self._snapshot_prediction_state()
            try:
                with self._use_metadata(metadata):
                    self._prepare_prediction(prompts)
                    runtime_options = self._native_runtime_options(
                        metadata,
                        prompts,
                    )
                    runtime_options.setdefault("batch", 1)
                    runtime_options.setdefault("vid_stride", 1)
                    # Model parameter `stream` can only be set via prefixed syntax (e.g. model_stream=True).
                    # The invocation control `stream` of predict() is never passed down to the model.
                    raw = self.model.predict(
                        source=source,
                        verbose=verbose,
                        **runtime_options,
                    )
                    names = dict(getattr(self.model, "names", self.names))
                    self.names = names
            except Exception:
                self._restore_prediction_state(previous)
                raise
            self._remember_prediction(metadata, prompts)
            isolate_native_stream = self._stream_state_isolation_required()
            is_generator = hasattr(raw, "__next__") or (hasattr(raw, "__iter__") and not isinstance(raw, (list, tuple)))
            prediction_snapshot = self._config_snapshot(metadata, prompts)

        def frames():
            iterator = iter(raw)
            index = 0
            try:
                while True:
                    if is_generator and isolate_native_stream:
                        with self._prediction_lock:
                            current = self._snapshot_prediction_state()
                            try:
                                with self._use_metadata(metadata):
                                    self._prepare_prediction(prompts)
                                    native_result = next(iterator)
                            except StopIteration:
                                break
                            finally:
                                self._restore_prediction_state(current)
                    else:
                        try:
                            native_result = next(iterator)
                        except StopIteration:
                            break
                    yield self._pack_native_frame(
                        native_result,
                        index,
                        metadata,
                        prompts,
                        prediction_snapshot,
                    )
                    index += 1
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()

        source_type = "video" if (
            isinstance(source, int)
            or (
                isinstance(source, (str, bytes, os.PathLike))
                and files.extension(source).lower() in media.VIDEO_SUFFIXES
            )
        ) else "folder" if (
            isinstance(source, (str, os.PathLike)) and files.is_dir(source)
        ) else "image"
        result_metadata = {
            "source_type": source_type,
            "fps": 30.0,
            "native": True,
            "requested_total_frames": None,
        }
        materialized = list(frames())
        return Detections(
            materialized,
            metadata=result_metadata,
            names=names,
            packed=packed,
        )

    def __call__(self, source: Any, *args, **kwargs) -> Union[Detections, Any]:
        """Run :meth:`predict` using callable model syntax."""
        return self.predict(source, *args, **kwargs)

    def forward(
        self,
        images: List[PIL.Image.Image],
        prompt: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Detection]:
        """Execute model forward pass. Subclasses implement this method."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement 'forward(self, images, prompt=None)'"
        )

    # =========================================================================
    # 2. PROPERTIES, LIFECYCLE & BENCHMARK
    # =========================================================================

    @property
    def device(self) -> Any:
        """Return the backend's current device without caching or coercion."""
        return self.current_device()

    @property
    def dtype(self) -> Any:
        """Return the backend's current dtype (for example, torch.dtype or str)."""
        return self.current_dtype()

    @property
    def predict_config(self) -> Dict[str, Any]:
        """Return an independent snapshot of defaults and remembered overrides."""
        with self._prediction_lock:
            metadata, prompt = self._resolve_prediction(_UNSET, {}, validate=False)
            return self._config_snapshot(metadata, prompt)

    def reset(self) -> "Detector":
        """Restore default settings and prompt state, and reset backend resources."""
        with self._prediction_lock:
            self._reset_prediction_state()
            common.reset(self.backend, self.model, getattr(self, "processor", None))
            self._settings = deepcopy(self._default_settings)
            self._predict_config = {"prompt": None}
            if self.state != "UNLOADED":
                self.state = "READY"
        return self

    def warmup(self) -> None:
        """Warm up with current settings without remembering temporary changes."""
        dummy_img = PIL.Image.new("RGB", (640, 640), color=(100, 100, 100))
        self.benchmark(dummy_img, iterations=1, warmup=0, verbose=False)

    def clear_cache(self) -> None:
        """Release unused backend memory blocks."""
        common.clear_cache(self.backend)

    def save(self, output_dir: str) -> None:
        """Serialise the model (weights + metadata) to a Klygo-standard directory."""
        export_metadata = dict(self.metadata)
        export_metadata["model_id"] = self.model_id
        export_metadata["flags"] = dict(
            utils.normalize_flags(export_metadata.get("flags", self.flags))
        )
        export_metadata.setdefault("priority", None)
        export_metadata.setdefault("details", None)
        if not export_metadata.get("class"):
            export_metadata["class"] = self.class_name

        common.save(
            backend=self.backend,
            model=self.model,
            output_dir=output_dir,
            metadata=export_metadata,
            class_module=self.__class__.__module__,
            processor=getattr(self, "processor", None),
        )
        if hasattr(self, "tokenizer") and self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(output_dir)

    def unload(self) -> None:
        """Release GPU/CPU memory and transition model state to ``UNLOADED``."""
        common.unload(self.backend, self.model, getattr(self, "processor", None))
        if hasattr(self, "tokenizer"):
            del self.tokenizer
            self.tokenizer = None
        if hasattr(self, "processor"):
            del self.processor
            self.processor = None
        if hasattr(self, "model"):
            del self.model
            self.model = None
        self.state = "UNLOADED"

    def benchmark(
        self,
        source: Optional[Any] = None,
        prompt: Any = _UNSET,
        iterations: int = 20,
        warmup: int = 5,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Measure prepared-input pipeline latency without persisting overrides."""
        validators.validate_count(iterations, "iterations")
        validators.validate_count(warmup, "warmup", minimum=0)
        validators.validate_type(verbose, bool, "verbose")
        if any(key in kwargs for key in ("stream", "batch", "workers", "prefetch", "vid_stride", "max_frames")):
            raise ValueError("benchmark accepts inference options, not stream or sampling controls")

        img = source if source is not None else PIL.Image.new("RGB", (640, 640))
        if hasattr(img, "__next__") or getattr(img, "is_stream", False):
            raise ValueError("Materialize a finite benchmark source before calling benchmark")

        with self._prediction_lock:
            metadata, prompts = self._resolve_prediction(prompt, kwargs)
            pixel_backend = "opencv" if self.backend == "Ultralytics" else "pil"
            images, _ = utils.resolve_images(img, stream=False, pixel_backend=pixel_backend)
            if not images:
                raise ValueError("benchmark source must contain at least one image")
            cur_dev, cur_dt = self.device, self.dtype
            latencies = []
            with self._isolated_prediction_state():
                for _ in range(warmup):
                    self._infer_batch(images, prompts, metadata)
                for _ in range(iterations):
                    common.synchronize(self.backend, device=cur_dev)
                    start = time.perf_counter()
                    result = self._infer_batch(images, prompts, metadata)
                    common.synchronize(self.backend, device=cur_dev, value=result)
                    latencies.append(time.perf_counter() - start)

            average = statistics.mean(latencies)
            report = {
                "model_id": self.model_id,
                "backend": self.backend,
                "device": str(cur_dev),
                "dtype": str(cur_dt).replace("torch.", ""),
                "image_size": f"{images[0].width}x{images[0].height}",
                "iterations": iterations,
                "warmup": warmup,
                "latency_avg_ms": round(average * 1000, 2),
                "latency_median_ms": round(statistics.median(latencies) * 1000, 2),
                "latency_p95_ms": round(sorted(latencies)[math.ceil(0.95 * iterations) - 1] * 1000, 2),
                "fps": round(len(images) / average, 1) if average else 0.0,
                "images_per_iteration": len(images),
                "measurement_scope": "prepared-input pipeline per iteration; media resolution excluded",
                "throughput_unit": "images/second",
                "predict_config": self._config_snapshot(metadata, prompts),
            }
            if verbose:
                from rich.console import Console
                from rich.table import Table
                table = Table(title=f"Model Benchmark: {self.model_id}")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                for key, value in report.items():
                    table.add_row(key, str(value))
                Console().print(table)
            return report

    def help(self) -> None:
        """Display standardized usage guide and parameters for the model."""
        from rich.console import Console
        from rich.table import Table

        table = Table(title=f"Model Help & Usage: {self.model_id}")
        table.add_column("Command / Method", style="cyan", no_wrap=True)
        table.add_column("Description", style="green")
        table.add_row("predict(source, prompt, ...)", "Perform detection on images, videos, directories, or streams.")
        table.add_row("benchmark(iterations=20, warmup=5)", "Evaluate inference latency (ms) and throughput (FPS).")
        table.add_row("info()", "Print architecture, backend, state, and active settings in a table.")

        settings_dict = self.settings
        has_params = any(bool(v) for v in settings_dict.values()) if isinstance(settings_dict, dict) else False
        if has_params:
            param_lines = [f"{k}={v} (group: {g})" for g, val in settings_dict.items() if isinstance(val, dict) for k, v in val.items()]
            if param_lines:
                table.add_row("Custom Parameters", "\n".join(param_lines))
        Console().print(table)

    # =========================================================================
    # 3. BACKEND EXECUTION DELEGATES
    # =========================================================================

    def current_device(self) -> Any:
        """Return the actual device hosting the wrapped model."""
        return common.current_device(self.model, backend=self.backend)

    def current_dtype(self) -> Any:
        """Return the actual dtype of the wrapped model's parameters."""
        return common.current_dtype(self.model, backend=self.backend)

    def cast_inputs(self, inputs: Any) -> Any:
        """Cast floating inputs to the model's device and dtype via the active backend."""
        return common.cast_inputs(self.backend, inputs, dev=self.current_device(), dtype=self.current_dtype())

    def run_inference(self, inputs: Any, **model_kwargs) -> Any:
        """Execute a forward pass using the native runtime of the selected backend."""
        if common.backend_family(self.backend) == "Ultralytics":
            options = dict(model_kwargs)
            model_kwargs["_reuse_predictor"] = (
                getattr(self, "_prepared_predict_options", None) == options
            )
            outputs = common.run_inference(
                self.backend,
                self.model,
                inputs,
                cur_dtype=self.current_dtype(),
                **model_kwargs,
            )
            self._prepared_predict_options = options
            return outputs
        return common.run_inference(
            self.backend,
            self.model,
            inputs,
            cur_dtype=self.current_dtype(),
            **model_kwargs,
        )

    def get_output_device(self, outputs: Any) -> Any:
        """Detect the device of the model's output tensors."""
        return common.get_output_device(self.backend, outputs, default_device=self.current_device())

    def sync_device(self, tensor: Any, target_device: Any) -> Any:
        """Move a tensor to target_device when it differs from the current device."""
        return common.sync_device(self.backend, tensor, target_device=target_device)

    def format_results(self, raw_outputs: Any) -> List[Dict[str, Any]]:
        """Normalise raw backend output into a list of {boxes, scores, labels} dicts."""
        return common.format_results(self.backend, raw_outputs)

    # =========================================================================
    # 4. INTERNAL STATE & BATCH INFERENCE HELPERS
    # =========================================================================

    def _infer_batch(self, images: List[Any], prompt: Any, metadata: Dict[str, Any],
                     offset: int = 0, frame_step: int = 1,
                     prepared_images: Optional[List[Any]] = None,
                     prediction_snapshot: Optional[Dict[str, Any]] = None) -> List[Detection]:
        with self._prediction_lock, self._use_metadata(metadata), common.inference_context(self.backend):
            self._prepare_prediction(prompt)
            common._optimize_model(self.backend, self.model)
            start = time.perf_counter()
            prepared = (
                prepared_images
                if prepared_images is not None
                else _materialize_model_images(images, backend=self.backend)
            )
            if len(prepared) != len(images):
                raise ValueError("prepared image count must match source image count")
            raw = list(self.forward(images=prepared, prompt=deepcopy(prompt)))
            elapsed = (time.perf_counter() - start) * 1000 / max(1, len(images))
            if len(raw) != len(images):
                raise ValueError("forward() must return exactly one result per input image")

            frames = []
            lat_ms = round(elapsed, 2)
            fps_val = round(1000 / max(elapsed, 0.001), 1)
            base_post = dict(metadata.get("post", {}))
            pred_snapshot = (
                prediction_snapshot
                if prediction_snapshot is not None
                else self._config_snapshot(metadata, prompt)
            )
            for index, (result, image) in enumerate(zip(raw, images)):
                validators.validate_type(result, (dict, Detection), "forward result")
                frame = self._pack_detection(result, image, offset + index * frame_step, lat_ms, fps_val)
                config = dict(base_post)
                config.update(frame.metadata)
                config.update({
                    "prompt": prompt,
                    "predict_config": pred_snapshot,
                    "model_id": self.model_id,
                    "backend": self.backend,
                    "source_path": frame.source_path,
                    "frame_index": frame.frame_index,
                    "image_frame_index": frame.frame_index,
                })
                frame.metadata = deepcopy(config)
                frames.append(frame)
            return frames

    def _pack_detection(self, det: Any, image: Any, frame_index: int, lat_ms: float, fps_val: float) -> Detection:
        """Normalise one raw result (dict or Detection) into a fully-populated Detection."""
        import numpy as np
        from klygo import media, outputs, preprocessing
        img_path = str(image.path) if isinstance(image, media.LazyImage) else getattr(image, "path", getattr(image, "filename", None))
        final_image = image if isinstance(image, media.LazyImage) else (media.LazyImage(img_path) if img_path is not None else image)
        actual_idx = getattr(final_image, "frame_index", None)
        actual_idx = int(frame_index if actual_idx is None else actual_idx)

        trace = preprocessing.get_trace(image)
        original_size = getattr(image, "_preprocessing_original_size", None)
        if trace and original_size is not None and getattr(image, "path", None) is not None:
            final_image = media.LazyImage(image.path, frame_index=getattr(image, "frame_index", None))
        target_size = original_size if (trace and original_size is not None) else getattr(final_image, "size", None)

        if isinstance(det, dict):
            b_list, s_list = det.get("boxes", []), det.get("scores", [1.0] * len(det.get("boxes", [])))
            l_list, c_list = det.get("labels", ["object"] * len(b_list)), det.get("class_ids", range(len(b_list)))
            if trace:
                b_list = preprocessing.restore_boxes(b_list, trace)
            det = outputs.build_detections(
                [final_image],
                [{"boxes": b_list, "scores": list(s_list), "labels": list(l_list), "class_ids": list(c_list)}],
                names=self.names,
            )[0]
            if trace:
                det._mark_restored()
            det._frame_index = actual_idx
            det._identity = (det._source_id, actual_idx)
            if target_size is not None:
                det._image_size = target_size
        else:
            fresh_output = bool(det.__dict__.pop("_klygo_fresh_output", False))
            if det._owner is not None or not fresh_output:
                det = det._clone()
            det._source_path = str(img_path) if img_path is not None else det._source_path
            det._image_size = target_size if target_size is not None else final_image.size
            det._source = final_image
            det._source_id, _ = outputs.detect._source(final_image, None)
            det._frame_index = actual_idx
            det._identity = (det._source_id, actual_idx)
            if trace:
                if len(det._labels) > 0:
                    restored = preprocessing.restore_boxes(det.boxes, trace)
                    dtype = np.float32 if det.dtype == 32 else np.float64
                    det._xyxy = np.array(restored, dtype=dtype)
                    if det._boxes_cache is not None:
                        for i, b in enumerate(det._boxes_cache):
                            if b is not None:
                                row = det._xyxy[i]
                                b._box = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                det._mark_restored()
        det._source_path = str(getattr(final_image, "path", getattr(final_image, "url", img_path))) if (img_path or hasattr(final_image, "url")) else None
        det.metadata.update({"speed": {"inference": lat_ms, "fps": fps_val}})
        return det

    def _stream_state_isolation_required(self) -> bool:
        """Return whether each lazy batch must restore mutable backend state."""
        family = common.backend_family(self.backend)
        if family == "Ultralytics":
            return hasattr(self, "original_names")
        if family == "Torchvision" or family in common.KERAS_BACKENDS:
            return False
        return True

    def _resolve_prediction(self, prompt: Any, kwargs: Dict[str, Any], *, validate: bool = True, runtime: bool = True) -> Any:
        base = deepcopy(dict(self._metadata))
        base.update(deepcopy(self.settings))
        saved = {k: v for k, v in self._predict_config.items() if k != "prompt"}
        base = utils.resolve_metadata(base, saved, runtime=True)
        resolved = utils.resolve_metadata(base, kwargs, runtime=runtime)
        effective_prompt = deepcopy(self._predict_config["prompt"]) if prompt is _UNSET else self._normalize_prediction_prompt(prompt)
        if validate:
            self._validate_prediction(resolved, effective_prompt)
        return resolved, effective_prompt

    def _remember_prediction(self, metadata: Dict[str, Any], prompt: Any) -> None:
        base = deepcopy(dict(self._metadata))
        base.update(deepcopy(self.settings))
        defaults = utils.parameter_groups(base)
        remembered = {"prompt": deepcopy(prompt)}
        for group in utils.runtime_groups(metadata):
            changed = {
                k: deepcopy(v) for k, v in metadata[group].items()
                if not (k in defaults[group] and _values_match(v, defaults[group][k]))
            }
            if changed:
                remembered[group] = changed
        self._predict_config = remembered

    def _config_snapshot(self, metadata: Dict[str, Any], prompt: Any) -> Dict[str, Any]:
        groups = utils.parameter_groups(metadata)
        priority = utils.normalize_priority(groups, metadata.get("priority"))
        result = {"prompt": deepcopy(prompt)}
        for group, values in groups.items():
            for key, value in values.items():
                if group == "model" and key in _LOADING_OPTIONS:
                    continue
                name = key if key in priority.get(group, ()) else f"{group}_{key}"
                result[name] = deepcopy(value)
        return result

    def _normalize_prediction_prompt(self, prompt: Any) -> Optional[List[str]]:
        if prompt is None:
            return None
        return utils.normalize_prompt(prompt)

    def _validate_prediction(self, metadata: Dict[str, Any], prompt: Any) -> None:
        for values in utils.parameter_groups(metadata).values():
            for name in ("threshold", "iou", "text_threshold"):
                if name in values:
                    value = values[name]
                    validators.validate_type(value, (int, float), name)
                    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"{name} must be a finite number between 0 and 1")
            for name in ("source", "verbose", "batch", "vid_stride", "max_frames", "native", "packed", "output_path"):
                if name in values:
                    raise ValueError(f"{name} is an execution control, not a persistent inference option")

    def _snapshot_prediction_state(self) -> Dict[str, Any]:
        state = {
            name: copy(getattr(self, name))
            for name in ("names", "_active_prompt")
            if hasattr(self, name)
        }
        if hasattr(getattr(self, "model", None), "names"):
            state["native_names"] = dict(self.model.names)
        return state

    def _restore_prediction_state(self, state: Dict[str, Any]) -> None:
        for name, value in state.items():
            if name != "native_names":
                setattr(self, name, copy(value))
        if "native_names" in state and hasattr(getattr(self, "model", None), "set_classes"):
            native = state["native_names"]
            if dict(self.model.names) != native:
                self.model.set_classes(list(native.values()))

    @contextmanager
    def _isolated_prediction_state(self):
        state = self._snapshot_prediction_state()
        try:
            yield
        finally:
            try:
                self._restore_prediction_state(state)
            except Exception as error:
                raise RuntimeError("Failed to restore backend inference state") from error

    def _reset_prediction_state(self) -> None:
        """Restore wrapper-specific prompt state without requiring a valid query."""
        if hasattr(self, "original_names") and hasattr(getattr(self, "model", None), "set_classes"):
            self.model.set_classes(list(self.original_names.values()))
            self.names = dict(self.original_names)
        elif self.backend == "Hugging Face":
            self.names = {}
            self._active_prompt = None

    def _prepare_prediction(self, prompt: Optional[List[str]]) -> None:
        """Prepare backend-specific class state for one prediction call."""
        if hasattr(getattr(self, "model", None), "set_classes") and hasattr(self, "original_names"):
            classes = list(self.original_names.values()) if prompt is None else list(prompt)
            if list(self.model.names.values()) != classes:
                self.model.set_classes(classes)
            self.names = dict(self.model.names)
        elif self.backend == "Hugging Face" and prompt is not None:
            if prompt != getattr(self, "_active_prompt", None):
                self.names = {index: name for index, name in enumerate(prompt)}
                self._active_prompt = list(prompt)

    @staticmethod
    def _validate_count(value: Any, name: str, minimum: int = 1) -> None:
        validators.validate_count(value, name, minimum)
