from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import cv2 as cv
import numpy as np
from PIL import Image


_EXECUTORS: Dict[str, Callable[..., Any]] = {}


def _is_lazy_image(value: Any) -> bool:
    return value.__class__.__name__ == "LazyImage" and hasattr(value, "_processing_ops")


def _clone_lazy(image: Any) -> Any:
    from klygo.media import LazyImage

    return LazyImage(image)


def _to_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Image.Image):
        return np.asarray(value)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            if array.ndim == 4 and array.shape[0] == 1:
                array = array[0]
            if array.ndim == 3 and array.shape[0] in (1, 3, 4):
                array = np.transpose(array, (1, 2, 0))
            return array
    except ImportError:
        pass
    return np.asarray(value)


def _to_pil(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    array = _to_array(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype.kind == "f":
        maximum = float(np.nanmax(array)) if array.size else 0.0
        if maximum <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array)


@dataclass(frozen=True)
class Operation:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    backend: str = "auto"
    requires_pixels: bool = True
    changes_geometry: bool = False
    terminal: bool = False

    def __call__(self, image: Any) -> Any:
        if _is_lazy_image(image):
            result = _clone_lazy(image)
            result._processing_ops = list(image._processing_ops) + [self]
            width, height = image.size
            if result._processing_original_size is None:
                result._processing_original_size = (width, height)
            if self.name == "crop":
                left, top, right, bottom = self.params["box"]
                result._cached_size = (max(0, right - left), max(0, bottom - top))
            elif self.name == "resize" and self.params.get("keep_ratio"):
                target_width, target_height = self.params["size"]
                scale = min(target_width / max(1, width), target_height / max(1, height))
                result._cached_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            elif self.name in ("resize", "fit", "cover", "letterbox"):
                result._cached_size = tuple(self.params["size"])
            elif self.name in ("thumbnail", "contain"):
                target_width, target_height = self.params["size"]
                scale = min(target_width / max(1, width), target_height / max(1, height))
                result._cached_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            elif self.name == "center_crop":
                result._cached_size = tuple(self.params["size"])
            elif self.name == "pad":
                padding = self.params["padding"]
                if isinstance(padding, int):
                    left = top = right = bottom = padding
                elif len(padding) == 2:
                    left = right = padding[0]
                    top = bottom = padding[1]
                else:
                    left, top, right, bottom = padding
                result._cached_size = (width + left + right, height + top + bottom)
            elif self.name in ("affine", "perspective", "warp") and self.params.get("size"):
                result._cached_size = tuple(self.params["size"])
            elif self.name == "transpose" and self.params.get("method") in ("rotate_90", "rotate_270", "transpose", "transverse"):
                result._cached_size = (height, width)

            if self.name in ("grayscale",):
                result._cached_mode = "L" if self.params.get("channels", 1) == 1 else "RGB"
            elif self.name == "convert_color":
                target = self.params.get("target")
                result._cached_mode = "L" if target in ("GRAY", "L") else target
            return result
        value, _ = execute_pipeline(image, (self,))
        return value

    def to_dict(self) -> Dict[str, Any]:
        params = {}
        for key, value in self.params.items():
            if callable(value):
                params[key] = getattr(value, "__name__", repr(value))
            elif isinstance(value, Operation):
                params[key] = value.to_dict()
            elif isinstance(value, Compose):
                params[key] = value.to_dict()
            elif isinstance(value, np.ndarray):
                params[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                params[key] = [item.to_dict() if isinstance(item, Operation) else item for item in value]
            else:
                params[key] = value
        return {
            "name": self.name,
            "backend": self.backend,
            "requires_pixels": self.requires_pixels,
            "changes_geometry": self.changes_geometry,
            "terminal": self.terminal,
            "params": params,
        }


class Compose:
    def __init__(self, operations: Iterable[Union[Operation, "Compose"]]) -> None:
        flattened: List[Operation] = []
        for operation in operations:
            if isinstance(operation, Compose):
                flattened.extend(operation.operations)
            elif isinstance(operation, Operation):
                flattened.append(operation)
            else:
                raise TypeError(f"Expected Operation or Compose, got {type(operation).__name__}")
        self.operations = tuple(flattened)

    def __call__(self, image: Any) -> Any:
        result = image
        for operation in self.operations:
            result = operation(result)
        return result

    def __iter__(self) -> Iterator[Operation]:
        return iter(self.operations)

    def __len__(self) -> int:
        return len(self.operations)

    def to_dict(self) -> Dict[str, Any]:
        return {"operations": [operation.to_dict() for operation in self.operations]}

    def __repr__(self) -> str:
        names = ", ".join(operation.name for operation in self.operations)
        return f"Compose([{names}])"


class Metric:
    def __init__(self, name: str, evaluator: Callable[[Any], Any], params: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.evaluator = evaluator
        self.params = dict(params or {})

    def __call__(self, image: Any) -> Any:
        return self.evaluator(image)

    def _compare(self, operator: Callable[[Any, Any], bool], value: Any, symbol: str) -> "Predicate":
        return Predicate(lambda image: operator(self(image), value), f"{self.name} {symbol} {value}")

    def __ge__(self, value: Any) -> "Predicate":
        return self._compare(lambda a, b: a >= b, value, ">=")

    def __gt__(self, value: Any) -> "Predicate":
        return self._compare(lambda a, b: a > b, value, ">")

    def __le__(self, value: Any) -> "Predicate":
        return self._compare(lambda a, b: a <= b, value, "<=")

    def __lt__(self, value: Any) -> "Predicate":
        return self._compare(lambda a, b: a < b, value, "<")

    def between(self, minimum: Any, maximum: Any) -> "Predicate":
        return Predicate(lambda image: minimum <= self(image) <= maximum, f"{minimum} <= {self.name} <= {maximum}")


class Predicate:
    def __init__(self, evaluator: Callable[[Any], bool], description: str = "predicate") -> None:
        self.evaluator = evaluator
        self.description = description

    def __call__(self, image: Any) -> bool:
        return bool(self.evaluator(image))

    def __and__(self, other: "Predicate") -> "Predicate":
        return Predicate(lambda image: self(image) and other(image), f"({self.description} and {other.description})")

    def __or__(self, other: "Predicate") -> "Predicate":
        return Predicate(lambda image: self(image) or other(image), f"({self.description} or {other.description})")

    def __invert__(self) -> "Predicate":
        return Predicate(lambda image: not self(image), f"not ({self.description})")


def compose(*operations: Any) -> Compose:
    if len(operations) == 1 and isinstance(operations[0], (list, tuple)):
        operations = tuple(operations[0])
    return Compose(operations)


def apply(image: Any, operation_or_pipeline: Union[Operation, Compose]) -> Any:
    if not isinstance(operation_or_pipeline, (Operation, Compose)):
        raise TypeError("operation_or_pipeline must be an Operation or Compose")
    return operation_or_pipeline(image)


def register(name: str, executor: Callable[..., Any]) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    if not callable(executor):
        raise TypeError("executor must be callable")
    _EXECUTORS[name] = executor


def custom(function: Callable[[Any], Any], backend: str = "auto", name: Optional[str] = None, **kwargs: Any) -> Operation:
    if not callable(function):
        raise TypeError("function must be callable")
    return Operation(
        name=name or getattr(function, "__name__", "custom"),
        params={"function": function, "kwargs": kwargs},
        backend=backend,
    )


def inspect(pipeline: Union[Operation, Compose, Any]) -> List[Dict[str, Any]]:
    if _is_lazy_image(pipeline):
        operations = pipeline._processing_ops
    elif isinstance(pipeline, Operation):
        operations = (pipeline,)
    elif isinstance(pipeline, Compose):
        operations = pipeline.operations
    else:
        raise TypeError("inspect expects Operation, Compose, or LazyImage")
    return [operation.to_dict() for operation in operations]


def optimize(pipeline: Union[Operation, Compose]) -> Compose:
    operations = [pipeline] if isinstance(pipeline, Operation) else list(pipeline.operations)
    optimized: List[Operation] = []
    for operation in operations:
        if optimized and operation.name == "convert_color" and optimized[-1].name == "convert_color":
            optimized[-1] = operation
        elif optimized and operation.name == "resize" and optimized[-1].name == "resize":
            optimized[-1] = operation
        else:
            optimized.append(operation)
    return Compose(optimized)


def execute_operation(value: Any, operation: Operation, trace: List[Dict[str, Any]]) -> Any:
    if "function" in operation.params:
        return operation.params["function"](value, **operation.params.get("kwargs", {}))
    executor = _EXECUTORS.get(operation.name)
    if executor is None:
        from .transforms import execute_builtin

        executor = execute_builtin
    return executor(value, operation, trace)


def execute_pipeline(
    value: Any,
    operations: Sequence[Operation],
    trace: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Any, List[Dict[str, Any]]]:
    active_trace = trace if trace is not None else []
    result = value
    for operation in operations:
        result = execute_operation(result, operation, active_trace)
    return result, active_trace


def materialize(image: Any, output: str = "pil", cache: bool = False) -> Any:
    output = output.lower()
    if _is_lazy_image(image):
        if output == "pil":
            return image.to_pil(cache=cache)
        if output in ("array", "numpy"):
            return image.to_array(cache=cache)
        if output == "tensor":
            return image.to_tensor(cache=cache)
        if output == "native":
            return image._execute_processing(cache=cache)
        raise ValueError("output must be 'pil', 'array', 'tensor', or 'native'")

    operations: Sequence[Operation] = ()
    value, _ = execute_pipeline(image, operations)
    if output == "pil":
        return _to_pil(value)
    if output in ("array", "numpy"):
        return _to_array(value)
    if output == "tensor":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for tensor output") from exc
        array = np.array(_to_array(value), copy=True, order="C")
        tensor = torch.from_numpy(array)
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        return tensor
    if output == "native":
        return value
    raise ValueError("output must be 'pil', 'array', 'tensor', or 'native'")


def all_of(*conditions: Predicate) -> Predicate:
    return Predicate(lambda image: all(condition(image) for condition in conditions), "all_of")


def any_of(*conditions: Predicate) -> Predicate:
    return Predicate(lambda image: any(condition(image) for condition in conditions), "any_of")


def not_(condition: Predicate) -> Predicate:
    return ~condition


def between(metric: Metric, minimum: Any, maximum: Any) -> Predicate:
    return metric.between(minimum, maximum)


__all__ = [
    "Operation",
    "Compose",
    "Metric",
    "Predicate",
    "compose",
    "apply",
    "materialize",
    "inspect",
    "optimize",
    "custom",
    "register",
    "execute_pipeline",
    "all_of",
    "any_of",
    "not_",
    "between",
]
