"""Composable, lazy-safe operations for normalized Klygo detection results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional


@dataclass(frozen=True)
class Operation:
    """A named transformation applied to one ``Detection``."""

    name: str
    function: Callable[[Any], Any]

    def __call__(self, result: Any) -> Any:
        return self.function(result)

    def __repr__(self) -> str:
        return f"Operation({self.name})"


class Pipeline:
    """Ordered result transformations; video streams remain lazy."""

    def __init__(self, operations: Iterable[Callable[[Any], Any]]) -> None:
        flattened = []
        for operation in operations:
            if isinstance(operation, Pipeline):
                flattened.extend(operation.operations)
            elif callable(operation):
                flattened.append(operation)
            else:
                raise TypeError("Every postprocessing operation must be callable")
        self.operations = tuple(flattened)

    def _apply_detection(self, result: Any) -> Any:
        current = result
        for operation in self.operations:
            current = operation(current)
        return current

    def __call__(self, result: Any) -> Any:
        from klygo.outputs.detect import Detections

        if isinstance(result, Detections):
            return result.map(self._apply_detection)
        return self._apply_detection(result)

    def __iter__(self) -> Iterator[Callable[[Any], Any]]:
        return iter(self.operations)

    def __len__(self) -> int:
        return len(self.operations)

    def __repr__(self) -> str:
        names = [getattr(item, "name", getattr(item, "__name__", type(item).__name__)) for item in self.operations]
        return f"Pipeline({', '.join(names)})"


def compose(*operations: Callable[[Any], Any]) -> Pipeline:
    """Build a reusable pipeline from variadic callables or one iterable."""
    if len(operations) == 1 and not callable(operations[0]):
        operations = tuple(operations[0])
    return Pipeline(operations)


def apply(result: Any, *operations: Callable[[Any], Any]) -> Any:
    """Apply one or more operations to ``Detection`` or ``Detections``."""
    if not operations:
        return result
    return compose(*operations)(result)


def custom(function: Callable[[Any], Any], name: Optional[str] = None) -> Operation:
    """Wrap an ordinary Python callable as a named operation."""
    if not callable(function):
        raise TypeError("function must be callable")
    return Operation(name or getattr(function, "__name__", "custom"), function)


__all__ = ["Operation", "Pipeline", "compose", "apply", "custom"]
