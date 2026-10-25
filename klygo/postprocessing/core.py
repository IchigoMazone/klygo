"""Composable in-place operations for detection results."""
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional

@dataclass(frozen=True)
class Operation:
    """Name one in-place result operation."""
    name: str
    function: Callable[[Any], None]
    def __call__(self, result: Any) -> None:
        self.function(result)

class Pipeline:
    """Apply ordered operations to the same result object."""
    def __init__(self, operations: Iterable[Callable[[Any], None]]) -> None:
        items = []
        for operation in operations:
            if isinstance(operation, Pipeline): items.extend(operation.operations)
            elif callable(operation): items.append(operation)
            else: raise TypeError("Every postprocessing operation must be callable")
        self.operations = tuple(items)
    def __call__(self, result: Any) -> None:
        if not self.operations:
            return
        from klygo.outputs import Detection, Detections
        if isinstance(result, Detection):
            draft = result._clone()
            for operation in self.operations:
                operation(draft)
            result._registry = dict(draft._registry)
            result._commit([box._clone() for box in draft], "pipeline")
            return
        if isinstance(result, Detections):
            draft = Detections(
                [frame._clone() for frame in result],
                metadata=result.metadata,
                names=result._registry,
            )
            for operation in self.operations:
                operation(draft)
            result._registry = dict(draft._registry)
            result._commit([frame._clone() for frame in draft], "pipeline")
            return
        for operation in self.operations:
            operation(result)
    def __iter__(self) -> Iterator[Callable[[Any], None]]: return iter(self.operations)
    def __len__(self) -> int: return len(self.operations)

def compose(*operations: Callable[[Any], None]) -> Pipeline:
    """Build an in-place pipeline."""
    if len(operations) == 1 and not callable(operations[0]): operations = tuple(operations[0])
    return Pipeline(operations)

def apply(result: Any, *operations: Callable[[Any], None]) -> None:
    """Apply operations to one result object."""
    compose(*operations)(result)

def custom(function: Callable[[Any], None], name: Optional[str] = None) -> Operation:
    """Wrap a custom in-place operation."""
    if not callable(function): raise TypeError("function must be callable")
    return Operation(name or getattr(function, "__name__", "custom"), function)

__all__ = ["Operation", "Pipeline", "compose", "apply", "custom"]
