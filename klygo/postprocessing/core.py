"""Composable in-place operations for detection results."""
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional


@dataclass(frozen=True)
class Operation:
    """Name one in-place result operation."""
    name: str
    function: Callable[[Any], None]
    _trusted: bool = False

    def __call__(self, result: Any) -> None:
        self.function(result)


class Pipeline:
    """Apply ordered operations to the same result object."""

    def __init__(self, operations: Iterable[Callable[[Any], None]]) -> None:
        items = []
        for operation in operations:
            if isinstance(operation, Pipeline):
                items.extend(operation.operations)
            elif callable(operation):
                items.append(operation)
            else:
                raise TypeError("Every postprocessing operation must be callable")
        self.operations = tuple(items)

    def __call__(self, result: Any) -> None:
        if not self.operations:
            return
        from klygo.outputs import Detection, Detections
        trusted = all(isinstance(operation, Operation) and operation._trusted for operation in self.operations)
        if isinstance(result, Detection):
            if trusted:
                for operation in self.operations:
                    operation(result)
                return
            draft = result._clone()
            for operation in self.operations:
                operation(draft)
            result._copy_from(draft, "pipeline")
            return
        if isinstance(result, Detections):
            if result.is_stream:
                def transform(frame: Detection) -> Detection:
                    self(frame)
                    return frame
                result._transform_stream(transform)
                return
            if trusted:
                with result._batch("pipeline"):
                    for operation in self.operations:
                        operation(result)
                return
            draft = Detections(
                [frame._clone() for frame in result],
                metadata=result.metadata,
                names=result._registry,
            )
            for operation in self.operations:
                operation(draft)
            same_order = len(result._frames) == len(draft._frames) and all(
                original.identity == staged.identity
                for original, staged in zip(result._frames, draft._frames)
            )
            same_frames = same_order and all(
                original._signature() == staged._signature()
                for original, staged in zip(result._frames, draft._frames)
            )
            same_registry = (
                result._registry == draft._registry
                and result._label_ids == draft._label_ids
                and result._next_class_id == draft._next_class_id
            )
            if same_frames and same_registry:
                return
            staged_by_identity = {frame.identity: frame for frame in draft._frames}
            for frame in result._frames:
                staged_frame = staged_by_identity.get(frame.identity)
                if staged_frame is None or frame._signature() != staged_frame._signature():
                    result._snapshot_frame_baseline(frame)
            result._registry = dict(draft._registry)
            result._label_ids = dict(draft._label_ids)
            result._next_class_id = draft._next_class_id
            staged = draft._frames
            for frame in staged:
                frame._owner = None
            result._commit(staged, "pipeline")
            draft._frames = []
            draft._identity_map = {}
            return
        for operation in self.operations:
            operation(result)

    def __iter__(self) -> Iterator[Callable[[Any], None]]:
        return iter(self.operations)

    def __len__(self) -> int:
        return len(self.operations)


def compose(*operations: Callable[[Any], None]) -> Pipeline:
    """Build an in-place pipeline."""
    if len(operations) == 1 and not callable(operations[0]):
        operations = tuple(operations[0])
    return Pipeline(operations)


def custom(function: Callable[[Any], None], name: Optional[str] = None) -> Operation:
    """Wrap a custom in-place operation."""
    if not callable(function):
        raise TypeError("function must be callable")
    return Operation(name or getattr(function, "__name__", "custom"), function)


__all__ = ["Operation", "Pipeline", "compose", "custom"]
