"""Three-level object-detection result containers."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha1
import math
import uuid
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

from klygo import files, validators

Names = dict[int, str]
Identity = tuple[str, int]


def _label(value: Any) -> str:
    result = "" if value is None else str(value).strip()
    if not result:
        raise ValueError("label must be a non-empty string")
    return result


def _score(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("score must be a finite number between 0 and 1")
    return result


def _coords(value: Iterable[Any]) -> tuple[float, float, float, float]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise TypeError("box must contain four numeric coordinates") from error
    if len(result) != 4:
        raise ValueError("box must contain exactly four coordinates")
    if not all(math.isfinite(item) for item in result):
        raise ValueError("box coordinates must be finite")
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("box must have positive width and height")
    return result


def _size(value: Any) -> tuple[int, int]:
    try:
        width, height = value
    except (TypeError, ValueError) as error:
        raise TypeError("image_size must contain width and height") from error
    if isinstance(width, bool) or isinstance(height, bool):
        raise TypeError("image_size values must be integers")
    result = int(width), int(height)
    if min(result) <= 0:
        raise ValueError("image_size values must be positive")
    return result


def _names(value: Optional[Mapping[int, Any] | Sequence[Any]]) -> Names:
    if value is None:
        return {}
    items = value.items() if isinstance(value, Mapping) else enumerate(value)
    result = {}
    for key, value in items:
        if isinstance(key, bool) or int(key) < 0:
            raise ValueError("class IDs must be non-negative integers")
        result[int(key)] = _label(value)
    return result


def _source(source: Any, source_id: Optional[str]) -> tuple[str, Optional[str]]:
    path = getattr(source, "path", None) or getattr(source, "filename", None)
    source_path = str(path) if path else None
    if source_id is not None:
        result = str(source_id).strip()
        if not result:
            raise ValueError("source_id must be a non-empty string")
        return result, source_path
    if source_path:
        return files.as_posix(files.resolve(source_path)), source_path
    return f"memory:{uuid.uuid4().hex}", None


def _list(value: Any, name: str) -> list[Any]:
    for method in ("detach", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        return list(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence") from error


class Box:
    """Represent one mutable object annotation with stable frame-local identity."""

    def __init__(self, box: Iterable[Any], *, label: str, score: float = 1.0,
                 class_id: Optional[int] = None, uid: Optional[int] = None,
                 track_id: Optional[int | str] = None,
                 metadata: Optional[Mapping[str, Any]] = None) -> None:
        self._owner: Optional[Detection] = None
        self._uid = self._integer(uid, "uid")
        self._class_id = self._integer(class_id, "class_id")
        self._label = _label(label)
        self._score = _score(score)
        self._box = _coords(box)
        self._track_id = self._track(track_id)
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")
        self._metadata = deepcopy(dict(metadata or {}))

    @staticmethod
    def _integer(value: Optional[int], name: str) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer or None")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @staticmethod
    def _track(value: Optional[int | str]) -> Optional[int | str]:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, str))):
            raise TypeError("track_id must be an integer, string, or None")
        return value

    def _notify(self, field: str, before: Any, after: Any) -> None:
        if self._owner is not None and before != after:
            self._owner._box_changed(self, field, before, after)

    def _clone(self) -> "Box":
        return Box(self.box, label=self.label, score=self.score, class_id=self._class_id,
                   uid=self._uid, track_id=self.track_id, metadata=self.metadata)

    @property
    def uid(self) -> int:
        """Return the stable frame-local UID."""
        if self._uid is None:
            raise RuntimeError("box UID has not been assigned by a Detection")
        return self._uid

    @property
    def label(self) -> str:
        """Return the read-only class label."""
        return self._label

    @property
    def class_id(self) -> int:
        """Return the internal class ID."""
        if self._class_id is None:
            raise RuntimeError("box class ID has not been assigned by a Detection")
        return self._class_id

    @class_id.setter
    def class_id(self, value: int) -> None:
        value = self._integer(value, "class_id")
        assert value is not None
        if self._owner is not None and self._owner._registry.get(value) != self.label:
            raise ValueError("class_id must map to the box label in the owning Detection")
        before, self._class_id = self._class_id, value
        self._notify("class_id", before, value)

    @property
    def score(self) -> float:
        """Return the confidence score."""
        return self._score

    @score.setter
    def score(self, value: float) -> None:
        value = _score(value)
        before, self._score = self._score, value
        self._notify("score", before, value)

    @property
    def box(self) -> tuple[float, float, float, float]:
        """Return immutable XYXY coordinates."""
        return self._box

    @box.setter
    def box(self, value: Iterable[Any]) -> None:
        value = _coords(value)
        before, self._box = self._box, value
        self._notify("box", before, value)

    @property
    def track_id(self) -> Optional[int | str]:
        """Return the optional tracking identity."""
        return self._track_id

    @track_id.setter
    def track_id(self, value: Optional[int | str]) -> None:
        value = self._track(value)
        before, self._track_id = self._track_id, value
        self._notify("track_id", before, value)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return box metadata."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("metadata must be a mapping")
        value = deepcopy(dict(value))
        before, self._metadata = self._metadata, value
        self._notify("metadata", before, value)

    xmin = property(lambda self: self.box[0], doc="Left coordinate.")
    ymin = property(lambda self: self.box[1], doc="Top coordinate.")
    xmax = property(lambda self: self.box[2], doc="Right coordinate.")
    ymax = property(lambda self: self.box[3], doc="Bottom coordinate.")
    width = property(lambda self: self.xmax - self.xmin, doc="Box width.")
    height = property(lambda self: self.ymax - self.ymin, doc="Box height.")
    area = property(lambda self: self.width * self.height, doc="Box area.")
    aspect_ratio = property(lambda self: self.width / self.height, doc="Width divided by height.")
    center_x = property(lambda self: (self.xmin + self.xmax) / 2, doc="Horizontal center.")
    center_y = property(lambda self: (self.ymin + self.ymax) / 2, doc="Vertical center.")
    center = property(lambda self: (self.center_x, self.center_y), doc="Box center.")
    size = property(lambda self: (self.width, self.height), doc="Box size.")

    def translate(self, dx: float = 0, dy: float = 0) -> None:
        """Move coordinates by a finite offset."""
        dx, dy = float(dx), float(dy)
        if not math.isfinite(dx) or not math.isfinite(dy):
            raise ValueError("dx and dy must be finite")
        self.box = self.xmin + dx, self.ymin + dy, self.xmax + dx, self.ymax + dy

    def scale(self, factor: float) -> None:
        """Scale coordinates around the box center."""
        factor = float(factor)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("factor must be a positive finite number")
        half_width, half_height = self.width * factor / 2, self.height * factor / 2
        self.box = (self.center_x - half_width, self.center_y - half_height,
                    self.center_x + half_width, self.center_y + half_height)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the box."""
        return {"uid": self.uid, "class_id": self.class_id, "label": self.label,
                "score": self.score, "box": list(self.box), "track_id": self.track_id,
                "metadata": deepcopy(self.metadata)}

    def __repr__(self) -> str:
        return f"Box(uid={self._uid!r}, label={self.label!r}, score={self.score:.3f})"


class Detection:
    """Contain annotations and source identity for one image or video frame."""

    def __init__(self, boxes: Optional[Iterable[Box]] = None, *, source_id: Optional[str] = None,
                 frame_index: int = 0, source_path: Optional[str] = None,
                 image_size: Optional[tuple[int, int]] = None, source: Any = None,
                 metadata: Optional[Mapping[str, Any]] = None,
                 names: Optional[Mapping[int, Any] | Sequence[Any]] = None) -> None:
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise TypeError("frame_index must be an integer")
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        self._source_id, inferred_path = _source(source, source_id)
        self._frame_index = frame_index
        self._source_path = str(source_path) if source_path is not None else inferred_path
        self._image_size = _size(image_size or getattr(source, "size", None))
        self._source = source
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")
        self.metadata = deepcopy(dict(metadata or {}))
        self._registry = _names(names)
        self._boxes: list[Box] = []
        self._uid_map: dict[int, Box] = {}
        self._next_uid = 0
        self._owner: Optional[Detections] = None
        self._commit(list(boxes or ()), None)

    identity = property(lambda self: (self.source_id, self.frame_index), doc="Stable frame identity.")
    source_id = property(lambda self: self._source_id, doc="Stable source identity.")
    frame_index = property(lambda self: self._frame_index, doc="Frame number in the source.")
    source_path = property(lambda self: self._source_path, doc="Optional physical source path.")
    image_size = property(lambda self: self._image_size, doc="Original width and height.")
    count = property(lambda self: len(self), doc="Current box count.")
    empty = property(lambda self: not self._boxes, doc="Whether the frame is empty.")
    has_objects = property(lambda self: bool(self._boxes), doc="Whether the frame has boxes.")
    boxes = property(lambda self: [box.box for box in self], doc="Coordinates in current order.")
    scores = property(lambda self: [box.score for box in self], doc="Scores in current order.")
    labels = property(lambda self: [box.label for box in self], doc="Labels in current order.")
    class_ids = property(lambda self: [box.class_id for box in self], doc="Class IDs in current order.")
    uids = property(lambda self: [box.uid for box in self], doc="UIDs in current order.")
    unique_labels = property(lambda self: list(dict.fromkeys(self.labels)), doc="Unique labels.")
    unique_class_ids = property(lambda self: list(dict.fromkeys(self.class_ids)), doc="Unique class IDs.")
    names = property(
        lambda self: {box.class_id: box.label for box in self},
        doc="Active class IDs mapped to labels.",
    )

    @property
    def label_counts(self) -> dict[str, int]:
        """Count boxes by label."""
        result: dict[str, int] = {}
        for label in self.labels:
            result[label] = result.get(label, 0) + 1
        return result

    def __len__(self) -> int:
        return len(self._boxes)

    def __iter__(self) -> Iterator[Box]:
        return iter(self._boxes)

    def __getitem__(self, index: int | slice) -> Box | tuple[Box, ...]:
        return tuple(self._boxes[index]) if isinstance(index, slice) else self._boxes[index]

    def get_box(self, uid: int) -> Box:
        """Return a box by stable UID."""
        try:
            return self._uid_map[uid]
        except KeyError as error:
            raise KeyError(f"box UID {uid} was not found") from error

    def _class(self, label: str, preferred: Optional[int] = None) -> int:
        found = next((key for key, value in self._registry.items() if value == label), None)
        if found is not None:
            return found
        result = preferred if preferred is not None and preferred not in self._registry else max(self._registry, default=-1) + 1
        self._registry[result] = label
        return result

    def _prepare(self, boxes: Iterable[Box]) -> tuple[list[Box], Names, int]:
        boxes = list(boxes)
        if any(not isinstance(box, Box) for box in boxes):
            raise TypeError("every box must be a Box")
        registry, prepared, used, next_uid = dict(self._registry), [], set(), self._next_uid
        for incoming in boxes:
            validators.validate_type(incoming, Box, "box")
            item = incoming if incoming._owner in (None, self) and incoming not in prepared else incoming._clone()
            class_id = next((key for key, value in registry.items() if value == item.label), None)
            if class_id is None:
                class_id = item._class_id if item._class_id is not None and item._class_id not in registry else max(registry, default=-1) + 1
                registry[class_id] = item.label
            uid = item._uid
            if uid is None or uid in used:
                while next_uid in used:
                    next_uid += 1
                uid, next_uid = next_uid, next_uid + 1
            used.add(uid)
            next_uid = max(next_uid, uid + 1)
            item._uid, item._class_id = uid, class_id
            prepared.append(item)
        return prepared, registry, next_uid

    def _commit(self, boxes: Iterable[Box], operation: Optional[str]) -> None:
        prepared, registry, next_uid = self._prepare(boxes)
        old = self._boxes
        for item in old:
            if item not in prepared:
                item._owner = None
        self._boxes, self._registry, self._next_uid = prepared, registry, next_uid
        self._uid_map = {item.uid: item for item in prepared}
        for item in prepared:
            item._owner = self
        if operation and self._owner is not None:
            self._owner._changed(self, operation, {"count": len(prepared)})

    def _box_changed(self, box: Box, field: str, before: Any, after: Any) -> None:
        if self._owner is not None:
            self._owner._changed(self, "update_box", {"uid": box.uid, "field": field})

    def _signature(self) -> tuple[Any, ...]:
        return tuple((b.uid, b.label, b.class_id, b.score, b.box, b.track_id, repr(b.metadata)) for b in self)

    def _clone(self, identity: Optional[Identity] = None) -> "Detection":
        source_id, frame_index = identity or self.identity
        return Detection([b._clone() for b in self], source_id=source_id, frame_index=frame_index,
                         source_path=self.source_path, image_size=self.image_size, source=self._source,
                         metadata=self.metadata, names=self._registry)

    def add_box(self, coordinates: Iterable[Any], *, label: str, score: float = 1.0,
                track_id: Optional[int | str] = None,
                metadata: Optional[Mapping[str, Any]] = None) -> None:
        """Create and append one box."""
        label = _label(label)
        self.append(Box(coordinates, label=label, score=score,
                        class_id=self._class(label), track_id=track_id, metadata=metadata))

    def append(self, box: Box) -> None:
        """Append one box."""
        self._commit([*self._boxes, box], "append_box")

    def extend(self, boxes: Iterable[Box]) -> None:
        """Append boxes atomically."""
        self._commit([*self._boxes, *list(boxes)], "extend_boxes")

    def insert(self, index: int, box: Box) -> None:
        """Insert a box by position."""
        candidates = list(self._boxes); candidates.insert(index, box)
        self._commit(candidates, "insert_box")

    def replace(self, index: int, box: Box) -> None:
        """Replace a box while preserving its UID."""
        validators.validate_type(box, Box, "box")
        item = box._clone(); item._uid = self._boxes[index].uid
        candidates = list(self._boxes); candidates[index] = item
        self._commit(candidates, "replace_box")

    def pop(self, index: int = -1) -> Box:
        """Remove and return a box by position."""
        candidates = list(self._boxes); item = candidates.pop(index)
        self._commit(candidates, "pop_box"); item._owner = None
        return item

    def remove_box(self, uid: int) -> None:
        """Remove a box by stable UID."""
        target = self.get_box(uid)
        self._commit((box for box in self if box is not target), "remove_box")

    def clear(self) -> None:
        """Remove every box."""
        self._commit([], "clear_boxes")

    def filter(self, predicate: Callable[[Box], bool]) -> None:
        """Remove boxes rejected by a predicate."""
        if not callable(predicate): raise TypeError("predicate must be callable")
        self._commit([box for box in self if bool(predicate(box))], "filter_boxes")

    def map(self, function: Callable[[Box], Optional[Box]]) -> None:
        """Transform boxes atomically using detached drafts."""
        if not callable(function): raise TypeError("function must be callable")
        transformed = []
        for original in self:
            draft = original._clone(); result = function(draft)
            result = draft if result is None else result
            validators.validate_type(result, Box, "mapped box")
            result._uid = original.uid; transformed.append(result)
        self._commit(transformed, "map_boxes")

    def drop(self, labels: Iterable[str]) -> None:
        """Remove boxes with matching labels."""
        if isinstance(labels, str):
            labels = (labels,)
        blocked = {_label(label) for label in labels}
        self._commit([box for box in self if box.label not in blocked], "drop_labels")

    def relabel(self, mapping: Mapping[str, str]) -> None:
        """Rename or merge labels."""
        if not isinstance(mapping, Mapping): raise TypeError("mapping must be a mapping")
        mapping = {_label(a): _label(b) for a, b in mapping.items()}
        items = []
        for box in self:
            item = box._clone(); item._label = mapping.get(item.label, item.label); item._class_id = None; items.append(item)
        self._registry = {}; self._commit(items, "relabel")

    def _resolve_source(self) -> Any:
        if self._source is not None: return self._source
        if self.source_path is None: raise RuntimeError("Cannot access frame pixels: source image is unavailable")
        from klygo import media
        loaded = media.load(self.source_path, verbose=False)
        for item in loaded:
            if len(loaded) == 1 or getattr(item, "frame_index", None) == self.frame_index:
                self._source = item; return item
        raise RuntimeError("Cannot access frame pixels: source frame is unavailable")

    def show(self, mode: str = "default", *, show_scores: bool = True,
             line_width: Optional[int] = None, font_size: Optional[int] = None,
             width: Optional[int] = None, title: str = "Detection", backend: str = "auto") -> None:
        """Display all boxes using default, positional-index, or UID captions."""
        if mode not in {"default", "index", "uid"}: raise ValueError("mode must be 'default', 'index', or 'uid'")
        from klygo import visual
        labels = self.labels if mode == "default" else ([f"[{i}] {b.label}" for i, b in enumerate(self)] if mode == "index" else [f"[uid:{b.uid}] {b.label}" for b in self])
        annotated = visual.draw_bboxes(self._resolve_source(), self.boxes, labels=labels, scores=self.scores,
                                       line_width=line_width, font_size=font_size,
                                       show_labels=True, show_scores=show_scores)
        visual.show_image(annotated, title=title, width=width, backend=backend)

    def to_dict(self) -> dict[str, Any]:
        """Serialize annotations without decoding pixels."""
        return {"source_id": self.source_id, "frame_index": self.frame_index,
                "source_path": self.source_path, "image_size": list(self.image_size),
                "metadata": deepcopy(self.metadata), "names": self.names,
                "boxes": [box.to_dict() for box in self]}


class Detections:
    """Hold an ordered materialized collection of detection frames."""

    def __init__(self, frames: Iterable[Detection] = (), *, metadata: Optional[Mapping[str, Any]] = None,
                 names: Optional[Mapping[int, Any] | Sequence[Any]] = None) -> None:
        if metadata is not None and not isinstance(metadata, Mapping): raise TypeError("metadata must be a mapping or None")
        self.metadata = deepcopy(dict(metadata or {})); self._registry = _names(names)
        self._frames: list[Detection] = []; self._identity_map: dict[Identity, Detection] = {}
        self._history: list[dict[str, Any]] = []; self._sequence = 0
        self._commit(list(frames), None); self._total_frames = len(self)
        self._baseline = {frame.identity: frame._signature() for frame in self}

    def _record(self, operation: str, **details: Any) -> None:
        self._sequence += 1
        self._history.append({"sequence_number": self._sequence, "operation": operation, **deepcopy(details)})

    def _reconcile(self, incoming: Detection, identity: Optional[Identity] = None) -> Detection:
        validators.validate_type(incoming, Detection, "frame")
        frame = incoming if incoming._owner in (None, self) else incoming._clone()
        if identity is not None and frame.identity != identity: frame = frame._clone(identity)
        for box in frame:
            found = next((key for key, value in self._registry.items() if value == box.label), None)
            if found is None:
                found = box._class_id if box._class_id is not None and box._class_id not in self._registry else max(self._registry, default=-1) + 1
                self._registry[found] = box.label
            box._class_id = found
        frame._registry = dict(self._registry)
        return frame

    def _prepare(self, frames: Iterable[Detection]) -> list[Detection]:
        frames = list(frames)
        if any(not isinstance(frame, Detection) for frame in frames):
            raise TypeError("every frame must be a Detection")
        raw_identities = [frame.identity for frame in frames]
        if len(raw_identities) != len(set(raw_identities)):
            raise ValueError("duplicate frame identity")
        registry, result, identities = dict(self._registry), [], set()
        try:
            for incoming in frames:
                frame = self._reconcile(incoming)
                if frame.identity in identities: raise ValueError(f"duplicate frame identity: {frame.identity!r}")
                identities.add(frame.identity); result.append(frame)
        except Exception:
            self._registry = registry; raise
        return result

    def _synchronize_registry(self) -> None:
        """Reconcile active labels after direct mutations on owned frames."""
        labels = list(dict.fromkeys(box.label for frame in self._frames for box in frame))
        previous = {label: class_id for class_id, label in self._registry.items()}
        reserved = {previous[label] for label in labels if label in previous}
        registry: Names = {}
        used: set[int] = set()
        next_id = max(self._registry, default=-1) + 1

        for label in labels:
            class_id = previous.get(label)
            if class_id is None or class_id in used:
                preferred = next(
                    (
                        box.class_id
                        for frame in self._frames
                        for box in frame
                        if box.label == label
                        and box.class_id not in used
                        and box.class_id not in reserved
                    ),
                    None,
                )
                if preferred is None:
                    while next_id in used or next_id in reserved:
                        next_id += 1
                    class_id, next_id = next_id, next_id + 1
                else:
                    class_id = preferred
            registry[class_id] = label
            used.add(class_id)

        self._registry = registry
        class_by_label = {label: class_id for class_id, label in registry.items()}
        for frame in self._frames:
            frame._registry = dict(registry)
            for box in frame:
                box._class_id = class_by_label[box.label]

    def _commit(self, frames: Iterable[Detection], operation: Optional[str]) -> None:
        prepared = self._prepare(frames)
        for frame in self._frames:
            if frame not in prepared: frame._owner = None
        self._frames = prepared; self._identity_map = {f.identity: f for f in prepared}
        for frame in prepared: frame._owner = self
        self._synchronize_registry()
        if operation: self._record(operation, count=len(prepared))

    def _changed(self, frame: Detection, operation: str, details: Mapping[str, Any]) -> None:
        if operation != "update_box":
            self._synchronize_registry()
        self._record(operation, target_identity=frame.identity, changed_fields=dict(details))

    count = property(lambda self: len(self), doc="Current frame count.")
    total_frames = property(lambda self: self._total_frames, doc="Initial frame count.")
    total_objects = property(lambda self: sum(f.count for f in self), doc="Current object count.")
    names = property(
        lambda self: {
            class_id: label
            for class_id, label in self._registry.items()
            if any(box.label == label for frame in self for box in frame)
        },
        doc="Active collection class IDs mapped to labels.",
    )
    history = property(lambda self: tuple(deepcopy(self._history)), doc="Read-only history.")
    added_frames = property(lambda self: tuple(i for i in self._identity_map if i not in self._baseline), doc="Added identities.")
    deleted_frames = property(lambda self: tuple(i for i in self._baseline if i not in self._identity_map), doc="Deleted identities.")
    modified_frames = property(lambda self: tuple(i for i, base in self._baseline.items() if i in self._identity_map and self._identity_map[i]._signature() != base), doc="Modified identities.")

    def __len__(self) -> int: return len(self._frames)
    def __iter__(self) -> Iterator[Detection]: return iter(self._frames)
    def __getitem__(self, index: int | slice) -> Detection | tuple[Detection, ...]:
        return tuple(self._frames[index]) if isinstance(index, slice) else self._frames[index]

    def get_frame(self, frame_index: int, *, source_id: Optional[str] = None) -> Detection:
        """Return a frame by stable source identity."""
        if source_id is not None:
            try: return self._identity_map[(str(source_id), frame_index)]
            except KeyError as error: raise KeyError("frame identity was not found") from error
        matches = [f for f in self if f.frame_index == frame_index]
        if not matches: raise KeyError(f"frame index {frame_index} was not found")
        if len(matches) > 1: raise ValueError("frame_index is ambiguous; provide source_id")
        return matches[0]

    def append(self, frame: Detection) -> None:
        """Append a frame."""
        validators.validate_type(frame, Detection, "frame")
        if frame.identity in self._identity_map: raise ValueError("frame identity already exists")
        self._commit([*self._frames, frame], "append_frame")

    def preview_add(self, frame: Detection) -> dict[str, Any]:
        """Preview identity and class reconciliation without mutation."""
        validators.validate_type(frame, Detection, "frame")
        identity_conflict = frame.identity in self._identity_map
        preview_frame = frame._clone()
        if identity_conflict:
            preview_frame = preview_frame._clone(
                (f"{frame.source_id}:preview:{uuid.uuid4().hex}", frame.frame_index)
            )

        draft = Detections(
            [*[item._clone() for item in self], preview_frame],
            metadata=self.metadata,
            names=self._registry,
        )
        resolved_frame = draft[-1]
        existing_labels = set(self.names.values())
        incoming_ids: dict[str, list[int]] = {}
        for box in frame:
            incoming_ids.setdefault(box.label, []).append(box.class_id)

        mapping = {}
        for label in resolved_frame.unique_labels:
            resolved_id = next(box.class_id for box in resolved_frame if box.label == label)
            original_ids = tuple(dict.fromkeys(incoming_ids[label]))
            mapping[label] = {
                "incoming_class_ids": original_ids,
                "resolved_class_id": resolved_id,
                "remapped": any(class_id != resolved_id for class_id in original_ids),
            }

        return {
            "can_add": not identity_conflict,
            "identity": frame.identity,
            "identity_conflict": identity_conflict,
            "new_labels": tuple(
                label for label in resolved_frame.unique_labels if label not in existing_labels
            ),
            "mapping": mapping,
        }

    def extend(self, frames: Iterable[Detection]) -> None:
        """Append frames atomically."""
        incoming = list(frames)
        if any(not isinstance(f, Detection) for f in incoming): raise TypeError("every frame must be a Detection")
        if {f.identity for f in incoming} & set(self._identity_map): raise ValueError("frame identity already exists")
        self._commit([*self._frames, *incoming], "extend_frames")

    def insert(self, index: int, frame: Detection) -> None:
        """Insert a frame by position."""
        if frame.identity in self._identity_map: raise ValueError("frame identity already exists")
        items = list(self._frames); items.insert(index, frame); self._commit(items, "insert_frame")

    def replace(self, index: int, frame: Detection) -> None:
        """Replace content while preserving frame identity."""
        validators.validate_type(frame, Detection, "frame")
        items = list(self._frames); items[index] = frame._clone(items[index].identity); self._commit(items, "replace_frame")

    def pop(self, index: int = -1) -> Detection:
        """Remove and return a frame by position."""
        items = list(self._frames); frame = items.pop(index); self._commit(items, "pop_frame"); frame._owner = None; return frame

    def remove_frame(self, frame_index: int, *, source_id: Optional[str] = None) -> None:
        """Remove a frame by source identity."""
        target = self.get_frame(frame_index, source_id=source_id)
        self._commit((f for f in self if f is not target), "remove_frame")

    def clear(self) -> None:
        """Remove every frame."""
        self._commit([], "clear_frames")

    def merge(self, other: "Detections") -> None:
        """Merge another collection, rejecting any identity collision."""
        validators.validate_type(other, Detections, "other"); self.extend(f._clone() for f in other); self._history[-1]["operation"] = "merge"

    def filter(self, predicate: Callable[[Detection], bool]) -> None:
        """Remove frames rejected by a predicate."""
        if not callable(predicate): raise TypeError("predicate must be callable")
        self._commit([f for f in self if bool(predicate(f))], "filter_frames")

    def map(self, function: Callable[[Detection], Optional[Detection]]) -> None:
        """Transform frames atomically using detached drafts."""
        if not callable(function): raise TypeError("function must be callable")
        transformed = []
        for original in self:
            draft = original._clone(); result = function(draft); result = draft if result is None else result
            validators.validate_type(result, Detection, "mapped frame"); transformed.append(result._clone(original.identity))
        self._commit(transformed, "map_frames")

    def drop(self, labels: Iterable[str]) -> None:
        """Remove matching boxes from every frame."""
        if isinstance(labels, str):
            labels = (labels,)
        labels = tuple(labels)
        drafts = [f._clone() for f in self]
        for frame in drafts: frame.drop(labels)
        self._commit(drafts, "drop_labels")

    def relabel(self, mapping: Mapping[str, str]) -> None:
        """Rename or merge labels across all frames."""
        drafts = [f._clone() for f in self]
        for frame in drafts: frame.relabel(mapping)
        self._registry = {}; self._commit(drafts, "relabel")

    def repredict(self, model: Any, index: Optional[int | Iterable[int]] = None, **options: Any) -> None:
        """Replace selected predictions while preserving source identities."""
        indices = list(range(len(self))) if index is None else ([index] if isinstance(index, int) and not isinstance(index, bool) else list(index))
        normalized = []
        for item in indices:
            if isinstance(item, bool) or not isinstance(item, int): raise TypeError("prediction indices must be integers")
            item = item + len(self) if item < 0 else item
            if item < 0 or item >= len(self): raise IndexError("prediction index out of range")
            if item not in normalized: normalized.append(item)
        if not normalized:
            return
        predicted = model.predict([self._frames[i]._resolve_source() for i in normalized], stream=False, **options)
        validators.validate_type(predicted, Detections, "model prediction")
        if len(predicted) != len(normalized): raise ValueError("model prediction must return one frame per selected source")
        items = list(self._frames)
        for position, fresh in zip(normalized, predicted):
            original = self._frames[position]; replacement = fresh._clone(original.identity)
            replacement._source, replacement._source_path, replacement._image_size = original._source, original.source_path, original.image_size
            items[position] = replacement
        self._commit(items, "repredict")

    def to_dict(self) -> dict[str, Any]:
        """Serialize annotations without decoding pixels."""
        return {"count": self.count, "total_frames": self.total_frames,
                "total_objects": self.total_objects, "names": self.names,
                "metadata": deepcopy(self.metadata), "frames": [f.to_dict() for f in self]}

    def export(self, path: str, *, format: str = "yolo", overwrite: bool = False) -> None:
        """Export annotations without copying source images."""
        format = str(format).lower()
        if format == "json": files.save(path, self.to_dict(), overwrite=overwrite, verbose=False); return
        if format != "yolo": raise ValueError("format must be 'yolo' or 'json'")
        if files.exists(path) and not overwrite: raise FileExistsError(f"output path already exists: {path}")
        if files.exists(path): files.remove(path, missing_ok=True)
        labels_dir = files.join(path, "labels"); files.mkdir(labels_dir)
        names, manifest = self.names, []; label_ids = {label: key for key, label in names.items()}
        for position, frame in enumerate(self):
            width, height = frame.image_size
            digest = sha1(f"{frame.source_id}:{frame.frame_index}".encode()).hexdigest()[:10]
            filename = f"frame_{position:06d}_{frame.frame_index}_{digest}.txt"
            lines = [f"{label_ids[b.label]} {(b.xmin+b.xmax)/2/width:.6f} {(b.ymin+b.ymax)/2/height:.6f} {b.width/width:.6f} {b.height/height:.6f}" for b in frame]
            files.save(files.join(labels_dir, filename), lines, overwrite=True, verbose=False)
            manifest.append({"source_id": frame.source_id, "frame_index": frame.frame_index,
                             "source_path": frame.source_path, "image_size": list(frame.image_size),
                             "label_file": files.as_posix(files.join("labels", filename))})
        files.save(files.join(path, "data.yaml"), {"names": names}, overwrite=True, verbose=False)
        files.save(files.join(path, "manifest.json"), manifest, overwrite=True, verbose=False)


def build_detections(images: Sequence[Any], predictions: Sequence[Mapping[str, Any]], *,
                     metadata: Optional[Mapping[str, Any]] = None,
                     names: Optional[Mapping[int, Any] | Sequence[Any]] = None) -> list[Detection]:
    """Build one frame per image from normalized backend predictions."""
    images, predictions, registry = list(images), list(predictions), _names(names)
    if len(images) != len(predictions): raise ValueError("images and predictions must have the same length")
    frames = []
    for position, (image, prediction) in enumerate(zip(images, predictions)):
        if not isinstance(prediction, Mapping): raise TypeError("each prediction must be a mapping")
        coords = _list(prediction.get("boxes", []), "boxes")
        scores = _list(prediction.get("scores", [1.0] * len(coords)), "scores")
        labels = _list(prediction.get("text_labels", prediction.get("labels", ["object"] * len(coords))), "labels")
        class_ids = _list(prediction.get("class_ids", [None] * len(coords)), "class_ids")
        if len({len(coords), len(scores), len(labels), len(class_ids)}) != 1: raise ValueError("boxes, scores, labels, and class_ids must have the same length")
        boxes, local = [], dict(registry)
        for uid, (coordinates, score, label, preferred) in enumerate(zip(coords, scores, labels, class_ids)):
            label = local[label] if isinstance(label, int) and label in local else _label(label)
            found = next((key for key, value in local.items() if value == label), None)
            if found is None:
                preferred = None if preferred is None else int(preferred)
                found = preferred if preferred is not None and preferred not in local else max(local, default=-1) + 1; local[found] = label
            boxes.append(Box(coordinates, label=label, score=score, class_id=found, uid=uid))
        frame_index = getattr(image, "frame_index", None); frame_index = position if frame_index is None else frame_index
        source_id, source_path = _source(image, None)
        frames.append(Detection(boxes, source_id=source_id, frame_index=frame_index,
                                source_path=source_path, image_size=getattr(image, "size", None),
                                source=image, metadata=deepcopy(dict(metadata or {})), names=local))
        registry.update(local)
    return frames


__all__ = ["build_detections", "Box", "Detection", "Detections"]
