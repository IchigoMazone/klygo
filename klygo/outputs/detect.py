"""Three-level object-detection result containers."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha1
import math
from numbers import Integral
import sys
import uuid
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np

from klygo import files, validators

Names = dict[int, str]
Identity = tuple[str, int]
_RESTORED = object()


def _label(value: Any) -> str:
    result = "" if value is None else str(value).strip()
    if not result:
        raise ValueError("label must be a non-empty string")
    return sys.intern(result)


def _score(value: Any) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise TypeError("score must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("score must be a finite number between 0 and 1")
    return result


def _coords(value: Iterable[Any]) -> tuple[float, float, float, float]:
    try:
        items = tuple(value)
    except (TypeError, ValueError) as error:
        raise TypeError("box must contain four numeric coordinates") from error
    if any(isinstance(item, (bool, str, bytes)) for item in items):
        raise TypeError("box must contain four numeric coordinates")
    try:
        result = tuple(float(item) for item in items)
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
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, Integral)
        or not isinstance(height, Integral)
    ):
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
        if isinstance(key, bool) or not isinstance(key, Integral):
            raise TypeError("class IDs must be non-negative integers")
        if key < 0:
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

    __slots__ = (
        "_owner",
        "_uid",
        "_class_id",
        "_label",
        "_score",
        "_box",
        "_track_id",
        "_metadata",
    )

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
        self._metadata: Optional[dict[str, Any]] = (
            deepcopy(dict(metadata)) if metadata else None
        )

    @classmethod
    def _fast_create(
        cls,
        coords: tuple[float, float, float, float],
        label: str,
        score: float,
        class_id: Optional[int],
        uid: Optional[int],
        track_id: Optional[int | str] = None,
        metadata: Optional[dict[str, Any]] = None,
        owner: Optional["Detection"] = None,
    ) -> "Box":
        instance = cls.__new__(cls)
        instance._owner = owner
        instance._uid = uid
        instance._class_id = class_id
        instance._label = label
        instance._score = score
        instance._box = coords
        instance._track_id = track_id
        instance._metadata = metadata
        return instance

    @staticmethod
    def _integer(value: Optional[int], name: str) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer or None")
        value = int(value)
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
        class_id = self.class_id if self._class_id is not None else None
        meta = deepcopy(self._metadata) if self._metadata else None
        return Box._fast_create(
            self._box,
            self._label,
            self._score,
            class_id,
            self._uid,
            self._track_id,
            meta,
            None,
        )

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
        if self._owner is not None:
            return self._owner._resolve_class_id(self._label, self._class_id)
        return self._class_id

    @class_id.setter
    def class_id(self, value: int) -> None:
        value = self._integer(value, "class_id")
        assert value is not None
        if (
            self._owner is not None
            and self._owner._resolve_class_id(self._label, value) != value
        ):
            raise ValueError("class_id must map to the box label in the owning Detection")
        before = self._class_id
        if self._owner is not None and before != value:
            self._owner._before_change()
        self._class_id = value
        self._notify("class_id", before, value)

    @property
    def score(self) -> float:
        """Return the confidence score."""
        return self._score

    @score.setter
    def score(self, value: float) -> None:
        value = _score(value)
        before = self._score
        if self._owner is not None and before != value:
            self._owner._before_change()
        self._score = value
        self._notify("score", before, value)

    @property
    def box(self) -> tuple[float, float, float, float]:
        """Return immutable XYXY coordinates."""
        return self._box

    @box.setter
    def box(self, value: Iterable[Any]) -> None:
        value = _coords(value)
        before = self._box
        if self._owner is not None and before != value:
            self._owner._before_change()
        self._box = value
        self._notify("box", before, value)

    @property
    def track_id(self) -> Optional[int | str]:
        """Return the optional tracking identity."""
        return self._track_id

    @track_id.setter
    def track_id(self, value: Optional[int | str]) -> None:
        value = self._track(value)
        before = self._track_id
        if self._owner is not None and before != value:
            self._owner._before_change()
        self._track_id = value
        self._notify("track_id", before, value)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return box metadata."""
        if self._owner is not None:
            self._owner._before_change()
        if self._metadata is None:
            self._metadata = {}
            if self._owner is not None:
                self._owner._sync_box_metadata_ref(self, self._metadata)
        return self._metadata

    @metadata.setter
    def metadata(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("metadata must be a mapping")
        value = deepcopy(dict(value))
        before = {} if self._metadata is None else self._metadata
        if self._owner is not None and before != value:
            self._owner._before_change()
        self._metadata = value if value else None
        self._notify("metadata", before, value)

    xmin = property(lambda self: self._box[0], doc="Left coordinate.")
    ymin = property(lambda self: self._box[1], doc="Top coordinate.")
    xmax = property(lambda self: self._box[2], doc="Right coordinate.")
    ymax = property(lambda self: self._box[3], doc="Bottom coordinate.")
    width = property(lambda self: self._box[2] - self._box[0], doc="Box width.")
    height = property(lambda self: self._box[3] - self._box[1], doc="Box height.")
    area = property(lambda self: (self._box[2] - self._box[0]) * (self._box[3] - self._box[1]), doc="Box area.")
    aspect_ratio = property(lambda self: (self._box[2] - self._box[0]) / (self._box[3] - self._box[1]), doc="Width divided by height.")
    center_x = property(lambda self: (self._box[0] + self._box[2]) / 2, doc="Horizontal center.")
    center_y = property(lambda self: (self._box[1] + self._box[3]) / 2, doc="Vertical center.")
    center = property(lambda self: (self.center_x, self.center_y), doc="Box center.")
    size = property(lambda self: (self.width, self.height), doc="Box size.")

    def translate(self, dx: float = 0, dy: float = 0) -> None:
        """Move coordinates by a finite offset."""
        if isinstance(dx, (bool, str, bytes)) or isinstance(dy, (bool, str, bytes)):
            raise TypeError("dx and dy must be numbers")
        dx, dy = float(dx), float(dy)
        if not math.isfinite(dx) or not math.isfinite(dy):
            raise ValueError("dx and dy must be finite")
        self.box = self.xmin + dx, self.ymin + dy, self.xmax + dx, self.ymax + dy

    def scale(self, factor: float) -> None:
        """Scale coordinates around the box center."""
        if isinstance(factor, (bool, str, bytes)):
            raise TypeError("factor must be a number")
        factor = float(factor)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("factor must be a positive finite number")
        half_width, half_height = self.width * factor / 2, self.height * factor / 2
        self.box = (self.center_x - half_width, self.center_y - half_height,
                    self.center_x + half_width, self.center_y + half_height)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the box."""
        return {"uid": self.uid, "class_id": self.class_id, "label": self._label,
                "score": self._score, "box": list(self._box), "track_id": self._track_id,
                "metadata": deepcopy(self._metadata) if self._metadata else {}}

    def __repr__(self) -> str:
        return f"Box(uid={self._uid!r}, label={self._label!r}, score={self._score:.3f})"


_EMPTY_XYXY = np.empty((0, 4), dtype=np.float64)
_EMPTY_SCORES = np.empty((0,), dtype=np.float64)
_EMPTY_INTS = np.empty((0,), dtype=np.int64)
_EMPTY_XYXY32 = np.empty((0, 4), dtype=np.float32)
_EMPTY_SCORES32 = np.empty((0,), dtype=np.float32)
_EMPTY_INTS32 = np.empty((0,), dtype=np.int32)


def _precision(value: Any) -> int:
    if isinstance(value, bool) or value not in (32, 64):
        raise ValueError("dtype must be 32 or 64")
    return int(value)


def _empty_arrays(bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bits == 32:
        return _EMPTY_XYXY32, _EMPTY_SCORES32, _EMPTY_INTS32
    return _EMPTY_XYXY, _EMPTY_SCORES, _EMPTY_INTS


class Detection:
    """Contain annotations and source identity for one image or video frame."""

    __slots__ = (
        "_source_id",
        "_frame_index",
        "_identity",
        "_precision",
        "_source_path",
        "_image_size",
        "_source",
        "_restore_trace",
        "metadata",
        "_registry",
        "_label_ids",
        "_xyxy",
        "_scores",
        "_class_ids",
        "_uids",
        "_labels",
        "_track_ids",
        "_metas",
        "_boxes_cache",
        "_uid_to_idx",
        "_next_uid",
        "_owner",
        "__dict__",
    )

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
        self._identity = (self._source_id, self._frame_index)
        self._precision = 64
        self._source_path = str(source_path) if source_path is not None else inferred_path
        self._image_size = _size(image_size or getattr(source, "size", None))
        self._source = source
        self._restore_trace = getattr(source, "_preprocessing_trace", None) or None
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")
        self.metadata = deepcopy(dict(metadata)) if metadata else {}
        self._registry = _names(names)
        self._label_ids = {label: class_id for class_id, label in self._registry.items()}
        self._xyxy: np.ndarray = _EMPTY_XYXY
        self._scores: np.ndarray = _EMPTY_SCORES
        self._class_ids: np.ndarray = _EMPTY_INTS
        self._uids: np.ndarray = _EMPTY_INTS
        self._labels: list[str] = []
        self._track_ids: Optional[list[Optional[int | str]]] = None
        self._metas: Optional[list[Optional[dict[str, Any]]]] = None
        self._boxes_cache: Optional[list[Optional[Box]]] = None
        self._uid_to_idx: Optional[dict[int, int]] = None
        self._next_uid = 0
        self._owner: Optional[Detections] = None
        if boxes is not None:
            box_list = list(boxes)
            if box_list:
                self._commit(box_list, None)

    @classmethod
    def _from_arrays(
        cls,
        xyxy: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        uids: np.ndarray,
        labels: list[str],
        *,
        track_ids: Optional[list[Optional[int | str]]] = None,
        metas: Optional[list[Optional[dict[str, Any]]]] = None,
        next_uid: Optional[int] = None,
        source_id: Optional[str] = None,
        frame_index: int = 0,
        source_path: Optional[str] = None,
        image_size: Optional[tuple[int, int]] = None,
        source: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        registry: Optional[Names] = None,
        label_ids: Optional[dict[str, int]] = None,
        precision: Optional[int] = None,
        restore_trace: Any = None,
    ) -> "Detection":
        instance = cls.__new__(cls)
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise TypeError("frame_index must be an integer")
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        instance._source_id, inferred_path = _source(source, source_id)
        instance._frame_index = frame_index
        instance._identity = (instance._source_id, instance._frame_index)
        if precision is None:
            arrays = (xyxy, scores, class_ids, uids)
            precision = 32 if arrays and all(array.dtype.itemsize <= 4 for array in arrays) else 64
        instance._precision = _precision(precision)
        instance._source_path = str(source_path) if source_path is not None else inferred_path
        instance._image_size = _size(image_size or getattr(source, "size", None))
        instance._source = source
        instance._restore_trace = (
            restore_trace
            if restore_trace is _RESTORED or restore_trace is not None
            else (getattr(source, "_preprocessing_trace", None) or None)
        )
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")
        instance.metadata = deepcopy(dict(metadata)) if metadata else {}
        instance._registry = dict(registry) if registry is not None else {}
        instance._label_ids = (
            dict(label_ids)
            if label_ids is not None
            else {lbl: cid for cid, lbl in instance._registry.items()}
        )
        float_dtype = np.float32 if instance._precision == 32 else np.float64
        int_dtype = np.int32 if instance._precision == 32 else np.int64
        if instance._precision == 32:
            int_limit = np.iinfo(np.int32).max
            if any(array.size and int(array.max()) > int_limit for array in (class_ids, uids)):
                raise OverflowError("class IDs and UIDs must fit in int32")
            if xyxy.size and np.any(np.abs(xyxy) > np.finfo(np.float32).max):
                raise OverflowError("box coordinates must fit in float32")
        instance._xyxy = np.asarray(xyxy, dtype=float_dtype)
        instance._scores = np.asarray(scores, dtype=float_dtype)
        instance._class_ids = np.asarray(class_ids, dtype=int_dtype)
        instance._uids = np.asarray(uids, dtype=int_dtype)
        instance._labels = labels
        instance._track_ids = track_ids
        instance._metas = metas
        instance._boxes_cache = None
        instance._uid_to_idx = None
        if next_uid is not None:
            instance._next_uid = next_uid
        elif len(uids) > 0:
            instance._next_uid = int(uids.max()) + 1
        else:
            instance._next_uid = 0
        instance._owner = None
        return instance

    identity = property(lambda self: self._identity, doc="Stable frame identity.")
    source_id = property(lambda self: self._source_id, doc="Stable source identity.")
    frame_index = property(lambda self: self._frame_index, doc="Frame number in the source.")
    source_path = property(lambda self: self._source_path, doc="Optional physical source path.")
    image_size = property(lambda self: self._image_size, doc="Original width and height.")
    dtype = property(lambda self: self._precision, doc="Numeric storage width, either 32 or 64 bits.")
    count = property(lambda self: len(self._labels), doc="Current box count.")
    empty = property(lambda self: not self._labels, doc="Whether the frame is empty.")
    has_objects = property(lambda self: bool(self._labels), doc="Whether the frame has boxes.")
    boxes = property(lambda self: [tuple(row) for row in self._xyxy.tolist()], doc="Coordinates in current order.")
    scores = property(lambda self: self._scores.tolist(), doc="Scores in current order.")
    labels = property(lambda self: list(self._labels), doc="Labels in current order.")

    @property
    def class_ids(self) -> list[int]:
        """Class IDs in current order."""
        if not self._labels:
            return []
        if self._owner is not None:
            lookup = self._owner._label_ids
        else:
            lookup = self._label_ids
        return [lookup.get(lbl, cid) for lbl, cid in zip(self._labels, self._class_ids.tolist())]

    uids = property(lambda self: self._uids.tolist(), doc="UIDs in current order.")
    unique_labels = property(lambda self: list(dict.fromkeys(self._labels)), doc="Unique labels.")
    unique_class_ids = property(lambda self: list(dict.fromkeys(self.class_ids)), doc="Unique class IDs.")
    names = property(
        lambda self: dict(zip(self.class_ids, self._labels)),
        doc="Active class IDs mapped to labels.",
    )

    @property
    def label_counts(self) -> dict[str, int]:
        """Count boxes by label."""
        result: dict[str, int] = {}
        for label in self._labels:
            result[label] = result.get(label, 0) + 1
        return result

    def __len__(self) -> int:
        return len(self._labels)

    def _get_box_at(self, index: int) -> Box:
        if self._boxes_cache is None:
            self._boxes_cache = [None] * len(self._labels)
        box = self._boxes_cache[index]
        if box is None:
            row = self._xyxy[index]
            coords = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            tid = None if self._track_ids is None else self._track_ids[index]
            meta = (
                None
                if self._metas is None or self._metas[index] is None
                else deepcopy(self._metas[index])
            )
            box = Box._fast_create(
                coords=coords,
                label=self._labels[index],
                score=float(self._scores[index]),
                class_id=int(self._class_ids[index]),
                uid=int(self._uids[index]),
                track_id=tid,
                metadata=meta,
                owner=self,
            )
            if meta is not None and self._metas is not None:
                self._metas[index] = meta
            self._boxes_cache[index] = box
        return box

    def _detached_box_at(self, index: int) -> Box:
        if self._boxes_cache is not None and self._boxes_cache[index] is not None:
            return self._boxes_cache[index]._clone()
        row = self._xyxy[index]
        coords = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        tid = None if self._track_ids is None else self._track_ids[index]
        meta = (
            None
            if self._metas is None or self._metas[index] is None
            else deepcopy(self._metas[index])
        )
        cid = int(self._class_ids[index])
        if self._owner is not None:
            cid = self._owner._resolve_class_id(self._labels[index], cid)
        else:
            cid = self._label_ids.get(self._labels[index], cid)
        return Box._fast_create(
            coords=coords,
            label=self._labels[index],
            score=float(self._scores[index]),
            class_id=cid,
            uid=int(self._uids[index]),
            track_id=tid,
            metadata=meta,
            owner=None,
        )

    def __iter__(self) -> Iterator[Box]:
        for index in range(len(self._labels)):
            yield self._get_box_at(index)

    def _subset(self, indices: Iterable[int]) -> "Detection":
        """Return an independent frame containing selected box positions."""
        n = len(self._labels)
        normalized: list[int] = []
        for item in indices:
            if isinstance(item, bool) or not isinstance(item, Integral):
                raise TypeError("box indices must be integers")
            position = int(item)
            position = position + n if position < 0 else position
            if position < 0 or position >= n:
                raise IndexError("box index out of range")
            if position in normalized:
                raise ValueError("box indices must not contain duplicates")
            normalized.append(position)

        int_dtype = np.int32 if self._precision == 32 else np.int64
        selected = np.asarray(normalized, dtype=np.int64)
        class_ids = np.asarray(self.class_ids, dtype=int_dtype)
        track_ids = None if self._track_ids is None else [self._track_ids[i] for i in normalized]
        metas = None if self._metas is None else [deepcopy(self._metas[i]) for i in normalized]
        registry = self._owner._registry if self._owner is not None else self._registry
        label_ids = self._owner._label_ids if self._owner is not None else self._label_ids
        return Detection._from_arrays(
            self._xyxy[selected].copy(),
            self._scores[selected].copy(),
            class_ids[selected].copy(),
            self._uids[selected].copy(),
            [self._labels[i] for i in normalized],
            track_ids=track_ids,
            metas=metas,
            next_uid=self._next_uid,
            source_id=self.source_id,
            frame_index=self.frame_index,
            source_path=self.source_path,
            image_size=self.image_size,
            source=self._source,
            metadata=self.metadata,
            registry=registry,
            label_ids=label_ids,
            precision=self._precision,
            restore_trace=self._restore_trace,
        )

    def __getitem__(self, index: int | slice | Iterable[int]) -> Box | tuple[Box, ...] | "Detection":
        n = len(self._labels)
        if isinstance(index, slice):
            return tuple(self._get_box_at(i) for i in range(*index.indices(n)))
        if isinstance(index, bool):
            raise TypeError("box index must be an integer, slice, or iterable of integers")
        if not isinstance(index, Integral):
            try:
                return self._subset(index)
            except TypeError as error:
                if "box indices" in str(error):
                    raise
                raise TypeError("box index must be an integer, slice, or iterable of integers") from error
        position = int(index)
        if position < 0:
            position += n
        if position < 0 or position >= n:
            raise IndexError("list index out of range")
        return self._get_box_at(position)

    def get_box(self, uid: int) -> Box:
        """Return a box by stable UID."""
        if self._uid_to_idx is None:
            self._uid_to_idx = {u: i for i, u in enumerate(self._uids.tolist())}
        try:
            idx = self._uid_to_idx[uid]
        except KeyError as error:
            raise KeyError(f"box UID {uid} was not found") from error
        return self._get_box_at(idx)

    def _before_change(self) -> None:
        if self._owner is not None:
            self._owner._snapshot_frame_baseline(self)

    def _ensure_private_registry(self) -> None:
        if self._owner is not None and self._registry is self._owner._registry:
            self._registry = dict(self._registry)
            self._label_ids = dict(self._label_ids)

    def _resolve_class_id(self, label: str, fallback: int) -> int:
        if self._owner is not None:
            return self._owner._resolve_class_id(label, fallback)
        return self._label_ids.get(label, fallback)

    def _class(self, label: str, preferred: Optional[int] = None) -> int:
        if self._owner is not None:
            return self._owner._class(label)
        found = self._label_ids.get(label)
        if found is not None:
            return found
        result = preferred if preferred is not None and preferred not in self._registry else max(self._registry, default=-1) + 1
        self._registry[result] = label
        self._label_ids[label] = result
        return result

    def _prepare(self, boxes: Iterable[Box]) -> tuple[list[Box], Names, int]:
        boxes = list(boxes)
        if any(not isinstance(box, Box) for box in boxes):
            raise TypeError("every box must be a Box")
        registry, label_ids = dict(self._registry), dict(self._label_ids)
        prepared, used, next_uid = [], set(), self._next_uid
        seen_objs: set[int] = set()
        for incoming in boxes:
            validators.validate_type(incoming, Box, "box")
            obj_id = id(incoming)
            item = incoming if incoming._owner in (None, self) and obj_id not in seen_objs else incoming._clone()
            seen_objs.add(id(item))
            class_id = label_ids.get(item.label)
            if class_id is None:
                class_id = item._class_id if item._class_id is not None and item._class_id not in registry else max(registry, default=-1) + 1
                registry[class_id] = item.label
                label_ids[item.label] = class_id
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

    def _commit(self, boxes: Iterable[Box], operation: Optional[str], *, retain_boxes: bool = True) -> None:
        prepared, registry, next_uid = self._prepare(boxes)
        n = len(prepared)
        if n and self._precision == 32:
            values = [value for b in prepared for value in (b._class_id, b._uid)]
            if max(values) > np.iinfo(np.int32).max:
                raise OverflowError("class IDs and UIDs must fit in int32")
            if any(abs(value) > np.finfo(np.float32).max for b in prepared for value in b._box):
                raise OverflowError("box coordinates must fit in float32")
        self._before_change()
        if self._boxes_cache is not None:
            surviving = {id(b) for b in prepared} if retain_boxes else set()
            for item in self._boxes_cache:
                if item is not None and id(item) not in surviving:
                    item._owner = None

        if n == 0:
            empty_xyxy, empty_scores, empty_ints = _empty_arrays(self._precision)
            self._xyxy = empty_xyxy
            self._scores = empty_scores
            self._class_ids = empty_ints
            self._uids = empty_ints
            self._labels = []
            self._track_ids = None
            self._metas = None
            self._boxes_cache = None
        else:
            float_dtype = np.float32 if self._precision == 32 else np.float64
            int_dtype = np.int32 if self._precision == 32 else np.int64
            self._xyxy = np.array([b._box for b in prepared], dtype=float_dtype)
            self._scores = np.array([b._score for b in prepared], dtype=float_dtype)
            self._class_ids = np.array([b._class_id for b in prepared], dtype=int_dtype)
            self._uids = np.array([b._uid for b in prepared], dtype=int_dtype)
            self._labels = [b._label for b in prepared]
            has_tracks = any(b._track_id is not None for b in prepared)
            self._track_ids = [b._track_id for b in prepared] if has_tracks else None
            has_metas = any(b._metadata for b in prepared)
            self._metas = [b._metadata if b._metadata else None for b in prepared] if has_metas else None
            if retain_boxes:
                self._boxes_cache = list(prepared)
                for item in prepared:
                    item._owner = self
            else:
                self._boxes_cache = None

        self._registry = registry
        self._label_ids = {label: class_id for class_id, label in registry.items()}
        self._next_uid = next_uid
        self._uid_to_idx = None
        if operation and self._owner is not None:
            self._owner._changed(self, operation, {"count": n})

    def _copy_from(self, draft: "Detection", operation: Optional[str]) -> None:
        self._before_change()
        self._precision = draft._precision
        empty_xyxy, empty_scores, empty_ints = _empty_arrays(self._precision)
        if self._boxes_cache is not None:
            for item in self._boxes_cache:
                if item is not None:
                    item._owner = None
            self._boxes_cache = None
        self._xyxy = draft._xyxy.copy() if len(draft._labels) else empty_xyxy
        self._scores = draft._scores.copy() if len(draft._labels) else empty_scores
        self._class_ids = draft._class_ids.copy() if len(draft._labels) else empty_ints
        self._uids = draft._uids.copy() if len(draft._labels) else empty_ints
        self._labels = list(draft._labels)
        self._track_ids = list(draft._track_ids) if draft._track_ids is not None else None
        self._metas = [deepcopy(m) if m else None for m in draft._metas] if draft._metas is not None else None
        self._registry = dict(draft._registry)
        self._label_ids = dict(draft._label_ids)
        self._next_uid = max(self._next_uid, draft._next_uid)
        self._uid_to_idx = None
        if operation and self._owner is not None:
            self._owner._changed(self, operation, {"count": len(self._labels)})

    def _slice_keep(
        self,
        indices: Sequence[int] | np.ndarray,
        operation: Optional[str],
        new_xyxy: Optional[np.ndarray] = None,
    ) -> None:
        idx_list = indices.tolist() if isinstance(indices, np.ndarray) else list(indices)
        unchanged_indices = len(idx_list) == len(self._labels) and all(
            index == position for position, index in enumerate(idx_list)
        )
        unchanged_boxes = new_xyxy is None or np.array_equal(
            np.asarray(new_xyxy), self._xyxy, equal_nan=True
        )
        if unchanged_indices and unchanged_boxes:
            return
        self._before_change()
        n = len(idx_list)
        if self._boxes_cache is not None:
            kept_set = set(idx_list)
            for i, b in enumerate(self._boxes_cache):
                if b is not None and i not in kept_set:
                    b._owner = None
            if n == 0:
                self._boxes_cache = None
            else:
                self._boxes_cache = [self._boxes_cache[i] for i in idx_list]
                if new_xyxy is not None:
                    for new_i, b in enumerate(self._boxes_cache):
                        if b is not None:
                            row = new_xyxy[new_i]
                            b._box = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        if n == 0:
            empty_xyxy, empty_scores, empty_ints = _empty_arrays(self._precision)
            self._xyxy = empty_xyxy
            self._scores = empty_scores
            self._class_ids = empty_ints
            self._uids = empty_ints
            self._labels = []
            self._track_ids = None
            self._metas = None
        else:
            self._xyxy = new_xyxy if new_xyxy is not None else self._xyxy[idx_list]
            self._scores = self._scores[idx_list]
            self._class_ids = self._class_ids[idx_list]
            self._uids = self._uids[idx_list]
            self._labels = [self._labels[i] for i in idx_list]
            if self._track_ids is not None:
                self._track_ids = [self._track_ids[i] for i in idx_list]
            if self._metas is not None:
                self._metas = [self._metas[i] for i in idx_list]
        self._uid_to_idx = None
        if operation and self._owner is not None:
            self._owner._changed(self, operation, {"count": n})

    def _find_box_index(self, box: Box) -> Optional[int]:
        uid = box._uid
        if uid is None:
            return None
        if self._uid_to_idx is None:
            self._uid_to_idx = {u: i for i, u in enumerate(self._uids.tolist())}
        idx = self._uid_to_idx.get(uid)
        if idx is not None and self._boxes_cache is not None and idx < len(self._boxes_cache) and self._boxes_cache[idx] is box:
            return idx
        return None

    def _sync_box_metadata_ref(self, box: Box, meta: dict[str, Any]) -> None:
        idx = self._find_box_index(box)
        if idx is not None:
            if self._metas is None:
                self._metas = [None] * len(self._labels)
            self._metas[idx] = meta

    def _box_changed(self, box: Box, field: str, before: Any, after: Any) -> None:
        idx = self._find_box_index(box)
        if idx is not None:
            if field == "box":
                self._xyxy[idx] = after
            elif field == "score":
                self._scores[idx] = after
            elif field == "class_id":
                self._class_ids[idx] = after
            elif field == "track_id":
                if self._track_ids is None:
                    self._track_ids = [None] * len(self._labels)
                self._track_ids[idx] = after
            elif field == "metadata":
                if self._metas is None:
                    self._metas = [None] * len(self._labels)
                self._metas[idx] = box._metadata
        if self._owner is not None:
            self._owner._changed(self, "update_box", {"uid": box.uid, "field": field})

    def _signature(self) -> tuple[Any, ...]:
        n = len(self._labels)
        if n == 0:
            return ()
        cids = tuple(self.class_ids)
        tids = self._track_ids
        metas = self._metas
        cache = self._boxes_cache
        has_extra = (
            (tids is not None and any(t is not None for t in tids))
            or (metas is not None and any(bool(m) for m in metas))
            or (cache is not None and any(b is not None and (b._track_id is not None or bool(b._metadata)) for b in cache))
        )
        if has_extra:
            extra = tuple(
                (
                    (cache[i]._track_id if cache is not None and cache[i] is not None else (None if tids is None else tids[i])),
                    repr(
                        (cache[i]._metadata or {})
                        if cache is not None and cache[i] is not None
                        else ({} if metas is None or metas[i] is None else metas[i])
                    ),
                )
                for i in range(n)
            )
        else:
            extra = ()
        return (
            self._uids.tobytes(),
            tuple(self._labels),
            cids,
            self._scores.tobytes(),
            self._xyxy.tobytes(),
            extra,
        )

    def _clone(self, identity: Optional[Identity] = None) -> "Detection":
        source_id, frame_index = identity or self.identity
        registry = self._owner._registry if self._owner is not None else self._registry
        label_ids = self._owner._label_ids if self._owner is not None else self._label_ids
        n = len(self._labels)
        if n == 0:
            empty_xyxy, empty_scores, empty_ints = _empty_arrays(self._precision)
            return Detection._from_arrays(
                empty_xyxy,
                empty_scores,
                empty_ints,
                empty_ints,
                [],
                source_id=source_id,
                frame_index=frame_index,
                source_path=self.source_path,
                image_size=self.image_size,
                source=self._source,
                metadata=self.metadata,
                registry=registry,
                label_ids=label_ids,
                precision=self._precision,
                restore_trace=self._restore_trace,
            )
        int_dtype = np.int32 if self._precision == 32 else np.int64
        cids = np.asarray(self.class_ids, dtype=int_dtype) if self._owner is not None else self._class_ids.copy()
        cache = self._boxes_cache
        tids = None
        if cache is not None and any(b is not None and b._track_id is not None for b in cache):
            tids = [
                (cache[i]._track_id if cache[i] is not None else (None if self._track_ids is None else self._track_ids[i]))
                for i in range(n)
            ]
        elif self._track_ids is not None:
            tids = list(self._track_ids)

        metas = None
        if cache is not None and any(b is not None and b._metadata for b in cache):
            metas = [
                (deepcopy(cache[i]._metadata) if cache[i] is not None and cache[i]._metadata else
                 (deepcopy(self._metas[i]) if self._metas is not None and self._metas[i] else None))
                for i in range(n)
            ]
        elif self._metas is not None:
            metas = [deepcopy(m) if m else None for m in self._metas]

        return Detection._from_arrays(
            self._xyxy.copy(),
            self._scores.copy(),
            cids,
            self._uids.copy(),
            list(self._labels),
            track_ids=tids,
            metas=metas,
            source_id=source_id,
            frame_index=frame_index,
            source_path=self.source_path,
            image_size=self.image_size,
            source=self._source,
            metadata=self.metadata,
            registry=registry,
            label_ids=label_ids,
            precision=self._precision,
            restore_trace=self._restore_trace,
        )

    def add_box(self, coordinates: Iterable[Any], *, label: str, score: float = 1.0,
                track_id: Optional[int | str] = None,
                metadata: Optional[Mapping[str, Any]] = None) -> None:
        """Create and append one box."""
        label = _label(label)
        coords = _coords(coordinates)
        score = _score(score)
        track_id = Box._track(track_id)
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")
        meta = deepcopy(dict(metadata)) if metadata else None

        self._before_change()
        self._ensure_private_registry()
        class_id = self._class(label)
        if class_id not in self._registry:
            self._registry[class_id] = label
            self._label_ids[label] = class_id
        uid = self._next_uid
        self._next_uid = uid + 1

        n = len(self._labels)
        float_dtype = np.float32 if self._precision == 32 else np.float64
        int_dtype = np.int32 if self._precision == 32 else np.int64
        if self._precision == 32:
            if class_id > np.iinfo(np.int32).max or uid > np.iinfo(np.int32).max:
                raise OverflowError("class ID and UID must fit in int32")
            if any(abs(value) > np.finfo(np.float32).max for value in coords):
                raise OverflowError("box coordinates must fit in float32")
        if n == 0:
            self._xyxy = np.array([coords], dtype=float_dtype)
            self._scores = np.array([score], dtype=float_dtype)
            self._class_ids = np.array([class_id], dtype=int_dtype)
            self._uids = np.array([uid], dtype=int_dtype)
        else:
            self._xyxy = np.vstack((self._xyxy, np.asarray([coords], dtype=float_dtype)))
            self._scores = np.concatenate((self._scores, np.asarray([score], dtype=float_dtype)))
            self._class_ids = np.concatenate((self._class_ids, np.asarray([class_id], dtype=int_dtype)))
            self._uids = np.concatenate((self._uids, np.asarray([uid], dtype=int_dtype)))
        self._labels.append(label)
        if track_id is not None or self._track_ids is not None:
            if self._track_ids is None:
                self._track_ids = [None] * n
            self._track_ids.append(track_id)
        if meta is not None or self._metas is not None:
            if self._metas is None:
                self._metas = [None] * n
            self._metas.append(meta)
        if self._boxes_cache is not None:
            self._boxes_cache.append(None)
        if self._uid_to_idx is not None:
            self._uid_to_idx[uid] = n
        if self._owner is not None:
            self._owner._changed(self, "append_box", {"count": n + 1})

    def append(self, box: Box) -> None:
        """Append one box."""
        self._commit([*list(self), box], "append_box")

    def extend(self, boxes: Iterable[Box]) -> None:
        """Append boxes atomically."""
        incoming = list(boxes)
        if any(not isinstance(b, Box) for b in incoming):
            raise TypeError("every box must be a Box")
        self._commit([*list(self), *incoming], "extend_boxes")

    def insert(self, index: int, box: Box) -> None:
        """Insert a box by position."""
        validators.validate_type(box, Box, "box")
        candidates = list(self)
        candidates.insert(index, box)
        self._commit(candidates, "insert_box")

    def replace(self, index: int, box: Box) -> None:
        """Replace a box while preserving its UID."""
        validators.validate_type(box, Box, "box")
        n = len(self._labels)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError("list index out of range")
        self._before_change()
        self._ensure_private_registry()
        label = box._label
        class_id = self._label_ids.get(label)
        if class_id is None:
            preferred = box._class_id
            class_id = (
                preferred
                if preferred is not None and preferred not in self._registry
                else max(self._registry, default=-1) + 1
            )
            self._registry[class_id] = label
            self._label_ids[label] = class_id
        self._xyxy[index] = box._box
        self._scores[index] = box._score
        self._class_ids[index] = class_id
        self._labels[index] = label
        if box._track_id is not None or self._track_ids is not None:
            if self._track_ids is None:
                self._track_ids = [None] * n
            self._track_ids[index] = box._track_id
        if box._metadata or self._metas is not None:
            if self._metas is None:
                self._metas = [None] * n
            self._metas[index] = deepcopy(box._metadata) if box._metadata else None
        if self._boxes_cache is not None:
            old_box = self._boxes_cache[index]
            if old_box is not None:
                old_box._owner = None
            self._boxes_cache[index] = None
        if self._owner is not None:
            self._owner._changed(self, "replace_box", {"count": n})

    def pop(self, index: int = -1) -> Box:
        """Remove and return a box by position."""
        n = len(self._labels)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError("pop index out of range")
        item = self._get_box_at(index)
        keep = [i for i in range(n) if i != index]
        self._slice_keep(keep, "pop_box")
        item._owner = None
        return item

    def remove_box(self, uid: int) -> None:
        """Remove a box by stable UID."""
        if self._uid_to_idx is None:
            self._uid_to_idx = {u: i for i, u in enumerate(self._uids.tolist())}
        try:
            target_idx = self._uid_to_idx[uid]
        except KeyError as error:
            raise KeyError(f"box UID {uid} was not found") from error
        keep = [i for i in range(len(self._labels)) if i != target_idx]
        self._slice_keep(keep, "remove_box")

    def clear(self) -> "Detection":
        """Remove every box."""
        self._slice_keep([], "clear_boxes")
        return self

    def filter(self, predicate: Callable[[Box], bool]) -> "Detection":
        """Remove boxes rejected by a predicate."""
        if not callable(predicate):
            raise TypeError("predicate must be callable")
        kept_indices = []
        for i in range(len(self._labels)):
            if bool(predicate(self._detached_box_at(i))):
                kept_indices.append(i)
        self._slice_keep(kept_indices, "filter_boxes")
        return self

    def map(self, function: Callable[[Box], Optional[Box]]) -> None:
        """Transform boxes atomically using detached drafts."""
        if not callable(function):
            raise TypeError("function must be callable")
        transformed = []
        uids = self._uids.tolist()
        for i in range(len(self._labels)):
            draft = self._detached_box_at(i)
            result = function(draft)
            result = draft if result is None else result
            validators.validate_type(result, Box, "mapped box")
            result._uid = uids[i]
            transformed.append(result)
        self._commit(transformed, "map_boxes")

    def drop(self, labels: Iterable[str]) -> "Detection":
        """Remove boxes with matching labels."""
        if isinstance(labels, str):
            labels = (labels,)
        blocked = {_label(label) for label in labels}
        keep = [i for i, lbl in enumerate(self._labels) if lbl not in blocked]
        self._slice_keep(keep, "drop_labels")
        return self

    def relabel(self, mapping: Mapping[str, str]) -> "Detection":
        """Rename or merge labels."""
        if not isinstance(mapping, Mapping):
            raise TypeError("mapping must be a mapping")
        mapping = {_label(a): _label(b) for a, b in mapping.items()}
        self._before_change()
        new_labels = [mapping.get(lbl, lbl) for lbl in self._labels]
        registry: Names = {}
        label_ids: dict[str, int] = {}
        new_cids: list[int] = []
        for lbl in new_labels:
            cid = label_ids.get(lbl)
            if cid is None:
                cid = len(registry)
                registry[cid] = lbl
                label_ids[lbl] = cid
            new_cids.append(cid)
        if self._boxes_cache is not None:
            for b in self._boxes_cache:
                if b is not None:
                    b._owner = None
            self._boxes_cache = None
        self._labels = new_labels
        int_dtype = np.int32 if self._precision == 32 else np.int64
        self._class_ids = np.asarray(new_cids, dtype=int_dtype) if new_cids else _empty_arrays(self._precision)[2]
        self._registry = registry
        self._label_ids = label_ids
        if self._owner is not None:
            self._owner._changed(self, "relabel", {"count": len(self._labels)})
        return self

    def _validate_precision(self, dtype: int) -> int:
        bits = _precision(dtype)
        if bits == 64:
            return bits
        int_limit = np.iinfo(np.int32).max
        if self._next_uid > int_limit:
            raise OverflowError("UIDs must fit in int32")
        if self._class_ids.size and int(self._class_ids.max()) > int_limit:
            raise OverflowError("class IDs must fit in int32")
        if self._uids.size and int(self._uids.max()) > int_limit:
            raise OverflowError("UIDs must fit in int32")
        if self._xyxy.size and np.any(np.abs(self._xyxy) > np.finfo(np.float32).max):
            raise OverflowError("box coordinates must fit in float32")
        return bits

    def astype(self, dtype: int) -> None:
        """Convert numeric annotation storage to 32 or 64 bits in-place."""
        bits = self._validate_precision(dtype)
        if bits == self._precision:
            return
        float_dtype = np.float32 if bits == 32 else np.float64
        int_dtype = np.int32 if bits == 32 else np.int64
        self._xyxy = self._xyxy.astype(float_dtype)
        self._scores = self._scores.astype(float_dtype)
        self._class_ids = self._class_ids.astype(int_dtype)
        self._uids = self._uids.astype(int_dtype)
        self._precision = bits
        if self._boxes_cache is not None:
            for index, box in enumerate(self._boxes_cache):
                if box is None:
                    continue
                row = self._xyxy[index]
                box._box = tuple(float(value) for value in row)
                box._score = float(self._scores[index])
                box._class_id = int(self._class_ids[index])
                box._uid = int(self._uids[index])

    def _resolve_source(self) -> Any:
        if self._source is not None: return self._source
        if self.source_path is None: raise RuntimeError("Cannot access frame pixels: source image is unavailable")
        from klygo import media
        loaded = media.load(self.source_path, verbose=False)
        for item in loaded:
            if len(loaded) == 1 or getattr(item, "frame_index", None) == self.frame_index:
                self._source = item; return item
        raise RuntimeError("Cannot access frame pixels: source frame is unavailable")

    def _render(self, mode: str = "default", *, show_scores: bool = True,
                line_width: Optional[int] = None, font_size: Optional[int] = None) -> Any:
        if mode not in {"default", "index", "uid"}:
            raise ValueError("mode must be 'default', 'index', or 'uid'")
        from klygo import visual
        labels = (
            self.labels
            if mode == "default"
            else ([f"{i} {label}" for i, label in enumerate(self._labels)]
                  if mode == "index"
                  else [f"{uid} {label}" for uid, label in zip(self._uids.tolist(), self._labels)])
        )
        return visual.draw_bboxes(
            self._resolve_source(), self.boxes, labels=labels, scores=self.scores,
            line_width=line_width, font_size=font_size,
            show_labels=True, show_scores=show_scores,
        )

    def show(self, mode: str = "default", *, show_scores: bool = True,
             line_width: Optional[int] = None, font_size: Optional[int] = None,
             width: Optional[int] = None, title: str = "Detection", backend: str = "auto") -> None:
        """Display all boxes using default, positional-index, or UID captions."""
        from klygo import visual
        annotated = self._render(mode, show_scores=show_scores, line_width=line_width, font_size=font_size)
        visual.show_image(annotated, title=title, width=width, backend=backend)

    def save(self, output_path: Any, mode: str = "default", *, show_scores: bool = True,
             line_width: Optional[int] = None, font_size: Optional[int] = None,
             overwrite: bool = False, verbose: bool = False) -> Any:
        """Draw this frame's boxes and save the annotated image."""
        from klygo import media
        annotated = self._render(
            mode, show_scores=show_scores, line_width=line_width, font_size=font_size,
        )
        return media.save(output_path, annotated, overwrite=overwrite, verbose=verbose)

    def preview_box(self, box: Box | Iterable[Any], *, label: Optional[str] = None,
                    score: float = 1.0, show_existing: bool = True,
                    show_scores: bool = True,
                    preview_color: tuple[int, int, int] = (255, 0, 0),
                    line_width: Optional[int] = None,
                    font_size: Optional[int] = None,
                    width: Optional[int] = None, title: str = "Box preview",
                    backend: str = "auto") -> Any:
        """Display and return a non-mutating preview of a proposed box."""
        if isinstance(box, Box):
            candidate = box
        else:
            if label is None:
                raise ValueError("label is required when previewing coordinates")
            candidate = Box(box, label=label, score=score)
        if (not isinstance(preview_color, tuple) or len(preview_color) != 3
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or not 0 <= value <= 255 for value in preview_color)):
            raise ValueError("preview_color must be an RGB tuple with values from 0 to 255")

        from klygo import visual
        source = self._resolve_source()
        annotated = source
        if show_existing and self:
            annotated = visual.draw_bboxes(
                source, self.boxes, labels=self.labels, scores=self.scores,
                line_width=line_width, font_size=font_size,
                show_labels=True, show_scores=show_scores,
            )
        annotated = visual.draw_bboxes(
            annotated, [candidate.box], labels=[f"[preview] {candidate.label}"],
            scores=[candidate.score], line_width=line_width, font_size=font_size,
            box_color=preview_color, show_labels=True, show_scores=show_scores,
        )
        visual.show_image(annotated, title=title, width=width, backend=backend)
        return annotated

    def to_dict(self) -> dict[str, Any]:
        """Serialize annotations without decoding pixels."""
        n = len(self._labels)
        if n == 0:
            boxes_dicts = []
        else:
            cids = self.class_ids
            uids = self._uids.tolist()
            scores = self._scores.tolist()
            coords_list = self._xyxy.tolist()
            tids = self._track_ids
            metas = self._metas
            cache = self._boxes_cache
            boxes_dicts = []
            for i in range(n):
                b_obj = None if cache is None else cache[i]
                meta = (
                    deepcopy(b_obj._metadata)
                    if b_obj is not None and b_obj._metadata
                    else (deepcopy(metas[i]) if metas is not None and metas[i] else {})
                )
                tid = b_obj._track_id if b_obj is not None else (None if tids is None else tids[i])
                boxes_dicts.append({
                    "uid": uids[i],
                    "class_id": cids[i],
                    "label": self._labels[i],
                    "score": scores[i],
                    "box": coords_list[i],
                    "track_id": tid,
                    "metadata": meta,
                })
        return {"source_id": self.source_id, "frame_index": self.frame_index,
                "source_path": self.source_path, "image_size": list(self.image_size),
                "metadata": deepcopy(self.metadata), "names": self.names,
                "boxes": boxes_dicts}

    def _mark_restored(self) -> None:
        """Mark coordinates as restored without applying the trace again."""
        self._restore_trace = _RESTORED

    def restore(self, trace: Optional[Iterable[Mapping[str, Any]]] = None) -> None:
        """Restore this frame's box coordinates to the original image space.

        Parameters
        ----------
        trace : iterable of mappings, optional
            Preprocessing geometry history. By default, use the trace carried
            by the source image.
        """
        if self._restore_trace is _RESTORED:
            return
        active = trace if trace is not None else self._restore_trace
        if active is None:
            active = getattr(self._source, "_preprocessing_trace", None)
        if active is None:
            return
        try:
            steps = list(active)
        except TypeError as error:
            raise TypeError("trace must be an iterable of mappings or None") from error
        if not steps:
            return
        if any(not isinstance(step, Mapping) for step in steps):
            raise TypeError("trace entries must be mappings")
        if "before" not in steps[0]:
            raise ValueError("the first trace entry must contain the original image size")

        from klygo import preprocessing

        restored = preprocessing.restore_boxes(self._xyxy, steps)
        image_size = _size(steps[0]["before"])
        dtype = np.float32 if self._precision == 32 else np.float64
        xyxy = np.asarray(restored, dtype=dtype).reshape((-1, 4))
        if self._owner is not None:
            self._owner._snapshot_frame_baseline(self)
        self._xyxy = xyxy
        self._image_size = image_size
        self._restore_trace = _RESTORED
        # A lazy source still points at the original file, so discard queued
        # preprocessing as coordinates return to that original pixel space.
        if self._source.__class__.__name__ == "LazyImage" and hasattr(self._source, "path"):
            from klygo import media

            self._source = media.LazyImage(
                self._source.path,
                backend=getattr(self._source, "backend", "pil"),
                frame_index=getattr(self._source, "frame_index", None),
                reader=getattr(self._source, "reader", None),
            )
        if self._boxes_cache is not None:
            for index, box in enumerate(self._boxes_cache):
                if box is not None:
                    row = xyxy[index]
                    box._box = tuple(float(value) for value in row)
        if self._owner is not None:
            self._owner._changed(self, "restore", {"box_count": len(self), "image_size": image_size})

    def transform(self, *operations: Any) -> "Detection":
        """Apply one or more postprocessing operations or a callable pipeline in-place.

        Parameters
        ----------
        *operations : callable or iterable of callables
            One or more callables, an iterable of callables, or a postprocessing Pipeline.

        Returns
        -------
        Detection
            Self with in-place changes applied.
        """
        if not operations:
            raise ValueError("At least one operation or pipeline must be provided")
        if len(operations) == 1:
            op = operations[0]
            if callable(op):
                op(self)
                return self
            if isinstance(op, Iterable):
                from klygo.postprocessing import compose
                compose(*op)(self)
                return self
            raise TypeError("Operation must be callable")
        from klygo.postprocessing import compose
        compose(*operations)(self)
        return self


class Detections:
    """Hold a materialized collection or a bounded-memory frame stream."""

    def __init__(self, frames: Iterable[Detection] = (), *, metadata: Optional[Mapping[str, Any]] = None,
                 names: Optional[Mapping[int, Any] | Sequence[Any]] = None,
                 stream: bool = False) -> None:
        if metadata is not None and not isinstance(metadata, Mapping): raise TypeError("metadata must be a mapping or None")
        validators.validate_type(stream, bool, "stream")
        self.metadata = deepcopy(dict(metadata or {})); initial_names = _names(names)
        self._registry = {} if stream else initial_names
        self._label_ids = {label: class_id for class_id, label in self._registry.items()}
        self._next_class_id = max(self._registry, default=-1) + 1
        self._frames: list[Detection] = []; self._identity_map: dict[Identity, Detection] = {}
        self._frame_labels: dict[Identity, set[str]] = {}
        self._label_uses: dict[str, int] = {}
        self._label_frames: dict[str, set[Identity]] = {}
        self._edit_depth = 0
        self._edit_dirty: dict[Identity, Detection] = {}
        self._edit_changes = 0
        self._edit_operation = "edit"
        self._history: list[dict[str, Any]] = []; self._sequence = 0
        self._stream_mode = stream
        self._stream = iter(frames) if stream else None
        self._stream_active = False
        self._stream_started = False
        self._stream_closed = False
        self._preferred_label_ids = {label: class_id for class_id, label in initial_names.items()}
        self._baseline_identities: tuple[Identity, ...] = ()
        self._baseline_set: set[Identity] = set()
        self._baseline_signatures: dict[Identity, tuple[Any, ...]] = {}
        if stream:
            self._total_frames = self.metadata.get("requested_total_frames")
        else:
            self._commit(list(frames), None)
            self._total_frames = len(self)
            self._baseline_identities = tuple(frame.identity for frame in self._frames)
            self._baseline_set = set(self._baseline_identities)

    def _snapshot_frame_baseline(self, frame: Detection) -> None:
        identity = frame.identity
        if identity in self._baseline_set and identity not in self._baseline_signatures:
            self._baseline_signatures[identity] = frame._signature()

    def _record(self, operation: str, **details: Any) -> None:
        self._sequence += 1
        self._history.append({"sequence_number": self._sequence, "operation": operation, **deepcopy(details)})

    def _class(self, label: str) -> int:
        found = self._label_ids.get(label)
        if found is not None:
            return found
        preferred = self._preferred_label_ids.get(label)
        found = preferred if preferred is not None and preferred not in self._registry else self._next_class_id
        while found in self._registry:
            found += 1
        self._next_class_id = max(self._next_class_id, found + 1)
        self._registry[found] = label
        self._label_ids[label] = found
        return found

    def _reconcile(self, incoming: Detection, identity: Optional[Identity] = None) -> Detection:
        validators.validate_type(incoming, Detection, "frame")
        frame = incoming if incoming._owner in (None, self) else incoming._clone()
        if identity is not None and frame.identity != identity:
            frame = frame._clone(identity)
        n = len(frame._labels)
        if n > 0:
            cids = frame._class_ids
            changed_cids = False
            new_cids = None
            for i, label in enumerate(frame._labels):
                found = self._label_ids.get(label)
                if found is None:
                    preferred = int(cids[i])
                    if preferred < 0:
                        preferred = self._preferred_label_ids.get(label)
                    if preferred is not None and preferred >= 0 and preferred not in self._registry:
                        found = preferred
                    else:
                        found = self._next_class_id
                        while found in self._registry:
                            found += 1
                    self._registry[found] = label
                    self._label_ids[label] = found
                    self._next_class_id = max(self._next_class_id, found + 1)
                if int(cids[i]) != found:
                    if frame.dtype == 32 and found > np.iinfo(np.int32).max:
                        raise OverflowError("class IDs must fit in int32")
                    if not changed_cids:
                        new_cids = cids.copy()
                        changed_cids = True
                    new_cids[i] = found
            if changed_cids and new_cids is not None:
                frame._class_ids = new_cids
                if frame._boxes_cache is not None:
                    for i, b in enumerate(frame._boxes_cache):
                        if b is not None:
                            b._class_id = int(new_cids[i])
        frame._registry = self._registry
        frame._label_ids = self._label_ids
        return frame

    def _prepare(self, frames: Iterable[Detection]) -> list[Detection]:
        frames = list(frames)
        if any(not isinstance(frame, Detection) for frame in frames):
            raise TypeError("every frame must be a Detection")
        raw_identities = [frame.identity for frame in frames]
        if len(raw_identities) != len(set(raw_identities)):
            raise ValueError("duplicate frame identity")
        registry, label_ids = dict(self._registry), dict(self._label_ids)
        next_class_id = self._next_class_id
        result, identities = [], set()
        try:
            for incoming in frames:
                frame = self._reconcile(incoming)
                if frame.identity in identities:
                    raise ValueError(f"duplicate frame identity: {frame.identity!r}")
                identities.add(frame.identity)
                result.append(frame)
        except Exception:
            self._registry, self._label_ids = registry, label_ids
            self._next_class_id = next_class_id
            raise
        return result

    def _synchronize_registry(self) -> None:
        """Reconcile active labels after direct mutations on owned frames."""
        ordered: dict[Identity, set[str]] = {}
        uses: dict[str, int] = {}
        label_frames: dict[str, set[Identity]] = {}
        labels: list[str] = []
        seen: set[str] = set()
        for frame in self._frames:
            active: set[str] = set()
            for label in frame._labels:
                active.add(label)
                if label not in seen:
                    seen.add(label)
                    labels.append(label)
            ordered[frame.identity] = active
            for label in active:
                uses[label] = uses.get(label, 0) + 1
                label_frames.setdefault(label, set()).add(frame.identity)
        previous = {label: class_id for class_id, label in self._registry.items()}
        reserved = {previous[label] for label in labels if label in previous}
        registry: Names = {}
        used: set[int] = set()

        for label in labels:
            class_id = previous.get(label)
            if class_id is None or class_id in used:
                while self._next_class_id in used or self._next_class_id in reserved:
                    self._next_class_id += 1
                class_id = self._next_class_id
                self._next_class_id += 1
            registry[class_id] = label
            used.add(class_id)

        self._registry = registry
        self._label_ids = {label: class_id for class_id, label in registry.items()}
        for frame in self._frames:
            frame._registry = self._registry
            frame._label_ids = self._label_ids
        self._frame_labels = ordered
        self._label_uses = uses
        self._label_frames = label_frames

    def _refresh_frame_labels(self, frame: Detection) -> None:
        self._refresh_frames((frame,))

    def _refresh_frames(self, frames: Iterable[Detection]) -> None:
        changes = []
        for frame in dict.fromkeys(frames):
            identity = frame.identity
            before = self._frame_labels.get(identity, set())
            after = set(frame._labels)
            changes.append((identity, before, after))

        for identity, before, after in changes:
            for label in before - after:
                self._label_uses[label] -= 1
                self._label_frames[label].discard(identity)
            for label in after - before:
                self._label_uses[label] = self._label_uses.get(label, 0) + 1
                self._label_frames.setdefault(label, set()).add(identity)
            self._frame_labels[identity] = after

        changed_labels = set().union(
            *(before ^ after for _identity, before, after in changes)
        ) if changes else set()
        for label in changed_labels:
            if self._label_uses.get(label, 0) == 0:
                self._label_uses.pop(label, None)
                self._label_frames.pop(label, None)
                class_id = self._label_ids.pop(label)
                del self._registry[class_id]
            elif label not in self._label_ids:
                self._class(label)

    def _resolve_class_id(self, label: str, fallback: int) -> int:
        return self._label_ids.get(label, fallback)

    def _commit(self, frames: Iterable[Detection], operation: Optional[str]) -> None:
        prepared = self._prepare(frames)
        prepared_ids = {id(f) for f in prepared}
        for frame in self._frames:
            if id(frame) not in prepared_ids:
                frame._registry = dict(self._registry)
                frame._label_ids = dict(self._label_ids)
                frame._owner = None
        self._frames = prepared
        self._identity_map = {f.identity: f for f in prepared}
        for frame in prepared:
            frame._owner = self
        self._synchronize_registry()
        self._edit_dirty.clear()
        if operation:
            self._record(operation, count=len(prepared))

    def _insert_frame(self, index: int, frame: Detection, operation: str) -> None:
        validators.validate_type(frame, Detection, "frame")
        if frame.identity in self._identity_map:
            raise ValueError("frame identity already exists")
        frame = self._reconcile(frame)
        self._frames.insert(index, frame)
        self._identity_map[frame.identity] = frame
        frame._owner = self
        self._refresh_frame_labels(frame)
        self._record(operation, count=len(self))

    def _pop_frame(self, index: int, operation: str) -> Detection:
        frame = self._frames.pop(index)
        self._edit_dirty.pop(frame.identity, None)
        registry = dict(self._registry)
        label_ids = dict(self._label_ids)
        del self._identity_map[frame.identity]
        for label in self._frame_labels.pop(frame.identity, set()):
            uses = self._label_uses[label] - 1
            frames = self._label_frames[label]
            frames.discard(frame.identity)
            if uses:
                self._label_uses[label] = uses
            else:
                del self._label_uses[label]
                del self._label_frames[label]
                class_id = self._label_ids.pop(label)
                del self._registry[class_id]
        frame._registry = registry
        frame._label_ids = label_ids
        frame._owner = None
        self._record(operation, count=len(self))
        return frame

    def _changed(self, frame: Detection, operation: str, details: Mapping[str, Any]) -> None:
        labels_unchanged = operation in ("update_box", "restore")
        if self._edit_depth:
            self._edit_changes += 1
            if not labels_unchanged:
                self._edit_dirty[frame.identity] = frame
            return
        if not labels_unchanged:
            self._refresh_frame_labels(frame)
        self._record(operation, target_identity=frame.identity, changed_fields=dict(details))

    @contextmanager
    def _batch(self, operation: str) -> Iterator["Detections"]:
        outermost = self._edit_depth == 0
        if outermost:
            self._edit_operation = operation
        self._edit_depth += 1
        try:
            yield self
        finally:
            self._edit_depth -= 1
            if outermost:
                dirty = tuple(self._edit_dirty.values())
                changes = self._edit_changes
                self._edit_dirty.clear()
                self._edit_changes = 0
                active = [
                    frame for frame in dirty
                    if self._identity_map.get(frame.identity) is frame
                ]
                self._refresh_frames(active)
                if changes:
                    self._record(self._edit_operation, changed_frames=len(dirty), changes=changes)

    def _require_materialized(self, operation: str) -> None:
        if self._stream_mode:
            raise TypeError(f"{operation} is unavailable for a stream; predict with stream=False")

    def _transform_stream(self, function: Callable[[Detection], Optional[Detection]]) -> None:
        if not self._stream_mode:
            raise TypeError("stream transforms require streaming detections")
        if self._stream_started:
            raise RuntimeError("cannot add a stream transform after consumption has started")
        source = self._stream

        def transformed() -> Iterator[Detection]:
            if source is None:
                return
            for frame in source:
                result = function(frame)
                yield frame if result is None else result

        self._stream = iter(transformed())

    def _iterate_stream(self) -> Iterator[Detection]:
        if self._stream_active:
            raise RuntimeError("detection stream is already being consumed")
        if self._stream_closed or self._stream is None:
            return
        self._stream_active = True
        self._stream_started = True
        try:
            while self._stream is not None:
                try:
                    incoming = next(self._stream)
                except StopIteration:
                    self.close()
                    break
                except Exception:
                    self.close()
                    raise
                try:
                    validators.validate_type(incoming, Detection, "stream frame")
                    frame = self._reconcile(incoming)
                    frame._registry = dict(self._registry)
                    frame._label_ids = dict(self._label_ids)
                    frame._owner = None
                    yield frame
                except Exception:
                    self.close()
                    raise
        finally:
            self._stream_active = False

    @property
    def is_stream(self) -> bool:
        """Whether this collection was created as a one-pass stream."""
        return self._stream_mode

    @property
    def stream_exhausted(self) -> bool:
        """Whether a streaming collection is closed or fully consumed."""
        return self._stream_mode and self._stream_closed

    def close(self) -> None:
        """Close a streaming collection and its underlying source."""
        if not self._stream_mode or self._stream_closed:
            return
        source, self._stream = self._stream, None
        self._stream_closed = True
        close = getattr(source, "close", None)
        if callable(close):
            close()

    count = property(lambda self: len(self), doc="Current frame count.")
    total_frames = property(lambda self: self._total_frames, doc="Initial frame count.")
    total_objects = property(lambda self: self._total_objects(), doc="Current object count.")
    names = property(
        lambda self: dict(self._registry),
        doc="Active collection class IDs mapped to labels.",
    )
    history = property(lambda self: tuple(deepcopy(self._history)), doc="Read-only history.")
    added_frames = property(lambda self: self._added_frames(), doc="Added identities.")
    deleted_frames = property(lambda self: self._deleted_frames(), doc="Deleted identities.")
    modified_frames = property(lambda self: self._modified_frames(), doc="Modified identities.")

    def _total_objects(self) -> int:
        self._require_materialized("total_objects")
        return sum(len(frame._labels) for frame in self._frames)

    def _added_frames(self) -> tuple[Identity, ...]:
        self._require_materialized("added_frames")
        return tuple(identity for identity in self._identity_map if identity not in self._baseline_set)

    def _deleted_frames(self) -> tuple[Identity, ...]:
        self._require_materialized("deleted_frames")
        return tuple(identity for identity in self._baseline_identities if identity not in self._identity_map)

    def _modified_frames(self) -> tuple[Identity, ...]:
        self._require_materialized("modified_frames")
        return tuple(
            identity for identity in self._baseline_identities
            if identity in self._identity_map
            and identity in self._baseline_signatures
            and self._identity_map[identity]._signature() != self._baseline_signatures[identity]
        )

    def __len__(self) -> int:
        self._require_materialized("len()")
        return len(self._frames)

    def __iter__(self) -> Iterator[Detection]:
        return self._iterate_stream() if self._stream_mode else iter(self._frames)

    def __getitem__(self, index: int | slice) -> Detection | tuple[Detection, ...]:
        self._require_materialized("indexing")
        return tuple(self._frames[index]) if isinstance(index, slice) else self._frames[index]

    def copy(self) -> "Detections":
        """Return an independent annotation copy sharing lazy source handles."""
        self._require_materialized("copy()")
        return Detections(
            [frame._clone() for frame in self],
            metadata=self.metadata,
            names=self._registry,
        )

    def get_frame(self, frame_index: int, *, source_id: Optional[str] = None) -> Detection:
        """Return a frame by stable source identity."""
        self._require_materialized("get_frame()")
        if source_id is not None:
            try: return self._identity_map[(str(source_id), frame_index)]
            except KeyError as error: raise KeyError("frame identity was not found") from error
        matches = [f for f in self if f.frame_index == frame_index]
        if not matches: raise KeyError(f"frame index {frame_index} was not found")
        if len(matches) > 1: raise ValueError("frame_index is ambiguous; provide source_id")
        return matches[0]

    def append(self, frame: Detection) -> None:
        """Append a frame."""
        self._require_materialized("append()")
        self._insert_frame(len(self), frame, "append_frame")

    def extend(self, frames: Iterable[Detection]) -> None:
        """Append frames atomically."""
        self._require_materialized("extend()")
        incoming = list(frames)
        if any(not isinstance(f, Detection) for f in incoming): raise TypeError("every frame must be a Detection")
        if {f.identity for f in incoming} & set(self._identity_map): raise ValueError("frame identity already exists")
        identities = [frame.identity for frame in incoming]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate frame identity")
        registry, label_ids, next_class_id = (
            dict(self._registry),
            dict(self._label_ids),
            self._next_class_id,
        )
        try:
            prepared = [self._reconcile(frame) for frame in incoming]
        except Exception:
            self._registry, self._label_ids = registry, label_ids
            for f in self._frames:
                f._registry = self._registry
                f._label_ids = self._label_ids
            self._next_class_id = next_class_id
            raise
        self._frames.extend(prepared)
        for frame in prepared:
            self._identity_map[frame.identity] = frame
            frame._owner = self
            active = set(frame._labels)
            self._frame_labels[frame.identity] = active
            for label in active:
                self._label_uses[label] = self._label_uses.get(label, 0) + 1
                self._label_frames.setdefault(label, set()).add(frame.identity)
        self._record("extend_frames", count=len(self))

    def insert(self, index: int, frame: Detection) -> None:
        """Insert a frame by position."""
        self._require_materialized("insert()")
        self._insert_frame(index, frame, "insert_frame")

    def replace(self, index: int, frame: Detection) -> None:
        """Replace content while preserving frame identity."""
        self._require_materialized("replace()")
        validators.validate_type(frame, Detection, "frame")
        original = self._frames[index]
        self._snapshot_frame_baseline(original)
        replacement = self._reconcile(frame._clone(original.identity))
        self._edit_dirty.pop(original.identity, None)
        original._registry = dict(self._registry)
        original._label_ids = dict(self._label_ids)
        original._owner = None
        self._frames[index] = replacement
        self._identity_map[original.identity] = replacement
        replacement._owner = self
        self._refresh_frame_labels(replacement)
        self._record("replace_frame", count=len(self))

    def pop(self, index: int = -1) -> Detection:
        """Remove and return a frame by position."""
        self._require_materialized("pop()")
        return self._pop_frame(index, "pop_frame")

    def remove_frame(self, frame_index: int, *, source_id: Optional[str] = None) -> None:
        """Remove a frame by source identity."""
        self._require_materialized("remove_frame()")
        target = self.get_frame(frame_index, source_id=source_id)
        self._pop_frame(self._frames.index(target), "remove_frame")

    def clear(self) -> "Detections":
        """Remove every frame."""
        self._require_materialized("clear()")
        self._commit([], "clear_frames")
        return self

    def merge(self, other: "Detections") -> None:
        """Merge another collection, rejecting any identity collision."""
        self._require_materialized("merge()")
        validators.validate_type(other, Detections, "other")
        other._require_materialized("merge()")
        self.extend(f._clone() for f in other); self._history[-1]["operation"] = "merge"

    def filter(self, predicate: Callable[[Detection], bool]) -> "Detections":
        """Remove frames rejected by a predicate."""
        if not callable(predicate): raise TypeError("predicate must be callable")
        if self._stream_mode:
            if self._stream_started:
                raise RuntimeError("cannot add a stream filter after consumption has started")
            source = self._stream

            def filtered() -> Iterator[Detection]:
                if source is None:
                    return
                for original in source:
                    validators.validate_type(original, Detection, "stream frame")
                    if bool(predicate(original._clone())):
                        yield original

            self._stream = iter(filtered())
            self._total_frames = None
            return self
        kept = []
        for frame in self:
            if bool(predicate(frame._clone())):
                kept.append(frame)
        self._commit(kept, "filter_frames")
        return self

    def map(self, function: Callable[[Detection], Optional[Detection]]) -> None:
        """Transform frames atomically using detached drafts."""
        if not callable(function): raise TypeError("function must be callable")
        if self._stream_mode:
            def transform(original: Detection) -> Detection:
                validators.validate_type(original, Detection, "stream frame")
                draft = original._clone()
                result = function(draft)
                result = draft if result is None else result
                validators.validate_type(result, Detection, "mapped frame")
                return draft if result is draft else result._clone(original.identity)
            self._transform_stream(transform)
            return
        transformed = []
        for original in self:
            draft = original._clone()
            result = function(draft)
            result = draft if result is None else result
            validators.validate_type(result, Detection, "mapped frame")
            final_frame = draft if (result is draft and draft.identity == original.identity and draft._owner is None) else result._clone(original.identity)
            transformed.append(final_frame)
        for original in self._frames:
            self._snapshot_frame_baseline(original)
        self._commit(transformed, "map_frames")

    def drop(self, labels: Iterable[str]) -> "Detections":
        """Remove matching boxes from every frame."""
        if isinstance(labels, str):
            labels = (labels,)
        blocked = {_label(label) for label in labels}
        if self._stream_mode:
            def transform(frame: Detection) -> Detection:
                frame.drop(blocked)
                return frame
            self._transform_stream(transform)
            return self
        identities: set[Identity] = set()
        for label in blocked:
            identities.update(self._label_frames.get(label, ()))
        with self._batch("drop_labels"):
            for identity in identities:
                self._identity_map[identity].drop(blocked)
        return self

    def relabel(self, mapping: Mapping[str, str]) -> "Detections":
        """Rename or merge labels across all frames."""
        if not isinstance(mapping, Mapping):
            raise TypeError("mapping must be a mapping")
        mapping = {_label(source): _label(target) for source, target in mapping.items()}
        if self._stream_mode:
            def transform(frame: Detection) -> Detection:
                frame.relabel(mapping)
                return frame
            self._transform_stream(transform)
            return self
        identities: set[Identity] = set()
        preferred: dict[str, int] = {}
        for label in mapping:
            identities.update(self._label_frames.get(label, ()))
            target = mapping[label]
            if target not in self._label_ids and label in self._label_ids:
                preferred.setdefault(target, self._label_ids[label])
        with self._batch("relabel"):
            for identity in identities:
                self._identity_map[identity].relabel(mapping)
        for label, class_id in preferred.items():
            current = self._label_ids.get(label)
            if current is not None and class_id not in self._registry:
                del self._registry[current]
                self._registry[class_id] = label
                self._label_ids[label] = class_id
        return self

    def compact(self) -> "Detections":
        """Compact active class IDs to a contiguous zero-based range."""
        self._require_materialized("compact()")
        active = sorted(self.names.items())
        mapping = {old_id: new_id for new_id, (old_id, _label) in enumerate(active)}
        if any(old_id != new_id for old_id, new_id in mapping.items()):
            for frame in self._frames:
                if frame._labels:
                    self._snapshot_frame_baseline(frame)
        registry = {mapping[old_id]: label for old_id, label in active}
        label_ids = {label: class_id for class_id, label in registry.items()}
        self._registry = registry
        self._label_ids = label_ids
        for frame in self._frames:
            frame._registry = self._registry
            frame._label_ids = self._label_ids
        self._next_class_id = len(registry)
        self._record("compact", changed_fields={"class_ids": mapping})
        return self

    def astype(self, dtype: int) -> None:
        """Convert every frame's numeric annotation storage to 32 or 64 bits."""
        bits = _precision(dtype)
        if self._stream_mode:
            def transform(frame: Detection) -> Detection:
                frame.astype(bits)
                return frame
            self._transform_stream(transform)
            return
        for frame in self._frames:
            frame._validate_precision(bits)
        changed = any(frame.dtype != bits for frame in self._frames)
        for frame in self._frames:
            frame.astype(bits)
        if changed:
            self._record("astype", dtype=bits)

    def restore(self, index: Optional[int | Iterable[int]] = None,
                trace: Optional[Iterable[Mapping[str, Any]]] = None) -> None:
        """Restore selected frames' boxes to their original image coordinates.

        Parameters
        ----------
        index : int or iterable of int, optional
            Frame positions to restore. Restore every frame when omitted.
        trace : iterable of mappings, optional
            Geometry history to apply to every selected frame. By default,
            each frame uses the trace carried by its source image.
        """
        shared_trace = trace
        if trace is not None:
            try:
                shared_trace = list(trace)
            except TypeError as error:
                raise TypeError("trace must be an iterable of mappings or None") from error

        if self._stream_mode:
            if index is not None:
                raise TypeError("indexed restore is unavailable for a stream")

            def transform(frame: Detection) -> Detection:
                frame.restore(shared_trace)
                return frame

            self._transform_stream(transform)
            return

        if index is None:
            indices = list(range(len(self._frames)))
        elif isinstance(index, bool):
            raise TypeError("restore indices must be integers")
        elif isinstance(index, Integral):
            indices = [int(index)]
        else:
            try:
                indices = list(index)
            except TypeError as error:
                raise TypeError("index must be an integer, iterable of integers, or None") from error

        normalized: list[int] = []
        for item in indices:
            if isinstance(item, bool) or not isinstance(item, Integral):
                raise TypeError("restore indices must be integers")
            position = int(item)
            position = position + len(self._frames) if position < 0 else position
            if position < 0 or position >= len(self._frames):
                raise IndexError("restore index out of range")
            if position not in normalized:
                normalized.append(position)
        if not normalized:
            return
        with self._batch("restore"):
            for position in normalized:
                self._frames[position].restore(shared_trace)

    def repredict(self, model: Any, index: Optional[int | Iterable[int]] = None, **options: Any) -> None:
        """Replace selected predictions while preserving source identities."""
        self._require_materialized("repredict()")
        if index is None:
            indices = list(range(len(self)))
        elif isinstance(index, bool):
            raise TypeError("prediction indices must be integers")
        elif isinstance(index, int):
            indices = [index]
        else:
            try:
                indices = list(index)
            except TypeError as error:
                raise TypeError("index must be an integer, iterable of integers, or None") from error
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
            original = self._frames[position]
            self._snapshot_frame_baseline(original)
            replacement = fresh._clone(original.identity)
            replacement._source, replacement._source_path, replacement._image_size = original._source, original.source_path, original.image_size
            merged_meta = deepcopy(original.metadata)
            merged_meta.update(fresh.metadata)
            replacement.metadata = merged_meta
            items[position] = replacement
        self._commit(items, "repredict")

    def to_dict(self) -> dict[str, Any]:
        """Serialize annotations without decoding pixels."""
        self._require_materialized("to_dict()")
        return {"count": self.count, "total_frames": self.total_frames,
                "total_objects": self.total_objects, "names": self.names,
                "metadata": deepcopy(self.metadata), "frames": [f.to_dict() for f in self]}

    def save_images(self, output_dir: Any, prefix: str = "frame", mode: str = "default", *,
                    extension: str = ".jpg", show_scores: bool = True,
                    line_width: Optional[int] = None, font_size: Optional[int] = None,
                    overwrite: bool = False, verbose: bool = False) -> list[Any]:
        """Draw boxes and save every frame as a numbered image sequence."""
        if mode not in {"default", "index", "uid"}:
            raise ValueError("mode must be 'default', 'index', or 'uid'")
        from klygo import media

        rendered = (
            frame._render(mode, show_scores=show_scores, line_width=line_width, font_size=font_size)
            for frame in self
        )
        return media.save_images(
            output_dir, rendered, prefix=prefix, extension=extension,
            overwrite=overwrite, verbose=verbose,
        )

    def save_video(self, output_path: Any, fps: float = 30.0, mode: str = "default", *,
                   fourcc: str = "mp4v", show_scores: bool = True,
                   line_width: Optional[int] = None, font_size: Optional[int] = None,
                   overwrite: bool = False, verbose: bool = False) -> Any:
        """Draw boxes and encode frames to video with bounded memory usage."""
        if mode not in {"default", "index", "uid"}:
            raise ValueError("mode must be 'default', 'index', or 'uid'")
        from klygo import media

        rendered = (
            frame._render(mode, show_scores=show_scores, line_width=line_width, font_size=font_size)
            for frame in self
        )
        return media.save_video(
            output_path, rendered, fps=fps, fourcc=fourcc,
            overwrite=overwrite, verbose=verbose,
        )

    def export(self, path: str, *, format: str = "yolo", overwrite: bool = False) -> None:
        """Export annotations without copying source images."""
        self._require_materialized("export()")
        format = str(format).lower()
        if format == "json": files.save(path, self.to_dict(), overwrite=overwrite, verbose=False); return
        if format != "yolo": raise ValueError("format must be 'yolo' or 'json'")
        if files.exists(path) and not overwrite: raise FileExistsError(f"output path already exists: {path}")
        if files.exists(path): files.remove(path, missing_ok=True)
        labels_dir = files.join(path, "labels"); files.mkdir(labels_dir)
        active = sorted(self.names.items())
        names = {index: label for index, (_class_id, label) in enumerate(active)}
        manifest = []
        label_ids = {label: key for key, label in names.items()}
        for position, frame in enumerate(self):
            width, height = frame.image_size
            digest = sha1(f"{frame.source_id}:{frame.frame_index}".encode()).hexdigest()[:10]
            filename = f"frame_{position:06d}_{frame.frame_index}_{digest}.txt"
            xyxy = frame._xyxy
            lines = [
                f"{label_ids[lbl]} {(row[0]+row[2])/2/width:.6f} {(row[1]+row[3])/2/height:.6f} {(row[2]-row[0])/width:.6f} {(row[3]-row[1])/height:.6f}"
                for lbl, row in zip(frame._labels, xyxy.tolist())
            ]
            files.save(files.join(labels_dir, filename), lines, overwrite=True, verbose=False)
            manifest.append({"source_id": frame.source_id, "frame_index": frame.frame_index,
                             "source_path": frame.source_path, "image_size": list(frame.image_size),
                             "label_file": files.as_posix(files.join("labels", filename))})
        files.save(files.join(path, "data.yaml"), {"names": names}, overwrite=True, verbose=False)
        files.save(files.join(path, "manifest.json"), manifest, overwrite=True, verbose=False)

    def transform(self, *operations: Any) -> "Detections":
        """Apply one or more postprocessing operations or a callable pipeline in-place.

        Parameters
        ----------
        *operations : callable or iterable of callables
            One or more callables, an iterable of callables, or a postprocessing Pipeline.

        Returns
        -------
        Detections
            Self with in-place changes applied.
        """
        if not operations:
            raise ValueError("At least one operation or pipeline must be provided")
        if len(operations) == 1:
            op = operations[0]
            if callable(op):
                op(self)
                return self
            if isinstance(op, Iterable):
                from klygo.postprocessing import compose
                compose(*op)(self)
                return self
            raise TypeError("Operation must be callable")
        from klygo.postprocessing import compose
        compose(*operations)(self)
        return self


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
        if len({len(coords), len(scores), len(labels), len(class_ids)}) != 1:
            raise ValueError("boxes, scores, labels, and class_ids must have the same length")

        local = dict(registry)
        local_rev = {value: key for key, value in local.items()}
        n = len(coords)
        if n == 0:
            xyxy_arr = _EMPTY_XYXY
            score_arr = _EMPTY_SCORES
            cid_arr = _EMPTY_INTS
            uid_arr = _EMPTY_INTS
            label_list: list[str] = []
        else:
            xyxy_rows = []
            score_vals = []
            cid_vals = []
            label_list = []
            for coordinates, score, label, preferred in zip(coords, scores, labels, class_ids):
                lbl = local[label] if not isinstance(label, bool) and isinstance(label, int) and label in local else _label(label)
                found = local_rev.get(lbl)
                if found is None:
                    preferred_id = None if preferred is None else Box._integer(preferred, "class_id")
                    found = preferred_id if preferred_id is not None and preferred_id not in local else max(local, default=-1) + 1
                    local[found] = lbl
                    local_rev[lbl] = found
                xyxy_rows.append(_coords(coordinates))
                score_vals.append(_score(score))
                cid_vals.append(found)
                label_list.append(lbl)
            xyxy_arr = np.array(xyxy_rows, dtype=np.float64)
            score_arr = np.array(score_vals, dtype=np.float64)
            cid_arr = np.array(cid_vals, dtype=np.int64)
            uid_arr = np.arange(n, dtype=np.int64)

        frame_index = getattr(image, "frame_index", None)
        frame_index = position if frame_index is None else frame_index
        source_id, source_path = _source(image, None)
        frames.append(
            Detection._from_arrays(
                xyxy_arr,
                score_arr,
                cid_arr,
                uid_arr,
                label_list,
                source_id=source_id,
                frame_index=frame_index,
                source_path=source_path,
                image_size=getattr(image, "size", None),
                source=image,
                metadata=metadata,
                registry=local,
                label_ids=local_rev,
            )
        )
        registry.update(local)
    return frames


__all__ = ["build_detections", "Box", "Detection", "Detections"]
