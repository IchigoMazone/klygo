"""Apply shared operations to independent same-source detection collections."""
from collections.abc import ItemsView, KeysView, Mapping, ValuesView
from klygo import validators
from .detect import Detections

__all__ = ["DetectionGroup", "group"]


class DetectionGroup:
    """Hold references to same-source materialized Detections collections.

    Parameters
    ----------
    *collections : Detections or mapping of str to Detections
        Distinct same-source collections, or one mapping assigning member names.
        Two positional members are named pred and truth; other counts use set_1,
        set_2, and so on. Names follow input order without inferring label roles.
        Frame order and annotation sets may differ.

    Notes
    -----
    Operations modify the original members and return this group for chaining.
    Source alignment is rechecked before each operation. Members are processed
    sequentially; a backend failure can leave earlier members already converted.
    Streaming members and duplicate object references are rejected.
    """
    def __init__(self, *collections: Detections | Mapping[str, Detections]):
        if not collections:
            raise ValueError("group requires at least one Detections collection")
        if len(collections) == 1 and isinstance(collections[0], Mapping):
            named = dict(collections[0])
            if not named:
                raise ValueError("group requires at least one Detections collection")
            for name in named:
                validators.validate_type(name, str, "member name")
                if not name.strip():
                    raise ValueError("member names must be non-empty strings")
            collections = tuple(named.values())
        else:
            names = ("pred", "truth") if len(collections) == 2 else tuple(
                f"set_{index + 1}" for index in range(len(collections)))
            named = dict(zip(names, collections))
        for collection in collections:
            validators.validate_type(collection, Detections, "collection")
        if len({id(item) for item in collections}) != len(collections):
            raise ValueError("group members must be distinct collections")
        self._members = tuple(collections)
        self._named = named
        self._validate()

    @property
    def members(self) -> tuple[Detections, ...]:
        """Return the original member collections in insertion order."""
        return self._members

    def __len__(self):
        return len(self._members)

    def __iter__(self):
        return iter(self._members)

    def __getitem__(self, index):
        return self._named[index] if isinstance(index, str) else self._members[index]

    def keys(self) -> KeysView[str]:
        """Return member names in insertion order."""
        return self._named.keys()

    def values(self) -> ValuesView[Detections]:
        """Return the original collections in insertion order."""
        return self._named.values()

    def items(self) -> ItemsView[str, Detections]:
        """Return name/collection pairs in insertion order."""
        return self._named.items()

    def rename(self, mapping: Mapping[str, str]) -> "DetectionGroup":
        """Rename members together while preserving their order and references.

        Parameters
        ----------
        mapping : mapping of str to str
            Existing names mapped to non-empty new names. Omitted names remain
            unchanged. Renames are simultaneous, so swapping names is supported.
            An empty mapping leaves names unchanged.

        Returns
        -------
        DetectionGroup
            This group, for chaining. Member annotations and devices are unchanged.

        Raises
        ------
        TypeError
            Mapping is not a mapping or a name is not a string.
        KeyError
            A source name does not exist.
        ValueError
            A new name is empty, final names collide, or sources no longer align.
            Validation finishes before any names are changed.
        """
        validators.validate_type(mapping, Mapping, "mapping")
        self._validate()
        mapping = dict(mapping)
        for old, new in mapping.items():
            validators.validate_type(old, str, "old name")
            validators.validate_type(new, str, "new name")
            if old not in self._named:
                raise KeyError(old)
            if not new.strip():
                raise ValueError("member names must be non-empty strings")
        names = [mapping.get(name, name) for name in self._named]
        if len(set(names)) != len(names):
            raise ValueError("renamed member names must be unique")
        self._named = dict(zip(names, self._members))
        return self

    def _validate(self):
        reference = None
        for collection in self._members:
            if collection.is_stream:
                raise ValueError("group requires materialized Detections collections")
            frames = list(collection)
            sources = {frame.identity: frame.image_size for frame in frames}
            if len(sources) != len(frames):
                raise ValueError("frame identities must be unique")
            if reference is not None and sources != reference:
                raise ValueError("group members must share frame identities and image sizes")
            reference = sources

    def _apply(self, method, *args, **kwargs):
        self._validate()
        for member in self._members:
            getattr(member, method)(*args, **kwargs)
        return self

    def cuda(self, device: int = 0) -> "DetectionGroup":
        """Move all original collections to one CUDA device."""
        return self._apply("cuda", device)

    def cpu(self) -> "DetectionGroup":
        """Move all original collections to CPU storage."""
        return self._apply("cpu")

    def astype(self, dtype: int) -> "DetectionGroup":
        """Convert all members to 16, 32, or 64-bit annotation storage."""
        return self._apply("astype", dtype)

    def drop(self, labels) -> "DetectionGroup":
        """Remove the same labels from every member."""
        labels = labels if isinstance(labels, str) else tuple(labels)
        return self._apply("drop", labels)

    def keep(self, labels) -> "DetectionGroup":
        """Keep the same labels in every member."""
        labels = labels if isinstance(labels, str) else tuple(labels)
        return self._apply("keep", labels)

    def relabel(self, mapping) -> "DetectionGroup":
        """Apply the same label mapping to every member."""
        return self._apply("relabel", mapping)

    def compact(self) -> "DetectionGroup":
        """Compact each member's own active class registry."""
        return self._apply("compact")

    def pack(self) -> "DetectionGroup":
        """Pack annotation storage for every member."""
        return self._apply("pack")

    def unpack(self) -> "DetectionGroup":
        """Unpack annotation storage for every member."""
        return self._apply("unpack")


def group(*collections: Detections | Mapping[str, Detections]) -> DetectionGroup:
    """Group independent same-source collections without merging their contents.

    Parameters
    ----------
    *collections : Detections or mapping of str to Detections
        Distinct materialized collections with matching frame identities and
        image sizes, or one mapping assigning names. Two positional members use
        pred/truth; other counts use set_1, set_2, and so on, in input order.
        Explicit mappings retain their names. Annotation sets may differ.

    Returns
    -------
    DetectionGroup
        Reference group supporting cuda, cpu, astype, label operations, and
        pack/unpack. Individual collection APIs remain available.

    Raises
    ------
    TypeError
        A member is not Detections or a mapping key is not a string.
    ValueError
        The group is empty, repeats an object, contains a stream, or has
        inconsistent frame identities or image sizes, or empty names.
    """
    return DetectionGroup(*collections)
