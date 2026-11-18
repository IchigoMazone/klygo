"""Apply shared operations to independent same-source detection collections."""
from collections.abc import ItemsView, KeysView, Mapping, ValuesView
from klygo import validators
from .detect import Detections

__all__ = ["DetectionGroup"]


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
    Shared storage and label operations modify the original members and return
    this group for chaining. show() displays an identity-aligned pair and
    returns None without changing annotations or member order.
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

    def show(
        self,
        index: int = 0,
        *,
        members: tuple[int | str, int | str] | list[int | str] = (0, 1),
        titles: tuple[str, str] | list[str] | None = None,
        show_labels: bool = True,
        show_scores: bool = True,
        backend: str = "auto",
    ) -> None:
        """Display two annotation sets for the same image side by side.

        Parameters
        ----------
        index : int, default=0
            Image position in the first selected member. Negative positions
            follow normal Python indexing.
        members : tuple or list of two int or str, default=(0, 1)
            Distinct members selected by position or name. The second image
            is looked up by identity, so independent shuffling is supported.
        titles : tuple or list of two str, optional
            Panel captions. Defaults to the current selected member names.
        show_labels, show_scores : bool, default=True
            Include class names and confidence scores on bounding boxes.
        backend : str, default='auto'
            Display backend forwarded to visual.compare_images().

        Returns
        -------
        None
            Displays one comparison without changing annotations or order.

        Raises
        ------
        TypeError
            The image index or member selectors have unsupported types.
        IndexError
            An image or member position is out of range.
        KeyError
            A selected member name does not exist.
        ValueError
            The selection does not contain two distinct members, group
            sources no longer align, or display options are invalid.
        RuntimeError
            Source pixels are unavailable for a selected image.
        """
        from klygo import visual

        validators.validate_type(index, int, "index")
        if isinstance(index, bool):
            raise TypeError("index must be an integer, not bool")
        validators.validate_type(members, (tuple, list), "members")
        if len(members) != 2:
            raise ValueError("members must contain exactly two selectors")
        names = tuple(self._named)
        selected_names = []
        selected = []
        for selector in members:
            validators.validate_type(selector, (int, str), "member selector")
            if isinstance(selector, bool):
                raise TypeError("member selectors must be integers or strings, not bool")
            member = self[selector]
            selected.append(member)
            selected_names.append(selector if isinstance(selector, str) else names[selector])
        if selected[0] is selected[1]:
            raise ValueError("members must select two distinct collections")
        self._validate()
        first = selected[0][index]
        second = selected[1].get_frame(first.frame_index, source_id=first.source_id)
        visual.compare_images(
            first, second,
            titles=tuple(selected_names) if titles is None else titles,
            show_labels=show_labels,
            show_scores=show_scores,
            backend=backend,
        )

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
