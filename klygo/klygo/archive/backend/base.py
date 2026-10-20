"""Capability-aware contracts for archive format backends.

This module is intended for backend authors. Most applications should use the
high-level :mod:`klygo.archive` functions, which select and validate a backend
automatically.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, List, Literal, Optional, Union


class UnsupportedOperationError(NotImplementedError):
    """Report an operation that an archive format cannot perform.

    Parameters
    ----------
    format_name : str
        Canonical format name, for example ``"gz"`` or ``"rar"``.
    operation : str
        Requested operation, for example ``"add"`` or ``"compress"``.

    Attributes
    ----------
    format_name : str
        Backend format that rejected the operation.
    operation : str
        Rejected operation name.

    Notes
    -----
    The exception subclasses :class:`NotImplementedError`, so existing code
    that catches that standard exception remains compatible.
    """

    def __init__(self, format_name: str, operation: str) -> None:
        super().__init__(f"The {format_name} backend does not support '{operation}'.")
        self.format_name = format_name
        self.operation = operation


class UnsupportedOptionError(ValueError):
    """Report a non-default option unsupported by a backend.

    Parameters
    ----------
    format_name : str
        Canonical format name.
    operation : str
        Operation receiving the option.
    option : str
        Unsupported option name.

    Attributes
    ----------
    format_name : str
        Backend format that rejected the option.
    operation : str
        Operation for which validation failed.
    option : str
        Rejected option name.

    Notes
    -----
    A backend may accept an unsupported option when it still has the public
    default value. This lets the shared facade keep one stable signature while
    preventing callers from assuming a changed value had an effect.
    """

    def __init__(self, format_name: str, operation: str, option: str) -> None:
        super().__init__(
            f"The {format_name} backend does not support option '{option}' "
            f"for '{operation}'."
        )
        self.format_name = format_name
        self.operation = operation
        self.option = option


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Describe operations and variable options implemented by a backend.

    Parameters
    ----------
    compress, add, remove, merge, split : bool, default=False
        Whether the corresponding mutating operation is implemented.
        Reading, listing, searching, metadata inspection, and integrity tests
        are mandatory parts of :class:`ArchiveBackend` and therefore are not
        represented as flags.
    compress_options : frozenset[str], default=frozenset()
        Non-universal ``compress`` options whose changed values are honored.
    extract_options : frozenset[str], default=frozenset()
        Non-universal ``extract`` options whose changed values are honored.
    add_options : frozenset[str], default=frozenset()
        Non-universal ``add`` options whose changed values are honored.

    Notes
    -----
    Instances are immutable and use slots. Capability data is safe to expose
    for introspection, documentation generation, and dispatch decisions.

    Examples
    --------
    >>> from klygo.archive.backend import ZipBackend
    >>> ZipBackend.capabilities.compress
    True
    >>> "method" in ZipBackend.capabilities.compress_options
    True
    """

    compress: bool = False
    add: bool = False
    remove: bool = False
    merge: bool = False
    split: bool = False
    compress_options: FrozenSet[str] = field(default_factory=frozenset)
    extract_options: FrozenSet[str] = field(default_factory=frozenset)
    add_options: FrozenSet[str] = field(default_factory=frozenset)


class ArchiveBackend(ABC):
    """Abstract contract implemented by archive format adapters.

    Subclasses must implement extraction, member enumeration, searching,
    metadata inspection, and integrity testing. Creation and mutation are
    optional: subclasses advertise them through :attr:`capabilities` and
    override the corresponding default method.

    Attributes
    ----------
    format_name : str
        Canonical name used in messages and metadata.
    capabilities : BackendCapabilities
        Immutable declaration of supported write operations and options.

    Raises
    ------
    UnsupportedOperationError
        When a subclass inherits an optional mutation method but does not
        advertise and implement that operation.
    UnsupportedOptionError
        When a caller changes an option that the selected format cannot honor.

    Notes
    -----
    Direct backend use expects normalized :class:`pathlib.Path` arguments.
    User-facing path conversion, existence checks, format detection, and common
    option validation belong to :mod:`klygo.archive`.

    Examples
    --------
    Applications should normally use the public facade:

    >>> from klygo import archive
    >>> archive.compress("dataset", "dataset.zip", verbose=False)

    Backend authors can inspect the common capability contract directly:

    >>> from klygo.archive.backend import ArchiveBackend
    >>> ArchiveBackend.capabilities.compress
    False
    """

    format_name = "archive"
    capabilities = BackendCapabilities()

    def require_operation(
        self,
        operation: Literal["compress", "add", "remove", "merge", "split"],
    ) -> None:
        """Require a mutating operation before filesystem work begins.

        Parameters
        ----------
        operation : {'compress', 'add', 'remove', 'merge', 'split'}
            Capability flag to validate.

        Raises
        ------
        UnsupportedOperationError
            If the capability is disabled for this backend.
        """
        if not getattr(self.capabilities, operation):
            raise UnsupportedOperationError(self.format_name, operation)

    def validate_option(
        self,
        operation: Literal["compress", "extract", "add"],
        option: str,
        value: Any,
        default: Any,
    ) -> None:
        """Validate a non-universal option against backend capabilities.

        Parameters
        ----------
        operation : {'compress', 'extract', 'add'}
            Operation receiving the option.
        option : str
            Option name as exposed by the shared archive API.
        value : Any
            Value supplied by the caller.
        default : Any
            Public default value. Unsupported defaults are accepted because
            they do not request format-specific behavior.

        Raises
        ------
        UnsupportedOptionError
            If ``value`` differs from ``default`` and the backend does not
            advertise the option.
        """
        supported = getattr(self.capabilities, f"{operation}_options")
        if value != default and option not in supported:
            raise UnsupportedOptionError(self.format_name, operation, option)

    def compress(self, *args: Any, **kwargs: Any) -> None:
        """Create an archive, or reject creation for a read-only backend."""
        raise UnsupportedOperationError(self.format_name, "compress")

    @abstractmethod
    def extract(
        self,
        archive_path: Path,
        output_dir: Path,
        password: Optional[str] = None,
        include: Optional[Union[str, List[str]]] = None,
        exclude: Optional[Union[str, List[str]]] = None,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        """Extract selected members while enforcing safety and overwrite rules."""

    @abstractmethod
    def extract_file(
        self,
        archive_path: Path,
        filename: str,
        output_dir: Path,
        password: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """Extract one exact member, raising ``KeyError`` when it is absent."""

    @abstractmethod
    def list_files(self, archive_path: Path) -> List[str]:
        """Return member names in archive order."""

    @abstractmethod
    def iter_files(self, archive_path: Path) -> Iterator[str]:
        """Yield member names without a caller-owned archive handle."""

    @abstractmethod
    def search(
        self,
        archive_path: Path,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
    ) -> List[str]:
        """Return members matching a glob or regular expression."""

    @abstractmethod
    def get_info(self, archive_path: Path) -> Dict[str, Any]:
        """Return normalized archive metadata used by the public facade."""

    @abstractmethod
    def test(self, archive_path: Path, raise_exception: bool = False) -> bool:
        """Read the archive and report whether its structure is valid."""

    def add(self, *args: Any, **kwargs: Any) -> None:
        """Add members, or reject mutation for an unsupported backend."""
        raise UnsupportedOperationError(self.format_name, "add")

    def remove(self, *args: Any, **kwargs: Any) -> None:
        """Remove members, or reject mutation for an unsupported backend."""
        raise UnsupportedOperationError(self.format_name, "remove")

    def merge(self, *args: Any, **kwargs: Any) -> None:
        """Merge archives, or reject the operation when unsupported."""
        raise UnsupportedOperationError(self.format_name, "merge")

    def split_by_size(self, *args: Any, **kwargs: Any) -> List[str]:
        """Split an archive, or reject the operation when unsupported."""
        raise UnsupportedOperationError(self.format_name, "split_by_size")


__all__ = [
    "ArchiveBackend",
    "BackendCapabilities",
    "UnsupportedOperationError",
    "UnsupportedOptionError",
]
