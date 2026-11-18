"""Stateful object interface for configuration files.

The functional APIs in :mod:`klygo.config` are preferable for isolated,
side-effect-free transformations. :class:`Config` complements them with a
path-bound, mutable in-memory workflow for applications that repeatedly read,
modify, inspect, and export the same configuration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from box import Box

from klygo import files
from klygo.validators.config import ConfigSource, ExportFile

from .access import delete as delete_value
from .access import get as get_value
from .access import has as has_value
from .access import set as set_value
from .creation import create
from .io import export, load, save
from .mapping import merge as merge_values
from .mapping import update as update_values


class Config:
    """Manage one configuration file through a stateful object interface.

    Construction stores a path but performs no I/O until :meth:`load` or
    :meth:`export_file` needs data. Function-oriented users can use the
    matching APIs directly from :mod:`klygo.config`.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the configuration file managed by this instance.

    Returns
    -------
    Config
        Instance bound to the configuration path.

    Raises
    ------
    TypeError
        If ``config_path`` is not a string or path-like object.

    Examples
    --------
    >>> from klygo import Config
    >>> settings = Config("settings.yaml")
    >>> data = settings.load(verbose=False)
    >>> data.model.name
    'yolo'
    """

    def __init__(self, config_path: str | Path) -> None:
        self._params = ConfigSource(config_path=config_path)
        self._box = Box()
        self._cfg: dict[str, Any] = {}

    @property
    def config_path(self) -> Path:
        """Return the source path managed by this instance.

        Returns
        -------
        pathlib.Path
            Validated path supplied at construction time.

        Examples
        --------
        >>> from klygo import Config
        >>> Config("settings.yaml").config_path.name
        'settings.yaml'
        """
        return self._params.config_path

    def load(self, verbose: bool = True) -> Box:
        """Load the source file and replace the complete in-memory state.

        Parameters
        ----------
        verbose : bool, default=True
            Enable the progress indicator used by the underlying file loader.

        Returns
        -------
        box.Box
            Newly loaded configuration with mapping and attribute access.

        Raises
        ------
        FileNotFoundError
            If :attr:`config_path` does not exist.
        TypeError
            If the decoded document root is not a mapping.
        ValueError
            If the file extension is unsupported.

        See Also
        --------
        klygo.config.load

        Examples
        --------
        >>> manager = Config("settings.yaml")
        >>> settings = manager.load(verbose=False)
        """
        self._box = load(self.config_path, verbose=verbose)
        self._cfg = self._box.to_dict()
        return self._box

    def to_dict(self) -> dict[str, Any]:
        """Return an independent dictionary representation of current state.

        Parameters
        ----------
        None
            Uses the configuration held by this instance.

        Returns
        -------
        dict[str, Any]
            Deep plain-dictionary conversion of the internal ``Box``. Mutating
            the returned dictionary does not mutate this manager.

        Raises
        ------
        None
            Conversion of the in-memory configuration does not raise.

        Notes
        -----
        Before :meth:`load`, this returns an empty dictionary.

        Examples
        --------
        >>> manager = Config("settings.yaml")
        >>> manager.to_dict()
        {}
        """
        return self._box.to_dict()

    def to_json(self, indent: int = 4) -> str:
        """Serialize current in-memory state to a Unicode JSON string.

        Parameters
        ----------
        indent : int, default=4
            Non-negative number of spaces used for pretty printing.

        Returns
        -------
        str
            JSON text produced with ``ensure_ascii=False``.

        Raises
        ------
        ValueError
            If ``indent`` is negative, boolean, or not an integer.
        TypeError
            If current values are not JSON serializable.

        Examples
        --------
        >>> manager = Config("settings.yaml")
        >>> manager.to_json(indent=2)
        '{}'
        """
        if not isinstance(indent, int) or isinstance(indent, bool) or indent < 0:
            raise ValueError("indent must be a non-negative integer")
        return json.dumps(self._cfg, ensure_ascii=False, indent=indent)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Return a nested in-memory value by dot-separated path.

        Parameters
        ----------
        key_path : str
            Non-empty path such as ``"model.optimizer.lr"``.
        default : Any, default=None
            Value returned when the path is absent.

        Returns
        -------
        Any
            Stored value or ``default``.

        Raises
        ------
        ValueError
            If ``key_path`` is empty.

        See Also
        --------
        has
        set

        Examples
        --------
        >>> manager = Config("settings.yaml")
        >>> manager.get("model.batch", 16)
        16
        """
        return get_value(self._cfg, key_path, default=default)

    def _set_state(self, box: Box) -> None:
        """Store updated Box and dict state."""
        self._box = box
        self._cfg = box.to_dict()

    def set(self, key_path: str, value: Any) -> None:
        """Assign a nested value and replace the current in-memory state.

        Missing intermediate dictionaries are created. This operation updates
        memory only; call :meth:`export_file` to persist the result.

        Parameters
        ----------
        key_path : str
            Non-empty destination path.
        value : Any
            Value to copy into the configuration.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``key_path`` is empty.

        Examples
        --------
        >>> manager = Config("settings.yaml")
        >>> manager.set("model.batch", 32)
        """
        self._set_state(set_value(self._cfg, key_path, value))

    def has(self, key_path: str) -> bool:
        """Return whether a nested path exists in current state.

        Parameters
        ----------
        key_path : str
            Non-empty path to inspect.

        Returns
        -------
        bool
            ``True`` even when the stored value is explicitly ``None``.

        Raises
        ------
        ValueError
            If ``key_path`` is empty.

        Examples
        --------
        >>> manager = Config("settings.yaml")
        >>> manager.has("model.batch")
        False
        """
        return has_value(self._cfg, key_path)

    def delete(self, key_path: str) -> bool:
        """Delete a nested value from current in-memory state.

        Parameters
        ----------
        key_path : str
            Non-empty path to remove.

        Returns
        -------
        bool
            ``True`` when a key was removed, otherwise ``False``.

        Raises
        ------
        ValueError
            If ``key_path`` is empty.

        Notes
        -----
        Empty parent mappings are retained and no file is written.

        Examples
        --------
        >>> manager = Config("settings.yaml")
        >>> manager.delete("model.batch")
        False
        """
        removed = delete_value(self._cfg, key_path)
        if removed:
            self._box = Box(self._cfg)
        return removed

    def merge(self, *configs: Mapping[str, Any] | Box, deep: bool = True) -> None:
        """Merge configurations into current state from left to right.

        Parameters
        ----------
        configs : mapping or box.Box
            Configurations in increasing precedence order.
        deep : bool, default=True
            Recursively combine nested mappings when true.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If any configuration is not mapping-like or ``deep`` is not bool.

        See Also
        --------
        update
        klygo.config.merge

        Examples
        --------
        >>> manager = Config("settings.yaml")
        >>> manager.merge({"model": {"batch": 16}}, {"seed": 7})
        """
        self._set_state(merge_values(self._cfg, *configs, deep=deep))

    @classmethod
    def create_default(
        cls,
        path: str | Path,
        default_data: Mapping[str, Any] | Box | None = None,
        overwrite: bool = False,
        verbose: bool = True,
        metadata: Mapping[str, Any] | Box | None = None,
    ) -> "Config":
        """Create a configuration file and return a loaded manager.

        Normal configuration creation deep-merges ``default_data`` into
        Klygo's built-in defaults. Supplying ``metadata`` instead writes that
        mapping exactly, which is suitable for the flat structure produced by
        :func:`klygo.models.metadata` and :func:`klygo.models.configure`.
        The returned manager is already populated in both modes; calling
        :meth:`load` again is optional.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination configuration file.
        default_data : mapping, box.Box, or None
            Optional deep overrides for built-in defaults.
        overwrite : bool, default=False
            Replace an existing destination.
        verbose : bool, default=True
            Enable file progress indicators.
        metadata : mapping, box.Box, or None
            Exact model metadata to write without injecting general Klygo
            configuration defaults. Mutually exclusive with ``default_data``.

        Returns
        -------
        Config
            New manager bound to ``path``.

        Raises
        ------
        FileExistsError
            If the destination exists and overwrite is disabled.
        ValueError
            If both ``default_data`` and ``metadata`` are supplied.

        Examples
        --------
        >>> manager = Config.create_default("settings.yaml", overwrite=True)
        >>> manager.config_path.name
        'settings.yaml'

        Persist model metadata without injecting general Klygo defaults:

        >>> from klygo import models
        >>> flags = models.flags(model=False, processor=True, post=True)
        >>> schema = models.metadata(flags)
        >>> manager = Config.create_default(
        ...     "model.toml", metadata=schema, overwrite=True, verbose=False
        ... )
        >>> manager.get("flags.processor")
        True
        """
        if default_data is not None and metadata is not None:
            raise ValueError("default_data and metadata are mutually exclusive")

        if metadata is not None:
            save(path, metadata, overwrite=overwrite, verbose=verbose)
            created = load(path, verbose=False)
        else:
            created = create(
                path,
                default_data=default_data,
                overwrite=overwrite,
                verbose=verbose,
            )

        manager = cls(path)
        manager._box = Box(created.to_dict())
        manager._cfg = manager._box.to_dict()
        return manager


__all__ = ["Config"]
