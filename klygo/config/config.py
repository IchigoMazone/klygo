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

from klygo.validators.config import ConfigSource, ExportFile

from .access import delete as delete_value
from .access import get as get_value
from .access import has as has_value
from .access import set as set_value
from .creation import create
from .io import export, load
from .mapping import merge as merge_values
from .mapping import update as update_values


class Config:
    """Manage one configuration file through a stateful object interface.

    Construction stores a path but performs no I/O until :meth:`read` or
    :meth:`export_file` needs data. Function-oriented users can use the
    matching APIs directly from :mod:`klygo.config`.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the configuration file managed by this instance.

    Examples
    --------
    >>> from klygo import Config
    >>> settings = Config("settings.yaml")
    >>> data = settings.read(verbose=False)
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

    def read(self, verbose: bool = True) -> Box:
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
        export_file

        Examples
        --------
        >>> manager = Config("settings.yaml")
        >>> settings = manager.read(verbose=False)
        """
        self._box = load(self.config_path, verbose=verbose)
        self._cfg = self._box.to_dict()
        return self._box

    def to_dict(self) -> dict[str, Any]:
        """Return an independent dictionary representation of current state.

        Returns
        -------
        dict[str, Any]
            Deep plain-dictionary conversion of the internal ``Box``. Mutating
            the returned dictionary does not mutate this manager.

        Notes
        -----
        Before :meth:`read`, this returns an empty dictionary.

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

    def set(self, key_path: str, value: Any) -> Box:
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
        box.Box
            Complete updated in-memory configuration.

        Raises
        ------
        ValueError
            If ``key_path`` is empty.

        Examples
        --------
        >>> manager = Config("settings.yaml")
        >>> manager.set("model.batch", 32).model.batch
        32
        """
        self._box = set_value(self._cfg, key_path, value)
        self._cfg = self._box.to_dict()
        return self._box

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

    def merge(self, *configs: Mapping[str, Any] | Box, deep: bool = True) -> Box:
        """Merge configurations into current state from left to right.

        Parameters
        ----------
        configs : mapping or box.Box
            Configurations in increasing precedence order.
        deep : bool, default=True
            Recursively combine nested mappings when true.

        Returns
        -------
        box.Box
            Complete merged in-memory configuration.

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
        >>> manager.merge({"model": {"batch": 16}}, {"seed": 7}).seed
        7
        """
        self._box = merge_values(self._cfg, *configs, deep=deep)
        self._cfg = self._box.to_dict()
        return self._box

    def update(self, updates: Mapping[str, Any] | Box, deep: bool = True) -> Box:
        """Apply one mapping to current state without mutating that mapping.

        Parameters
        ----------
        updates : mapping or box.Box
            Values to apply.
        deep : bool, default=True
            Recursively combine nested mappings; false replaces top-level
            values in full.

        Returns
        -------
        box.Box
            Complete updated in-memory configuration.

        Raises
        ------
        TypeError
            If ``updates`` is not mapping-like or ``deep`` is not boolean.

        Examples
        --------
        >>> manager = Config("settings.yaml")
        >>> manager.update({"model": {"epochs": 10}}).model.epochs
        10
        """
        self._box = update_values(self._cfg, updates, deep=deep)
        self._cfg = self._box.to_dict()
        return self._box

    @classmethod
    def create_default(
        cls,
        path: str | Path,
        default_data: Mapping[str, Any] | Box | None = None,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> "Config":
        """Create a default configuration file and return its manager.

        The returned manager is path-bound but intentionally unread. Call
        :meth:`read` when in-memory access is required.

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

        Returns
        -------
        Config
            New manager bound to ``path``.

        Raises
        ------
        FileExistsError
            If the destination exists and overwrite is disabled.

        Examples
        --------
        >>> manager = Config.create_default("settings.yaml", overwrite=True)
        >>> manager.config_path.name
        'settings.yaml'
        """
        create(path, default_data=default_data, overwrite=overwrite, verbose=verbose)
        return cls(path)

    def export_file(
        self,
        name: str,
        suffix: str | None = None,
        output_dir: str | Path | None = None,
        overwrite: bool = False,
        verbose: bool = True,
        ext: str | None = None,
    ) -> Path:
        """Export current state under a new filename and format.

        ``ext`` is retained as a compatibility alias for ``suffix``. When no
        output directory is supplied, ``default.root`` is used when available,
        otherwise the current directory is selected.

        Parameters
        ----------
        name : str
            Output filename with or without its destination extension.
        suffix : str or None
            Output extension; defaults to ``.json``.
        output_dir : str, pathlib.Path, or None
            Explicit destination directory. If omitted, current state may
            supply ``default.root``.
        overwrite : bool, default=False
            Replace an existing output file.
        verbose : bool, default=True
            Enable the output progress indicator.
        ext : str or None
            Backward-compatible alias taking precedence over ``suffix``.

        Returns
        -------
        pathlib.Path
            Exported configuration path.

        Raises
        ------
        ValueError
            If the selected suffix is empty or unsupported.
        FileExistsError
            If the destination exists and overwrite is disabled.
        FileNotFoundError
            If state is empty and the bound source file cannot be loaded.

        See Also
        --------
        read
        klygo.config.export

        Examples
        --------
        >>> manager = Config("settings.yaml")
        >>> manager.read(verbose=False)
        >>> manager.export_file("resolved", ".toml", overwrite=True)
        """
        actual_suffix = ext or suffix or ".json"
        if not isinstance(actual_suffix, str) or not actual_suffix:
            raise ValueError("suffix must be a non-empty string")
        if not actual_suffix.startswith("."):
            actual_suffix = f".{actual_suffix}"
        clean_name = name[:-len(actual_suffix)] if name.lower().endswith(actual_suffix.lower()) else name
        params = ExportFile(
            name=clean_name,
            suffix=actual_suffix,
            output_dir=output_dir if output_dir is not None else ".",
            overwrite=overwrite,
            verbose=verbose,
        )
        if output_dir is not None:
            destination_dir = Path(output_dir)
        elif isinstance(self._cfg.get("default"), dict) and "root" in self._cfg["default"]:
            destination_dir = Path(self._cfg["default"]["root"])
        else:
            destination_dir = Path(".")
        destination = destination_dir / f"{params.name}{params.suffix}"
        data = self._cfg or load(self.config_path, verbose=False)
        return export(data, destination, overwrite=params.overwrite, verbose=params.verbose)
