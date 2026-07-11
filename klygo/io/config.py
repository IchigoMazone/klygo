from pathlib import Path
from functools import reduce
from operator import getitem
from typing import Any

from box import Box

from klygo.validators.io import ConfigSource, ExportFile
from klygo.io.read import read_file
from klygo.io.write import write_file


class Config:
    def __init__(self, src: str | Path) -> None:
        self._params = ConfigSource(src=src)
        self._keys: list = []
        self._cfg: dict = {}

    @property
    def src(self) -> Path:
        """Path to the source config file."""
        return self._params.src

    def _assign(self, keys: list, cfg: dict, value: Any | None = None) -> None:
        """Set a nested dict value by key path."""
        cur = reduce(getitem, keys[:-1], cfg)
        cur[keys[-1]] = value

    def _get_value(self, keys: list, cfg: dict) -> Any:
        """Get a nested dict value by key path."""
        return reduce(getitem, keys, cfg)

    def _traverse(self, tree: dict) -> None:
        """Walk the config tree and expand relative paths under ``default.root``."""
        for key, value in tree.items():
            if key == "default":
                continue

            self._keys.append(key)

            if len(self._keys) > 1:
                current = self._get_value(self._keys, self._cfg)
                if isinstance(current, str) and current.startswith("."):
                    root = self._cfg["default"]["root"]
                    value = root + current[1:]
                    self._assign(self._keys, self._cfg, value)

            if isinstance(value, dict):
                self._traverse(value)

            self._keys.pop()

    def read(self, verbose: bool = True) -> Box:
        """Load the config file and return a dot-accessible :class:`Box`.

        If the config contains a ``default.root`` key, any string value
        starting with ``.`` is expanded to an absolute path by prepending
        ``default.root``.

        Parameters
        ----------
        verbose : bool, optional
            If *True* (default), display a progress bar.

        Returns
        -------
        Box
            Config contents accessible by attribute or dict-key notation.

        Examples
        --------
        >>> cfg = Config("config.yaml").read()
        >>> print(cfg.model.lr)
        0.001
        """
        self._cfg = dict(read_file(self._params.src, verbose=verbose))

        if "default" in self._cfg and isinstance(self._cfg["default"], dict) and "root" in self._cfg["default"]:
            self._traverse(self._cfg)

        return Box(self._cfg)

    def imread(self, verbose: bool = True) -> Box:
        """Deprecated alias for Config.read(). Use Config.read() instead."""
        import warnings
        warnings.warn(
            "`imread` is deprecated and will be removed in a future version. "
            "Use `read()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.read(verbose=verbose)

    def to_dict(self) -> dict[str, Any]:
        """Convert the parsed config into a standard Python dict.

        Returns
        -------
        dict
            Standard Python dictionary representing config contents.
        """
        return dict(self._cfg)

    def to_json(self, indent: int = 4) -> str:
        """Convert the parsed config into a JSON string.

        Parameters
        ----------
        indent : int, optional
            Number of spaces for indentation. Default is 4.

        Returns
        -------
        str
            JSON string representing config contents.
        """
        import json
        return json.dumps(self._cfg, ensure_ascii=False, indent=indent)

    @staticmethod
    def create_default(
        path: str | Path,
        default_data: dict[str, Any] | None = None,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> "Config":
        """Initialize a default template configuration file.

        Parameters
        ----------
        path : str or Path
            Path to the output configuration file.
        default_data : dict, optional
            Default config dictionary data to write. If not provided,
            a standard default configuration template is written.
        overwrite : bool, optional
            If *True*, overwrite the config file if it already exists.
            Default is *False*.
        verbose : bool, optional
            If *True* (default), display progress bar.

        Returns
        -------
        Config
            Config instance configured with the created file.
        """
        if default_data is None:
            default_data = {
                "default": {
                    "root": "./data",
                },
                "model": {
                    "name": "yolov8n",
                    "epochs": 100,
                    "batch": 16,
                    "lr": 0.01,
                }
            }
        path = Path(path)
        write_file(path, default_data, overwrite=overwrite, verbose=verbose)
        return Config(path)

    def export_file(
        self,
        name: str,
        suffix: str,
        output_dir: str | Path | None = None,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        """Export the config to a different file format.

        Parameters
        ----------
        name : str
            Output filename without extension.
        suffix : str
            Target format extension (e.g. ``".json"``, ``"toml"``).
            A leading dot is added automatically if omitted.
        output_dir : str or Path, optional
            Directory to write the output file into.
            Defaults to ``default.root`` in the config (if present),
            otherwise the current working directory.
        overwrite : bool, optional
            If *True*, overwrite the output file if it exists.
            Default is *False*.
        verbose : bool, optional
            If *True* (default), display a progress bar.

        Raises
        ------
        ValueError
            If ``suffix`` is the same as the source file's extension,
            or if ``suffix`` is not a supported format.
        FileExistsError
            If output file exists and ``overwrite`` is *False*.

        Examples
        --------
        >>> cfg = Config("config.yaml")
        >>> cfg.read()
        >>> cfg.export_file("config", ".json", output_dir="out/")
        """
        params = ExportFile(
            name=name,
            suffix=suffix,
            output_dir=output_dir if output_dir is not None else ".",
            overwrite=overwrite,
            verbose=verbose,
        )

        if params.suffix == self._params.src.suffix:
            raise ValueError(
                f"Export suffix {params.suffix!r} must be different "
                f"from source suffix {self._params.src.suffix!r}."
            )

        # Resolve output directory
        if output_dir is not None:
            out_dir = Path(output_dir)
        elif "default" in self._cfg and isinstance(self._cfg["default"], dict) and "root" in self._cfg["default"]:
            out_dir = Path(self._cfg["default"]["root"])
        else:
            out_dir = Path(".")

        file_path = out_dir / f"{params.name}{params.suffix}"
        data = read_file(self._params.src, verbose=False)
        write_file(
            file_path,
            data,
            overwrite=params.overwrite,
            verbose=params.verbose,
        )
