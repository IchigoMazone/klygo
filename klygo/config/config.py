from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from box import Box

from klygo.config import operations
from klygo.validators.config import ConfigSource, ExportFile


class Config:
    """
    Quản lý file cấu hình thông minh: hỗ trợ đọc/ghi file YAML, JSON, TOML với thuộc tính dot-notation (Box),
    kế thừa khối `default:`, và tự động xuất file.
    """

    def __init__(self, config_path: str | Path) -> None:
        self._params = ConfigSource(config_path=config_path)
        self._box: Box = Box()
        self._cfg: dict = {}

    @property
    def config_path(self) -> Path:
        return self._params.config_path

    def read(self, verbose: bool = True) -> Box:
        self._box = operations.load(self._params.config_path, verbose=verbose)
        self._cfg = self._box.to_dict()
        return self._box

    def to_dict(self) -> dict[str, Any]:
        return dict(self._cfg)

    def to_json(self, indent: int = 4) -> str:
        import json
        return json.dumps(self._cfg, ensure_ascii=False, indent=indent)

    def get(self, key_path: str, default: Any = None) -> Any:
        return operations.get(self._cfg, key_path, default=default)

    def set(self, key_path: str, value: Any) -> Box:
        self._box = operations.set(self._cfg, key_path, value)
        self._cfg = self._box.to_dict()
        return self._box

    def has(self, key_path: str) -> bool:
        return operations.has(self._cfg, key_path)

    def delete(self, key_path: str) -> bool:
        res = operations.delete(self._cfg, key_path)
        if res:
            self._box = Box(self._cfg)
        return res

    def merge(self, *configs: dict, deep: bool = True) -> Box:
        self._box = operations.merge(self._cfg, *configs, deep=deep)
        self._cfg = self._box.to_dict()
        return self._box

    def update(self, updates: dict, deep: bool = True) -> Box:
        self._box = operations.update(self._cfg, updates, deep=deep)
        self._cfg = self._box.to_dict()
        return self._box

    @classmethod
    def create_default(
        cls,
        path: str | Path,
        default_data: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> "Config":
        operations.create(path, default_data=default_data, overwrite=overwrite, verbose=verbose)
        return cls(path)

    def export_file(
        self,
        name: str,
        suffix: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
        verbose: bool = True,
        ext: Optional[str] = None,
    ) -> Path:
        actual_suffix = ext or suffix or ".json"
        if not actual_suffix.startswith("."):
            actual_suffix = f".{actual_suffix}"

        clean_name = name
        if clean_name.lower().endswith(actual_suffix.lower()):
            clean_name = clean_name[:-len(actual_suffix)]

        params = ExportFile(
            name=clean_name,
            suffix=actual_suffix,
            output_dir=output_dir if output_dir is not None else ".",
            overwrite=overwrite,
            verbose=verbose,
        )

        if output_dir is not None:
            out_dir = Path(output_dir)
        elif "default" in self._cfg and isinstance(self._cfg["default"], dict) and "root" in self._cfg["default"]:
            out_dir = Path(self._cfg["default"]["root"])
        else:
            out_dir = Path(".")

        file_path = out_dir / f"{params.name}{params.suffix}"
        data = self._cfg if self._cfg else operations.load(self._params.config_path, verbose=False)
        return operations.export(data, file_path, overwrite=params.overwrite, verbose=params.verbose)
