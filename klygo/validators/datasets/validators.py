import math
import os

from klygo import files
from klygo.validators import validate_type

class Partition:

    def __init__(
        self,
        source: str | os.PathLike[str],
        target: str | os.PathLike[str],
        ratios: tuple[float, ...],
        overwrite: bool,
        verbose: bool,
    ) -> None:

        """Validate and normalize partition arguments."""
        validate_type(source, (str, os.PathLike), "source")
        validate_type(target, (str, os.PathLike), "target")
        validate_type(ratios, tuple, "ratios")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        if not all(isinstance(x, float) for x in ratios):
            raise TypeError(
                "ratios must be a tuple of floats."
            )
        if not all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in ratios):
            raise ValueError("ratios must contain finite values between 0 and 1")

        source = files.path(source)
        target = files.path(target)

        if not files.exists(source):
            raise FileNotFoundError(f"source does not exist: {source}")
        if files.exists(target) and not overwrite:
            raise FileExistsError(f"target already exists: {target}")
        if not (1 <= len(ratios) and len(ratios) <= 3):
            raise ValueError(f"ratios len must contain between 1 and 3 values: {len(ratios)}")
        if len(ratios) == 1:
            train = ratios[0]
            if not (0.5 <= train <= 0.95):
                raise ValueError(
                    "train ratio must be between 0.5 and 0.95."
                )
        elif len(ratios) == 2:
            train, val = ratios
            if not (0.525 <= train + val <= 0.975):
                raise ValueError(
                    "train + val ratio must be between 0.525 and 0.975."
                )
        else:
            train, val, test = ratios
            if abs(train + val + test - 1.0) > 1e-6:
                raise ValueError(
                    "The sum of ratios must equal 1."
                )
        self.source = source
        self.target = target
        self.ratios = ratios
        self.overwrite = overwrite
        self.verbose = verbose


class Repartition(Partition):
    pass


class Merge:
    def __init__(
        self,
        sources: list[str | os.PathLike[str]],
        output_path: str | os.PathLike[str],
        overwrite: bool,
        verbose: bool,
    ) -> None:
        """Validate and normalize merge arguments."""
        validate_type(sources, list, "sources")
        validate_type(output_path, (str, os.PathLike), "output_path")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        if not all(isinstance(x, (str, os.PathLike)) for x in sources):
            raise TypeError("sources must be a list of strings or path-like objects.")

        sources = [files.path(s) for s in sources]
        output_path = files.path(output_path)

        for src in sources:
            if not files.exists(src):
                raise FileNotFoundError(f"source does not exist: {src}")

        if not str(output_path).lower().endswith(".zip"):
            raise ValueError("output_path must end with '.zip'")

        if files.exists(output_path) and not overwrite:
            raise FileExistsError(f"output_path already exists: {output_path}")

        self.sources = sources
        self.output_path = output_path
        self.overwrite = overwrite
        self.verbose = verbose


class Split:
    def __init__(
        self,
        source: str | os.PathLike[str],
        output_dir: str | os.PathLike[str],
        by_class: bool = False,
        class_groups: list[list[str]] | None = None,
        ratios: tuple[float, ...] | None = None,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        """Validate and normalize split arguments."""
        validate_type(source, (str, os.PathLike), "source")
        validate_type(output_dir, (str, os.PathLike), "output_dir")
        validate_type(by_class, bool, "by_class")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        if class_groups is not None:
            validate_type(class_groups, list, "class_groups")
            if not class_groups or any(not group for group in class_groups):
                raise ValueError("class_groups must contain non-empty groups")
            for group in class_groups:
                validate_type(group, list, "each group in class_groups")
                if not all(isinstance(x, str) for x in group):
                    raise TypeError("class_groups must be a list of lists of strings.")

        if ratios is not None:
            validate_type(ratios, tuple, "ratios")
            if not all(isinstance(x, float) for x in ratios):
                raise TypeError("ratios must be a tuple of floats.")
            if not all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in ratios):
                raise ValueError("ratios must contain finite values between 0 and 1")
            if len(ratios) < 2:
                raise ValueError("ratios must contain at least 2 values.")
            if abs(sum(ratios) - 1.0) > 1e-4:
                raise ValueError("The sum of ratios must equal 1.0.")

        self.source = files.path(source)
        self.output_dir = files.path(output_dir)
        self.by_class = by_class
        self.class_groups = class_groups
        self.ratios = ratios
        self.overwrite = overwrite
        self.verbose = verbose

        if not files.exists(self.source):
            raise FileNotFoundError(f"source does not exist: {self.source}")

        if files.exists(self.output_dir) and not files.is_dir(self.output_dir):
            raise ValueError(
                f"output_dir must be a directory, got file: {self.output_dir}"
            )

        if not self.by_class and self.class_groups is None and self.ratios is None:
            raise ValueError("Must specify either by_class=True, class_groups, or ratios.")
        if sum((self.by_class, self.class_groups is not None, self.ratios is not None)) != 1:
            raise ValueError("exactly one split strategy must be specified")


class RemapClasses:
    def __init__(
        self,
        source: str | os.PathLike[str],
        target: str | os.PathLike[str],
        class_map: dict,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        """Validate and normalize remapclasses arguments."""
        validate_type(source, (str, os.PathLike), "source")
        validate_type(target, (str, os.PathLike), "target")
        validate_type(class_map, dict, "class_map")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        self.source = files.path(source)
        self.target = files.path(target)
        self.class_map = class_map
        self.overwrite = overwrite
        self.verbose = verbose

        if not files.exists(self.source):
            raise FileNotFoundError(f"source does not exist: {self.source}")

        if files.exists(self.target) and not self.overwrite:
            raise FileExistsError(
                f"target already exists: {self.target}"
            )







