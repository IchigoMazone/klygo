from typing import Any
from pathlib import Path

from klygo.validators import validate_type

class Partition:

    def __init__(
        self,
        source: str | Path,
        output: str | Path,
        ratios: tuple[float],
        overwrite: bool,
        verbose: bool,
    ) -> None:
        
        validate_type(source, (str, Path), "source")
        validate_type(output, (str, Path), "output")
        validate_type(ratios, tuple, "ratios")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        if not all(isinstance(x, float) for x in ratios):
            raise TypeError(
                "ratios must be a tuple of floats."
            )

        source = Path(source)
        output = Path(output) 

        if not source.exists():
            raise FileNotFoundError(f"source does not exist: {source}")
        if output.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {output}")
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
        self.output = output
        self.ratios = ratios 
        self.overwrite = overwrite
        self.verbose = verbose


class Repartition(Partition):
    pass


class Merge:
    def __init__(
        self,
        sources: list,
        output: str | Path,
        overwrite: bool,
        verbose: bool,
    ) -> None:
        validate_type(sources, list, "sources")
        validate_type(output, (str, Path), "output")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        if not all(isinstance(x, (str, Path)) for x in sources):
            raise TypeError("sources must be a list of strings or Path objects.")

        sources = [Path(s) for s in sources]
        output = Path(output)

        for src in sources:
            if not src.exists():
                raise FileNotFoundError(f"source does not exist: {src}")

        if not str(output).lower().endswith(".zip"):
            raise ValueError("output path must end with '.zip'")

        if output.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {output}")

        self.sources = sources
        self.output = output
        self.overwrite = overwrite
        self.verbose = verbose


class Split:
    def __init__(
        self,
        source: str | Path,
        output_dir: str | Path,
        by_class: bool = False,
        class_groups: list = None,
        ratios: tuple = None,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        validate_type(source, (str, Path), "source")
        validate_type(output_dir, (str, Path), "output_dir")
        validate_type(by_class, bool, "by_class")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        if class_groups is not None:
            validate_type(class_groups, list, "class_groups")
            for group in class_groups:
                validate_type(group, list, "each group in class_groups")
                if not all(isinstance(x, str) for x in group):
                    raise TypeError("class_groups must be a list of lists of strings.")

        if ratios is not None:
            validate_type(ratios, tuple, "ratios")
            if not all(isinstance(x, float) for x in ratios):
                raise TypeError("ratios must be a tuple of floats.")
            if len(ratios) < 2:
                raise ValueError("ratios must contain at least 2 values.")
            if abs(sum(ratios) - 1.0) > 1e-4:
                raise ValueError("The sum of ratios must equal 1.0.")

        self.source = Path(source)
        self.output_dir = Path(output_dir)
        self.by_class = by_class
        self.class_groups = class_groups
        self.ratios = ratios
        self.overwrite = overwrite
        self.verbose = verbose

        if not self.source.exists():
            raise FileNotFoundError(f"source does not exist: {self.source}")

        if not self.by_class and self.class_groups is None and self.ratios is None:
            raise ValueError("Must specify either by_class=True, class_groups, or ratios.")


class RemapClasses:
    def __init__(
        self,
        source: str | Path,
        output: str | Path,
        class_map: dict,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        validate_type(source, (str, Path), "source")
        validate_type(output, (str, Path), "output")
        validate_type(class_map, dict, "class_map")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        self.source = Path(source)
        self.output = Path(output)
        self.class_map = class_map
        self.overwrite = overwrite
        self.verbose = verbose

        if not self.source.exists():
            raise FileNotFoundError(f"source does not exist: {self.source}")

        if self.output.exists() and not self.overwrite:
            raise FileExistsError(f"output already exists: {self.output}")







