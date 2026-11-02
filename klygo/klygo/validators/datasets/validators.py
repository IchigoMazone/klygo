from typing import Any
from pathlib import Path

from klygo.validators import validate_type

class Partition:

    def __init__(
        self,
        source: str | Path,
        target: str | Path,
        ratios: tuple[float],
        overwrite: bool,
        verbose: bool,
    ) -> None:

        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - source: File hoặc thư mục đầu vào
        - target: File hoặc thư mục đầu ra
        - ratios: Tham số ratios của hàm
        - overwrite: Trạng thái cho phép ghi đè
        - verbose: Trạng thái hiển thị tiến trình

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - FileExistsError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(source, (str, Path), "source")
        validate_type(target, (str, Path), "target")
        validate_type(ratios, tuple, "ratios")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        if not all(isinstance(x, float) for x in ratios):
            raise TypeError(
                "ratios must be a tuple of floats."
            )

        source = Path(source)
        target = Path(target)

        if not source.exists():
            raise FileNotFoundError(f"source does not exist: {source}")
        if target.exists() and not overwrite:
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
        sources: list,
        output_path: str | Path,
        overwrite: bool,
        verbose: bool,
    ) -> None:
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - sources: Danh sách file hoặc thư mục đầu vào
        - output_path: Đường dẫn file đầu ra
        - overwrite: Trạng thái cho phép ghi đè
        - verbose: Trạng thái hiển thị tiến trình

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - FileExistsError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(sources, list, "sources")
        validate_type(output_path, (str, Path), "output_path")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        if not all(isinstance(x, (str, Path)) for x in sources):
            raise TypeError("sources must be a list of strings or Path objects.")

        sources = [Path(s) for s in sources]
        output_path = Path(output_path)

        for src in sources:
            if not src.exists():
                raise FileNotFoundError(f"source does not exist: {src}")

        if not str(output_path).lower().endswith(".zip"):
            raise ValueError("output_path must end with '.zip'")

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"output_path already exists: {output_path}")

        self.sources = sources
        self.output_path = output_path
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
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - source: File hoặc thư mục đầu vào
        - output_dir: Đường dẫn thư mục đầu ra
        - by_class: Tham số by_class của hàm
        - class_groups: Tham số class_groups của hàm
        - ratios: Tham số ratios của hàm
        - overwrite: Trạng thái cho phép ghi đè
        - verbose: Trạng thái hiển thị tiến trình

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - TypeError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - ValueError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
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

        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError(
                f"output_dir must be a directory, got file: {self.output_dir}"
            )

        if not self.by_class and self.class_groups is None and self.ratios is None:
            raise ValueError("Must specify either by_class=True, class_groups, or ratios.")


class RemapClasses:
    def __init__(
        self,
        source: str | Path,
        target: str | Path,
        class_map: dict,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Tác dụng:
        - Khởi tạo đối tượng và kiểm tra các tham số ban đầu

        Đầu vào:
        - self: Đối tượng hiện tại
        - source: File hoặc thư mục đầu vào
        - target: File hoặc thư mục đầu ra
        - class_map: Tham số class_map của hàm
        - overwrite: Trạng thái cho phép ghi đè
        - verbose: Trạng thái hiển thị tiến trình

        Đầu ra:
        - Không trả về dữ liệu

        Ngoại lệ:
        - FileNotFoundError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ
        - FileExistsError: Phát sinh khi dữ liệu hoặc thao tác không hợp lệ

        Nguồn: TrinhNhuNhat_12072026.
        """
        validate_type(source, (str, Path), "source")
        validate_type(target, (str, Path), "target")
        validate_type(class_map, dict, "class_map")
        validate_type(overwrite, bool, "overwrite")
        validate_type(verbose, bool, "verbose")

        self.source = Path(source)
        self.target = Path(target)
        self.class_map = class_map
        self.overwrite = overwrite
        self.verbose = verbose

        if not self.source.exists():
            raise FileNotFoundError(f"source does not exist: {self.source}")

        if self.target.exists() and not self.overwrite:
            raise FileExistsError(
                f"target already exists: {self.target}"
            )







