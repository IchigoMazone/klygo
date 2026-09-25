"""Contract tests for ArchiveBackend."""

import unittest
from pathlib import Path
from typing import Any, Dict, Iterator, List

from klygo.archive.backend import (
    ArchiveBackend,
    UnsupportedOperationError,
)


class ReadOnlyBackend(ArchiveBackend):
    """Small complete test double using base mutation fallbacks."""

    format_name = "readonly"

    def extract(self, archive_path: Path, output_dir: Path, **kwargs: Any) -> None:
        return None

    def extract_file(
        self, archive_path: Path, filename: str, output_dir: Path, **kwargs: Any
    ) -> None:
        return None

    def list_files(self, archive_path: Path) -> List[str]:
        return ["member.txt"]

    def iter_files(self, archive_path: Path) -> Iterator[str]:
        yield from self.list_files(archive_path)

    def search(
        self,
        archive_path: Path,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
    ) -> List[str]:
        return [name for name in self.list_files(archive_path) if pattern in name]

    def get_info(self, archive_path: Path) -> Dict[str, Any]:
        return {"format": self.format_name}

    def test(self, archive_path: Path, raise_exception: bool = False) -> bool:
        return True


class TestArchiveBackend(unittest.TestCase):
    def test_abstract_contract_names_are_stable(self):
        self.assertEqual(
            ArchiveBackend.__abstractmethods__,
            {"extract", "extract_file", "list_files", "iter_files", "search", "get_info", "test"},
        )

    def test_default_mutation_methods_raise_consistent_errors(self):
        backend = ReadOnlyBackend()
        for method_name in ("compress", "add", "remove", "merge", "split_by_size"):
            with self.subTest(method=method_name):
                with self.assertRaises(UnsupportedOperationError):
                    getattr(backend, method_name)()


if __name__ == "__main__":
    unittest.main()

