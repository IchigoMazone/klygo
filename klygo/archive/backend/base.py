from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Any, Dict, List, Optional, Union, Literal


class ArchiveBackend(ABC):
    """
    Abstract Base Class for all Archive Backends in klygo.archive.
    """

    @abstractmethod
    def compress(
        self,
        source: Path,
        output_path: Path,
        compresslevel: int = 6,
        method: Optional[str] = None,
        preserve_timestamp: bool = True,
        preserve_permissions: bool = True,
        follow_symlinks: bool = False,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        """Compress file or directory to archive."""
        pass

    @abstractmethod
    def extract(
        self,
        archive_path: Path,
        output_dir: Path,
        password: Optional[str] = None,
        include: Optional[Union[str, List[str]]] = None,
        exclude: Optional[Union[str, List[str]]] = None,
        preserve_timestamp: bool = True,
        preserve_permissions: bool = True,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        """Extract all or selected members from archive."""
        pass

    @abstractmethod
    def extract_file(
        self,
        archive_path: Path,
        filename: str,
        output_dir: Path,
        password: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """Extract a single specific file from archive."""
        pass

    @abstractmethod
    def list_files(self, archive_path: Path) -> List[str]:
        """List all filenames inside archive."""
        pass

    @abstractmethod
    def iter_files(self, archive_path: Path) -> Iterator[str]:
        """Iterate over filenames inside archive without loading all into memory."""
        pass

    @abstractmethod
    def search(
        self,
        archive_path: Path,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
    ) -> List[str]:
        """Search files inside archive matching pattern or regex."""
        pass

    @abstractmethod
    def get_info(self, archive_path: Path) -> Dict[str, Any]:
        """Get detailed metadata and statistics about the archive."""
        pass

    @abstractmethod
    def test(self, archive_path: Path, raise_exception: bool = False) -> bool:
        """Test archive integrity."""
        pass

    @abstractmethod
    def add(
        self,
        archive_path: Path,
        files: List[Path],
        on_conflict: Literal["rename", "overwrite", "skip"] = "rename",
        verbose: bool = True,
    ) -> None:
        """Add files/directories to an existing archive."""
        pass

    @abstractmethod
    def remove(self, archive_path: Path, files: List[str]) -> None:
        """Remove specified files from archive."""
        pass

    @abstractmethod
    def merge(
        self,
        archive_paths: List[Path],
        output_path: Path,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        """Merge multiple archives into one output archive."""
        pass

    @abstractmethod
    def split_by_size(
        self,
        archive_path: Path,
        size: float,  # size in MB
        output_dir: Path,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> List[str]:
        """Split archive into smaller part archives by max size (MB)."""
        pass
