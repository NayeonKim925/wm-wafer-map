"""Automated acquisition and verification of the WM-811K dataset.

:class:`DatasetManager` is the single entry point for making the dataset
available on disk.  Its responsibilities:

* **locate** an existing copy in the project-local ``datasets/`` directory,
* **download** it from Kaggle via :mod:`kagglehub` when it is missing,
* **verify** that the expected files are present and non-truncated,
* **return** a stable, project-local path to the dataset directory.

The raw dataset is never committed to the repository; it is fetched on demand.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from src.config import DatasetConfig
from src.utils.logging import get_logger
from src.utils.paths import resolve_path

_KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map"


class DatasetError(RuntimeError):
    """Base class for dataset-acquisition failures."""


class DatasetDownloadError(DatasetError):
    """Raised when the dataset cannot be downloaded."""


class DatasetVerificationError(DatasetError):
    """Raised when the on-disk dataset fails integrity checks."""


class DatasetManager:
    """Locate, download and verify the WM-811K dataset.

    Parameters
    ----------
    config:
        The :class:`~src.config.DatasetConfig` describing where the data should
        live and how to fetch it.
    logger:
        Optional logger; a module logger is used when omitted.
    """

    def __init__(self, config: DatasetConfig, logger: Any | None = None) -> None:
        self.config = config
        self.logger = logger or get_logger(__name__)
        self.local_root: Path = resolve_path(config.root_dir)
        self.dataset_dir: Path = self.local_root / config.dataset_name

    # -- Public API ---------------------------------------------------------
    @property
    def primary_file(self) -> Path:
        """Path to the main artefact we train on (e.g. ``LSWMD.pkl``)."""
        return self.dataset_dir / self.config.pickle_filename

    def prepare(self) -> Path:
        """Ensure the dataset is present and verified; return its directory.

        This is the method training/eval/inference call.  It skips the download
        entirely when a valid copy already exists (unless ``force_download``).
        """
        if self.config.force_download:
            self.logger.info("force_download=True - fetching a fresh copy.")
            return self.download()

        if self.is_available():
            self.logger.info(
                "Dataset already present and verified at %s - skipping download.",
                self.dataset_dir,
            )
            return self.dataset_dir

        self.logger.info("Dataset not found locally at %s.", self.dataset_dir)
        return self.download()

    def locate(self) -> Path | None:
        """Return the dataset directory if a verified copy exists, else ``None``."""
        return self.dataset_dir if self.is_available() else None

    def is_available(self) -> bool:
        """Whether a verified dataset is already on disk."""
        try:
            self.verify()
            return True
        except DatasetVerificationError:
            return False

    def download(self) -> Path:
        """Download the dataset from Kaggle and link it into ``datasets/``."""
        self.logger.info(
            "Downloading '%s' via kagglehub (this can take a while)...",
            self.config.kaggle_handle,
        )
        source_dir = self._kagglehub_download()
        self.logger.info("Dataset fetched into kagglehub cache: %s", source_dir)

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self._materialise(source_dir)

        self.verify()  # raises DatasetVerificationError on any problem
        self.logger.info("Dataset ready at %s", self.dataset_dir)
        return self.dataset_dir

    def verify(self) -> None:
        """Validate that every expected file exists and is large enough.

        Raises
        ------
        DatasetVerificationError
            If any expected file is missing or smaller than the configured
            minimum size (guards against truncated downloads / LFS pointers).
        """
        missing: list[str] = []
        too_small: list[str] = []

        for name in self.config.expected_files:
            file_path = self.dataset_dir / name
            if not file_path.exists():
                missing.append(name)
                continue
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb < self.config.min_file_size_mb:
                too_small.append(f"{name} ({size_mb:.1f} MB < {self.config.min_file_size_mb} MB)")

        if missing or too_small:
            problems = []
            if missing:
                problems.append(f"missing files: {missing}")
            if too_small:
                problems.append(f"undersized files: {too_small}")
            raise DatasetVerificationError(
                f"Dataset verification failed in {self.dataset_dir}: {'; '.join(problems)}"
            )

    def describe(self) -> dict[str, Any]:
        """Return a small human/JSON-friendly summary of the dataset state."""
        files = []
        for name in self.config.expected_files:
            fp = self.dataset_dir / name
            files.append(
                {
                    "name": name,
                    "path": str(fp),
                    "exists": fp.exists(),
                    "size_mb": round(fp.stat().st_size / (1024 * 1024), 2) if fp.exists() else 0.0,
                }
            )
        return {
            "kaggle_handle": self.config.kaggle_handle,
            "dataset_dir": str(self.dataset_dir),
            "available": self.is_available(),
            "files": files,
        }

    # -- Internals ----------------------------------------------------------
    def _kagglehub_download(self) -> Path:
        try:
            import kagglehub
        except ImportError as exc:  # pragma: no cover - dependency guaranteed in practice
            raise DatasetDownloadError(
                "kagglehub is not installed. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        try:
            raw_path = kagglehub.dataset_download(
                self.config.kaggle_handle,
                force_download=self.config.force_download,
            )
        except TypeError:
            # Older kagglehub versions lack the force_download keyword.
            raw_path = kagglehub.dataset_download(self.config.kaggle_handle)
        except Exception as exc:  # noqa: BLE001 - surface a helpful, actionable message
            raise DatasetDownloadError(self._download_hint(exc)) from exc

        path = Path(raw_path)
        if not path.exists():
            raise DatasetDownloadError(
                f"kagglehub reported success but returned a non-existent path: {path}"
            )
        return path

    def _materialise(self, source_dir: Path) -> None:
        """Link/copy each expected file from the kagglehub cache into ``datasets/``."""
        for name in self.config.expected_files:
            src = self._find_file(source_dir, name)
            if src is None:
                raise DatasetDownloadError(
                    f"Expected file '{name}' was not found anywhere under {source_dir}. "
                    f"The Kaggle dataset layout may have changed; check {_KAGGLE_DATASET_URL}."
                )
            self._place(src, self.dataset_dir / name)

    @staticmethod
    def _find_file(root: Path, name: str) -> Path | None:
        direct = root / name
        if direct.exists():
            return direct
        matches = sorted(root.rglob(name))
        return matches[0] if matches else None

    def _place(self, src: Path, dst: Path) -> None:
        """Expose ``src`` at ``dst`` using the configured link mode."""
        if dst.is_symlink() or dst.exists():
            dst.unlink()

        mode = (self.config.link_mode or "symlink").lower()
        if mode == "copy":
            self.logger.info("Copying %s -> %s", src, dst)
            shutil.copy2(src, dst)
            return

        try:
            dst.symlink_to(src.resolve())
            self.logger.info("Linked %s -> %s", dst, src)
        except OSError as exc:
            self.logger.warning("Symlink failed (%s); copying instead.", exc)
            shutil.copy2(src, dst)

    def _download_hint(self, error: Exception) -> str:
        return (
            f"Failed to download '{self.config.kaggle_handle}' from Kaggle: {error}\n"
            "This is often an authentication issue. Provide Kaggle credentials via "
            "either environment variables:\n"
            "    export KAGGLE_USERNAME=<username> KAGGLE_KEY=<api_key>\n"
            "or a token file at ~/.kaggle/kaggle.json (chmod 600). See .env.example.\n"
            f"Dataset page: {_KAGGLE_DATASET_URL}"
        )
