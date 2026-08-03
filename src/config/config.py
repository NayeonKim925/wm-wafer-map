"""Typed, hierarchical configuration.

The configuration is a tree of :mod:`dataclasses` with sensible defaults, so it
is fully usable from code (``Config()``) and from YAML.  Precedence, lowest to
highest:

1. dataclass defaults (the safety net that keeps the code runnable and testable
   without any YAML file present),
2. values from a YAML file (the primary editable surface, ``configs/default.yaml``),
3. dotted CLI overrides such as ``training.epochs=5``.

All file-system paths are :class:`~pathlib.Path` instances and are resolved to
absolute paths (relative to the repo root) only at their point of use, so the
serialised config stays portable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from src.utils.paths import get_project_root

DEFAULT_CONFIG_PATH = "configs/default.yaml"


class ConfigError(ValueError):
    """Raised when a configuration file or override is invalid."""


# ---------------------------------------------------------------------------
# Configuration schema
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    """Where the dataset comes from and lands on disk."""

    kaggle_handle: str = "qingyi/wm811k-wafer-map"
    root_dir: Path = Path("datasets")
    dataset_name: str = "wm811k"
    pickle_filename: str = "LSWMD.pkl"
    expected_files: list[str] = field(default_factory=lambda: ["LSWMD.pkl"])
    min_file_size_mb: float = 100.0
    link_mode: str = "symlink"
    force_download: bool = False


@dataclass
class DataConfig:
    """How raw wafer maps become batched tensors."""

    image_size: int = 64
    representation: str = "onehot"
    include_none: bool = True
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    max_samples: int | None = None


@dataclass
class ModelConfig:
    """Model architecture selection and hyper-parameters."""

    name: str = "wafer_cnn"
    base_channels: int = 32
    dropout: float = 0.3
    num_classes: int | None = None  # derived from the label mapping at build time


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    step_size: int = 10
    gamma: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 30
    device: str = "auto"
    class_weighting: bool = True
    early_stopping_patience: int = 8
    grad_clip: float | None = 5.0
    mixed_precision: bool = True
    log_interval: int = 50
    evaluate_on_test: bool = True
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class OutputConfig:
    root_dir: Path = Path("outputs")
    experiment_name: str = "wafer_cnn"
    timestamp_subdir: bool = True


@dataclass
class Config:
    """Root configuration object."""

    seed: int = 42
    log_level: str = "INFO"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # -- Construction -------------------------------------------------------
    @classmethod
    def load(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> Config:
        """Build a :class:`Config` from a YAML file plus optional CLI overrides.

        Parameters
        ----------
        config_path:
            Path to a YAML config.  ``None`` uses ``configs/default.yaml`` if it
            exists, otherwise pure dataclass defaults.
        overrides:
            List of ``dotted.key=value`` strings (values parsed as YAML scalars).
        """
        data: dict[str, Any] = {}

        resolved = _resolve_config_path(config_path)
        if resolved is not None:
            data = _load_yaml(resolved)

        for override in overrides or []:
            _apply_override(data, override)

        try:
            return _instantiate(cls, data, path="")
        except ConfigError:
            raise
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ConfigError(f"Failed to build configuration: {exc}") from exc

    @classmethod
    def from_dict(cls, data: dict[str, Any], overrides: list[str] | None = None) -> Config:
        """Build a :class:`Config` directly from a dict (e.g. a checkpoint's config)."""
        merged = dict(data or {})
        for override in overrides or []:
            _apply_override(merged, override)
        return _instantiate(cls, merged, path="")

    # -- Serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a plain, YAML-serialisable dict (Paths become strings)."""
        return _stringify(asdict(self))

    def save(self, path: str | Path) -> Path:
        """Write the resolved config to ``path`` for run reproducibility."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_config_path(config_path: str | Path | None) -> Path | None:
    if config_path is not None:
        p = Path(config_path)
        if not p.is_absolute():
            p = get_project_root() / p
        if not p.is_file():
            raise ConfigError(f"Config file not found: {p}")
        return p

    default = get_project_root() / DEFAULT_CONFIG_PATH
    return default if default.is_file() else None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(loaded).__name__} in {path}")
    return loaded


def _apply_override(data: dict[str, Any], override: str) -> None:
    """Apply a single ``dotted.key=value`` override into ``data`` in place."""
    if "=" not in override:
        raise ConfigError(f"Override must be of the form key=value, got: {override!r}")
    key, _, raw_value = override.partition("=")
    key = key.strip()
    if not key:
        raise ConfigError(f"Override is missing a key: {override!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse override value in {override!r}: {exc}") from exc

    node = data
    parts = key.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _instantiate(cls: type, data: dict[str, Any], *, path: str) -> Any:
    """Recursively build dataclass ``cls`` from ``data``, validating keys."""
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a mapping at {path or '<root>'!r}, got {type(data).__name__}")

    hints = get_type_hints(cls)
    valid = {f.name for f in fields(cls)}
    for key in data:
        if key not in valid:
            location = f"{path}.{key}" if path else key
            raise ConfigError(
                f"Unknown config key {location!r}. Valid keys here: {sorted(valid)}"
            )

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints.get(f.name, f.type)
        child_path = f"{path}.{f.name}" if path else f.name
        if is_dataclass(ftype):
            kwargs[f.name] = _instantiate(ftype, value, path=child_path)
        elif ftype is Path:
            kwargs[f.name] = Path(value) if value is not None else None
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _stringify(obj: Any) -> Any:
    """Recursively convert Path values to strings for serialisation."""
    if isinstance(obj, dict):
        return {k: _stringify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
