"""
Configuration management for scryptorum integration with existing Python projects.
"""

import json
import os
from pathlib import Path
from typing import Optional, Union


class ScryptorumConfig:
    """Manages .scryptorum configuration file for project integration."""

    CONFIG_FILENAME = ".scryptorum"

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.cwd() / self.CONFIG_FILENAME
        self._config_data = None

    @classmethod
    def find_config(
        cls, start_path: Optional[Path] = None
    ) -> Optional["ScryptorumConfig"]:
        """
        Find .scryptorum config file by walking up the directory tree.

        Args:
            start_path: Directory to start searching from (default: current directory)

        Returns:
            ScryptorumConfig instance if found, None otherwise
        """
        search_path = start_path or Path.cwd()

        # Walk up the directory tree looking for .scryptorum file
        for parent in [search_path] + list(search_path.parents):
            config_file = parent / cls.CONFIG_FILENAME
            if config_file.exists():
                return cls(config_file)

        return None

    def exists(self) -> bool:
        """Check if config file exists."""
        return self.config_path.exists()

    def load(self) -> dict:
        """Load configuration from file."""
        if self._config_data is None:
            if not self.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")

            with open(self.config_path, "r") as f:
                self._config_data = json.load(f)

        return self._config_data

    def save(self, config_data: dict) -> None:
        """Save configuration to file."""
        with open(self.config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        self._config_data = config_data

    def create(
        self, experiments_dir: Union[str, Path], project_name: Optional[str] = None
    ) -> None:
        """
        Create a new .scryptorum config file.

        Args:
            experiments_dir: Path to the scryptorum experiments directory
            project_name: Optional project name (defaults to directory name)
        """
        experiments_path = Path(experiments_dir).resolve()

        if project_name is None:
            project_name = self.config_path.parent.name

        config_data = {
            "version": "1.0",
            "project_name": project_name,
            "experiments_dir": str(experiments_path),
            "created_from": str(self.config_path.parent.resolve()),
        }

        self.save(config_data)

    def get_experiments_dir(self) -> Path:
        """Get the experiments directory path."""
        config = self.load()
        return Path(config["experiments_dir"])

    def get_project_name(self) -> str:
        """Get the project name."""
        config = self.load()
        return config["project_name"]

    def get_project_root(self) -> Path:
        """Get the full project root path (experiments_dir/project_name)."""
        return self.get_experiments_dir() / self.get_project_name()


def find_scryptorum_project() -> Optional[Path]:
    """
    Find the scryptorum project root by looking for .scryptorum config.

    Returns:
        Project root path if found, None otherwise
    """
    config = ScryptorumConfig.find_config()
    if config:
        return config.get_project_root()
    return None


def resolve_experiments_dir(experiments_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve experiments directory from multiple sources.

    Priority:
    1. Explicit experiments_dir parameter
    2. SCRYPTORUM_EXPERIMENTS_DIR environment variable
    3. Current directory / "experiments"

    Args:
        experiments_dir: Explicit experiments directory path

    Returns:
        Resolved experiments directory path
    """
    if experiments_dir is not None:
        return Path(experiments_dir).resolve()

    env_dir = os.getenv("SCRYPTORUM_EXPERIMENTS_DIR")
    if env_dir:
        return Path(env_dir).resolve()

    return Path.cwd() / "experiments"
