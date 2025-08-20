"""
Experiment management and project structure.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from .runs import Run, RunType
from .config import ConfigManager
from .project_config import resolve_experiments_dir

# Removed logging_utils import - now use logger directly from runs


class Experiment:
    """Manages experiment lifecycle and run creation."""

    def __init__(self, project_root: Union[str, Path], experiment_name: str):
        self.project_root = Path(project_root)
        self.experiment_name = experiment_name
        # Use resolve_experiments_dir to support SCRYPTORUM_EXPERIMENTS_DIR env var
        # If a project_root is provided, use its experiments subdirectory
        experiments_base_dir = resolve_experiments_dir(
            self.project_root / "experiments"
        )
        self.experiment_path = experiments_base_dir / experiment_name

        # Ensure experiment directory structure exists
        self._setup_experiment()

        # Initialize config manager
        self.config = ConfigManager(self.config_dir)

        # Create default configs after managers are initialized
        self._create_default_configs()

    def _setup_experiment(self) -> None:
        """Create experiment directory structure."""
        directories = [
            self.experiment_path,
            self.experiment_path / "runs",
            self.experiment_path / "data",
            self.experiment_path / "config",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        # Create or update experiment metadata
        self._update_experiment_metadata()

    def _update_experiment_metadata(self) -> None:
        """Create or update experiment metadata file."""
        metadata_file = self.experiment_path / "experiment.json"

        metadata = {
            "name": self.experiment_name,
            "project_root": str(self.project_root),
            "created_at": (
                metadata_file.stat().st_ctime if metadata_file.exists() else None
            ),
        }

        if not metadata_file.exists():
            from datetime import datetime

            metadata["created_at"] = datetime.now().isoformat()

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def _create_default_configs(self) -> None:
        """Create default configuration files if needed."""
        # Base class does nothing - subclasses can override
        pass

    def create_run(
        self, run_type: RunType = RunType.TRIAL, run_id: Optional[str] = None
    ) -> Run:
        """Create a run of the specified type."""
        run = Run(self.experiment_path, run_type, run_id)

        # Milestone runs can be extended by subclasses for additional snapshots

        return run

    def list_runs(self) -> List[Dict[str, str]]:
        """List all runs in this experiment."""
        runs_dir = self.experiment_path / "runs"
        if not runs_dir.exists():
            return []

        runs = []
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                run_log = run_dir / "run.jsonl"
                if run_log.exists():
                    # Read first line to get run metadata
                    with open(run_log, "r") as f:
                        first_line = f.readline().strip()
                        if first_line:
                            try:
                                metadata = json.loads(first_line)
                                runs.append(
                                    {
                                        "run_id": metadata.get("run_id", run_dir.name),
                                        "run_type": metadata.get("run_type", "unknown"),
                                        "start_time": metadata.get(
                                            "start_time", "unknown"
                                        ),
                                        "path": str(run_dir),
                                    }
                                )
                            except json.JSONDecodeError:
                                continue

        return sorted(runs, key=lambda x: x["start_time"], reverse=True)

    def load_run(self, run_id: str) -> Optional[Run]:
        """Load an existing run by ID."""
        run_dir = self.experiment_path / "runs" / run_id
        if not run_dir.exists():
            return None

        # Determine run type from logs
        run_log = run_dir / "run.jsonl"
        if not run_log.exists():
            return None

        with open(run_log, "r") as f:
            first_line = f.readline().strip()
            if first_line:
                try:
                    metadata = json.loads(first_line)
                    run_type_str = metadata.get("run_type", "trial")
                    run_type = RunType(run_type_str)
                    return Run(self.experiment_path, run_type, run_id)
                except (json.JSONDecodeError, ValueError):
                    pass

        return None

    @property
    def data_dir(self) -> Path:
        """Get experiment data directory."""
        return self.experiment_path / "data"

    @property
    def config_dir(self) -> Path:
        """Get experiment config directory."""
        return self.experiment_path / "config"


def create_project(name: str, parent_path: Optional[Union[str, Path]] = None) -> Path:
    """Create a new scryptorum project structure."""
    # Determine parent directory
    if parent_path is not None:
        parent_dir = Path(parent_path)
    else:
        # Check environment variable first, then fall back to current directory
        env_parent_dir = os.getenv("SCRYPTORUM_PROJECTS_DIR")
        if env_parent_dir:
            parent_dir = Path(env_parent_dir)
        else:
            parent_dir = Path(".")

    project_root = parent_dir / name

    directories = [
        project_root,
        project_root / "experiments",
        project_root / "data",
        project_root / "models",
        project_root / "artifacts",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    # Create project metadata
    project_config = {"name": name, "framework": "scryptorum", "version": "0.1.0"}

    with open(project_root / "project.json", "w") as f:
        json.dump(project_config, f, indent=2)

    return project_root


def resolve_project_root(
    project_name: str, parent_path: Optional[Union[str, Path]] = None
) -> Path:
    """Resolve the full project root path from project name and optional parent path."""
    # Determine parent directory
    if parent_path is not None:
        parent_dir = Path(parent_path)
    else:
        # Check environment variable first, then fall back to current directory
        env_parent_dir = os.getenv("SCRYPTORUM_PROJECTS_DIR")
        if env_parent_dir:
            parent_dir = Path(env_parent_dir)
        else:
            parent_dir = Path(".")

    return parent_dir / project_name
