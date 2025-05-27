"""
Experiment management and project structure.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from .runs import Run, RunType
from .prompts import PromptManager
from .config import ConfigManager, create_default_agent_config
from .logging_utils import log_info


class Experiment:
    """Manages experiment lifecycle and run creation."""

    def __init__(self, project_root: Union[str, Path], experiment_name: str):
        self.project_root = Path(project_root)
        self.experiment_name = experiment_name
        self.experiment_path = self.project_root / "experiments" / experiment_name

        # Ensure experiment directory structure exists
        self._setup_experiment()

        # Initialize prompt manager
        self.prompts = PromptManager(self.prompts_dir)

        # Initialize config manager
        self.config = ConfigManager(self.config_dir)

        # Create default configs after managers are initialized
        self._create_default_configs()
        
        # Create sample prompt if no prompts exist
        self._create_sample_prompt()

    def _setup_experiment(self) -> None:
        """Create experiment directory structure."""
        directories = [
            self.experiment_path,
            self.experiment_path / "runs",
            self.experiment_path / "data",
            self.experiment_path / "config",
            self.experiment_path / "prompts",
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
        """Create default agent configuration files."""
        # Create a default agent config if no configs exist
        existing_configs = list(self.config_dir.glob("*_config.json"))

        if not existing_configs:
            # Create a default agent config
            default_agent_name = f"{self.experiment_name}_agent"
            default_config = create_default_agent_config(default_agent_name)
            config_file = self.config.create_agent_config(
                default_agent_name, default_config
            )

            # Log the creation
            log_info(f"Created default agent config: {config_file}")

    def _create_sample_prompt(self) -> None:
        """Create a sample prompt file if no prompts exist."""
        # Check if any prompts already exist
        existing_prompts = self.prompts.list_prompts()
        
        if not existing_prompts:
            # Create a simple sample prompt
            sample_prompt_text = """You are a helpful AI assistant working on an experiment.

Please analyze the given data and provide insights based on the following context:
- Experiment name: {experiment_name}
- Task: [Describe your specific task here]
- Data: [Your data will be provided here]

Respond with clear, actionable insights and recommendations.
"""
            
            # Create the sample prompt file
            prompt_file = self.prompts.create_prompt("sample_prompt", sample_prompt_text)
            log_info(f"Created sample prompt: {prompt_file}")

    def create_agent_config(self, agent_name: str, **config_overrides) -> Path:
        """Create a new agent configuration with optional overrides."""
        default_config = create_default_agent_config(agent_name)

        # Apply any overrides
        for key, value in config_overrides.items():
            if hasattr(default_config, key):
                setattr(default_config, key, value)
            elif key in ["llm_meta", "custom_configs"]:
                # For nested dictionaries, update them
                if key == "llm_meta":
                    default_config.llm_meta.update(value)
                elif key == "custom_configs":
                    default_config.custom_configs.update(value)

        return self.config.create_agent_config(agent_name, default_config)

    def create_run(
        self, run_type: RunType = RunType.TRIAL, run_id: Optional[str] = None
    ) -> Run:
        """Create a run of the specified type."""
        run = Run(self.experiment_path, run_type, run_id)

        # For milestone runs, automatically snapshot prompts
        if run_type == RunType.MILESTONE:
            run.snapshot_prompts(self.prompts_dir)

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

    @property
    def prompts_dir(self) -> Path:
        """Get experiment prompts directory."""
        return self.experiment_path / "prompts"


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
