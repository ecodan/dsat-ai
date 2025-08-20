"""
Run management system supporting lightweight trials and full milestone runs.
"""

import json
import time
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

from .logging_utils import create_experiment_logger, cleanup_logger


class RunType(Enum):
    """Type of experiment run."""

    TRIAL = "trial"  # Lightweight: logs only, no asset versioning
    MILESTONE = "milestone"  # Full: complete versioning and artifact capture


class Run:
    """Unified run class that handles both trial and milestone runs."""

    def __init__(
        self, experiment_path: Path, run_type: RunType, run_id: Optional[str] = None
    ):
        self.experiment_path = experiment_path
        self.run_type = run_type
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None

        # For trial runs, always use "trial_run" directory (gets reset)
        # For milestone runs, use temporally sequential run_id directory
        if run_type == RunType.TRIAL:
            self.run_id = "trial_run"
        else:
            self.run_id = run_id or self._generate_milestone_run_id()

        # Thread safety for logging
        self._lock = threading.RLock()

        # Initialize run directory and logging
        self._setup_run()

        # Create experiment logger
        self.logger = create_experiment_logger(self.run_dir)

    def _setup_run(self) -> None:
        """Setup run directory and logging infrastructure."""
        self.run_dir = self.experiment_path / "runs" / self.run_id

        # For trial runs, clear the directory if it exists
        if self.run_type == RunType.TRIAL and self.run_dir.exists():
            import shutil

            shutil.rmtree(self.run_dir)

        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Core log files
        self.log_file = self.run_dir / "run.jsonl"
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.timings_file = self.run_dir / "timings.jsonl"

        # Create empty log files
        self.metrics_file.touch()
        self.timings_file.touch()

        # For milestone runs, create additional directories
        if self.run_type == RunType.MILESTONE:
            self.artifacts_dir = self.run_dir / "artifacts"
            self.code_snapshot_dir = self.run_dir / "code_snapshot"
            self.agent_configs_snapshot_dir = self.run_dir / "agent_configs"
            self.artifacts_dir.mkdir(exist_ok=True)
            self.code_snapshot_dir.mkdir(exist_ok=True)
            self.agent_configs_snapshot_dir.mkdir(exist_ok=True)
        else:
            # Ensure these attributes exist for trial runs too (they just won't be used)
            self.artifacts_dir = self.run_dir / "artifacts"
            self.code_snapshot_dir = self.run_dir / "code_snapshot"
            self.agent_configs_snapshot_dir = self.run_dir / "agent_configs"

        # Initialize with run metadata
        self.log_event(
            "run_started",
            {
                "run_id": self.run_id,
                "run_type": self.run_type.value,
                "start_time": self.start_time.isoformat(),
            },
        )

    def _generate_milestone_run_id(self) -> str:
        """Generate a temporally sequential, human-readable milestone run ID."""
        # Format: ms-YYYYMMDD-HHMMSS-{4char_hex}
        # Example: ms-20240315-143022-a4b7
        timestamp = self.start_time.strftime("%Y%m%d-%H%M%S")
        unique_suffix = uuid.uuid4().hex[:4]
        return f"ms-{timestamp}-{unique_suffix}"

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Thread-safe event logging."""
        with self._lock:
            event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "run_id": self.run_id,
                **data,
            }
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event) + "\n")

    def log_metric(
        self,
        name: str,
        value: Union[float, int],
        metric_type: str = "custom",
        **metadata,
    ) -> None:
        """Log a metric value."""
        metric_data = {"name": name, "value": value, "type": metric_type, **metadata}

        with self._lock:
            with open(self.metrics_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "run_id": self.run_id,
                            **metric_data,
                        }
                    )
                    + "\n"
                )

        self.log_event("metric_logged", metric_data)

    def log_timing(self, operation: str, duration_ms: float, **metadata) -> None:
        """Log timing information."""
        timing_data = {"operation": operation, "duration_ms": duration_ms, **metadata}

        with self._lock:
            with open(self.timings_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "run_id": self.run_id,
                            **timing_data,
                        }
                    )
                    + "\n"
                )

        self.log_event("timing_logged", timing_data)

    def log_llm_call(
        self,
        model: str,
        input_data: Any,
        output_data: Any,
        duration_ms: Optional[float] = None,
        **metadata,
    ) -> None:
        """Log LLM invocation details."""
        llm_data = {
            "model": model,
            "input": input_data,
            "output": output_data,
            "duration_ms": duration_ms,
            **metadata,
        }
        self.log_event("llm_call", llm_data)

    def finish(self) -> None:
        """Mark run as completed."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        self.log_event(
            "run_finished",
            {"end_time": self.end_time.isoformat(), "duration_seconds": duration},
        )

        # Log completion message
        self.logger.info(f"Run {self.run_id} completed in {duration:.2f} seconds")

        # Clean up logging when run finishes
        cleanup_logger(self.logger)

    def preserve_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Preserve run artifacts (behavior varies by run type)."""
        if self.run_type == RunType.TRIAL:
            # For trial runs, only log artifact metadata
            self.log_event(
                "artifacts_logged",
                {
                    "artifact_count": len(artifacts),
                    "artifact_types": list(artifacts.keys()),
                },
            )
        else:
            # For milestone runs, save complete artifacts to disk
            preserved = {}

            for name, artifact in artifacts.items():
                artifact_path = self.artifacts_dir / f"{name}.json"

                try:
                    with open(artifact_path, "w") as f:
                        json.dump(artifact, f, indent=2, default=str)
                    preserved[name] = str(artifact_path)
                except Exception as e:
                    self.log_event(
                        "artifact_save_error", {"artifact_name": name, "error": str(e)}
                    )

            self.log_event(
                "artifacts_preserved",
                {"preserved_artifacts": preserved, "artifact_count": len(preserved)},
            )

    def snapshot_code(self, source_paths: List[Path]) -> None:
        """Create code snapshot for reproducibility (milestone runs only)."""
        if self.run_type != RunType.MILESTONE:
            self.log_event("code_snapshot_skipped", {"reason": "trial_run_type"})
            return

        import shutil

        for source_path in source_paths:
            if source_path.exists():
                if source_path.is_file():
                    dest = self.code_snapshot_dir / source_path.name
                    shutil.copy2(source_path, dest)
                elif source_path.is_dir():
                    dest = self.code_snapshot_dir / source_path.name
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(source_path, dest)

        self.log_event(
            "code_snapshot_created",
            {
                "snapshot_path": str(self.code_snapshot_dir),
                "source_paths": [str(p) for p in source_paths],
            },
        )

    def snapshot_agent_configs(self, config_dir: Path) -> None:
        """Create agent configs snapshot for reproducibility (milestone runs only)."""
        if self.run_type != RunType.MILESTONE:
            self.log_event(
                "agent_configs_snapshot_skipped", {"reason": "trial_run_type"}
            )
            return

        if not config_dir.exists():
            self.log_event(
                "agent_configs_snapshot_skipped",
                {"reason": "config_dir_not_found", "config_dir": str(config_dir)},
            )
            return

        import shutil

        # Copy all agent config JSON files
        copied_files = []
        for config_file in config_dir.glob("*_config.json"):
            dest_file = self.agent_configs_snapshot_dir / config_file.name
            shutil.copy2(config_file, dest_file)
            copied_files.append(config_file.name)

        self.log_event(
            "agent_configs_snapshot_created",
            {
                "snapshot_path": str(self.agent_configs_snapshot_dir),
                "source_dir": str(config_dir),
                "copied_files": copied_files,
                "file_count": len(copied_files),
            },
        )


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, run: Run, operation: str, **metadata):
        self.run = run
        self.operation = operation
        self.metadata = metadata
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time() * 1000  # Convert to milliseconds
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() * 1000) - self.start_time
            self.run.log_timing(self.operation, duration_ms, **self.metadata)
