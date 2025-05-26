"""
Execution engine for running experiments.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

from ..core.decorators import run_context
from ..core.experiment import Experiment
from ..core.runs import Run, RunType


class Runner:
    """Orchestrates experiment execution."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)

    def run_experiment(
        self,
        experiment_name: str,
        runnable_class: Optional[Type] = None,
        runnable_module: Optional[str] = None,
        run_type: RunType = RunType.TRIAL,
        run_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Run:
        """
        Execute an experiment with the specified runnable.

        Args:
            experiment_name: Name of the experiment
            runnable_class: Class to execute (must have run() method)
            runnable_module: Module path containing runnable class
            run_type: Type of run to create
            run_id: Optional specific run ID
            config: Configuration to pass to runnable
        """
        # Create experiment and run
        experiment = Experiment(self.project_root, experiment_name)
        run = experiment.create_run(run_type, run_id)

        try:
            with run_context(run):
                # Load runnable class if module specified
                if runnable_module and not runnable_class:
                    runnable_class = self._load_runnable_class(runnable_module)

                # Execute the runnable
                if runnable_class:
                    runnable = runnable_class(experiment, run, config or {})
                    self._execute_runnable(runnable)
                else:
                    run._log_event("warning", {"message": "No runnable specified"})

                run.finish()
                return run

        except Exception as e:
            run._log_event("execution_error", {"error": str(e)})
            run.finish()
            raise

    def run_function(
        self,
        experiment_name: str,
        func: Callable,
        run_type: RunType = RunType.TRIAL,
        run_id: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute a single function as an experiment.

        Args:
            experiment_name: Name of the experiment
            func: Function to execute
            run_type: Type of run to create
            run_id: Optional specific run ID
            *args, **kwargs: Arguments to pass to function
        """
        experiment = Experiment(self.project_root, experiment_name)
        run = experiment.create_run(run_type, run_id)

        try:
            with run_context(run):
                result = func(*args, **kwargs)
                run.finish()
                return result

        except Exception as e:
            run._log_event("execution_error", {"error": str(e)})
            run.finish()
            raise

    def _load_runnable_class(self, module_path: str) -> Type:
        """Dynamically load a runnable class from module path."""
        try:
            # Handle different module path formats
            if "." in module_path:
                # Package.module format
                module = importlib.import_module(module_path)
                # Look for a class that has a 'run' method
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and hasattr(attr, "run")
                        and callable(getattr(attr, "run"))
                    ):
                        return attr
                raise ValueError(f"No runnable class found in {module_path}")
            else:
                # File path format
                module_file = Path(module_path)
                if not module_file.exists():
                    raise FileNotFoundError(f"Module file not found: {module_path}")

                spec = importlib.util.spec_from_file_location(
                    "runnable_module", module_file
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load spec from {module_path}")

                module = importlib.util.module_from_spec(spec)
                sys.modules["runnable_module"] = module
                spec.loader.exec_module(module)

                # Look for runnable class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and hasattr(attr, "run")
                        and callable(getattr(attr, "run"))
                    ):
                        return attr

                raise ValueError(f"No runnable class found in {module_path}")

        except Exception as e:
            raise ImportError(f"Failed to load runnable from {module_path}: {e}")

    def _execute_runnable(self, runnable: Any) -> None:
        """Execute a runnable instance through its lifecycle."""
        stages = ["prepare", "run", "score", "cleanup"]

        for stage in stages:
            if hasattr(runnable, stage):
                method = getattr(runnable, stage)
                if callable(method):
                    try:
                        method()
                    except Exception as e:
                        # Log stage error but continue to cleanup
                        if hasattr(runnable, "run") and hasattr(
                            runnable.run, "_log_event"
                        ):
                            runnable.run._log_event(f"{stage}_error", {"error": str(e)})
                        if stage == "cleanup":
                            # Don't re-raise cleanup errors
                            continue
                        raise


class BaseRunnable:
    """Base class for experiment runnables."""

    def __init__(self, experiment: Experiment, run: Run, config: Dict[str, Any]):
        self.experiment = experiment
        self.run = run
        self.config = config

    def prepare(self) -> None:
        """Override to add preparation logic."""
        pass

    def run(self) -> None:
        """Override to add main execution logic."""
        raise NotImplementedError("Subclasses must implement run()")

    def score(self) -> None:
        """Override to add scoring/evaluation logic."""
        pass

    def cleanup(self) -> None:
        """Override to add cleanup logic."""
        pass
