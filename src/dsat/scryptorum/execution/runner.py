"""
Execution engine for running experiments.
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

from ..core.decorators import run_context
from ..core.experiment import Experiment
from ..core.runs import Run, RunType

# Removed logging_utils import - now use logger directly from runs


class Runner:
    """Orchestrates experiment execution."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)

    def _get_diagnostic_info(self) -> str:
        """Get diagnostic information for debugging path issues."""
        return (
            f"Current working directory: {os.getcwd()}\n"
            f"Python path: {sys.path}\n"
            f"Project root: {self.project_root}\n"
            f"sys.modules keys (first 10): {list(sys.modules.keys())[:10]}"
        )

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
                    run.logger.info(f"Found runnable class: {runnable_class.__name__}")
                    runnable = runnable_class(experiment, run, config or {})
                    run.logger.debug(f"Created runnable instance: {type(runnable)}")
                    self._execute_runnable(runnable, run)
                else:
                    run.log_event("warning", {"message": "No runnable specified"})

                run.finish()
                return run

        except Exception as e:
            run.log_event("execution_error", {"error": str(e)})
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
            run.log_event("execution_error", {"error": str(e)})
            run.finish()
            raise

    def _load_runnable_class(self, module_path: str) -> Type:
        """Dynamically load a runnable class from module path."""
        # Add current working directory's src folder to Python path if it exists
        cwd_src = Path.cwd() / "src"
        if cwd_src.exists() and str(cwd_src) not in sys.path:
            sys.path.insert(0, str(cwd_src))

        try:
            # Handle different module path formats
            if "/" in module_path or "\\" in module_path or module_path.endswith(".py"):
                # File path format
                module_file = Path(module_path)
                if not module_file.exists():
                    diagnostic_info = self._get_diagnostic_info()
                    raise FileNotFoundError(
                        f"Module file not found: {module_path}\n"
                        f"Resolved path: {module_file.resolve()}\n"
                        f"{diagnostic_info}"
                    )

                spec = importlib.util.spec_from_file_location(
                    "runnable_module", module_file
                )
                if spec is None or spec.loader is None:
                    diagnostic_info = self._get_diagnostic_info()
                    raise ImportError(
                        f"Could not load spec from {module_path}\n"
                        f"Module file: {module_file}\n"
                        f"Module file exists: {module_file.exists()}\n"
                        f"{diagnostic_info}"
                    )

                module = importlib.util.module_from_spec(spec)
                sys.modules["runnable_module"] = module
                spec.loader.exec_module(module)

                # Look for runnable class
                found_classes = []
                runnable_candidates = []
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type):
                        found_classes.append(attr_name)
                        if (
                            hasattr(attr, "execute")
                            and callable(getattr(attr, "execute"))
                        ) or (hasattr(attr, "run") and callable(getattr(attr, "run"))):
                            # Found a runnable class (debug)
                            # Skip BaseRunnable itself - we want concrete implementations
                            if attr_name != "BaseRunnable":
                                runnable_candidates.append(attr)

                # Return the first concrete runnable found
                if runnable_candidates:
                    return runnable_candidates[0]

                # All classes in module (debug): {found_classes}
                diagnostic_info = self._get_diagnostic_info()
                available_attrs = [
                    attr for attr in dir(module) if not attr.startswith("_")
                ]
                raise ValueError(
                    f"No runnable class found in {module_path}\n"
                    f"Available attributes: {available_attrs}\n"
                    f"All classes: {found_classes}\n"
                    f"{diagnostic_info}"
                )
            elif "." in module_path:
                # Package.module format
                try:
                    module = importlib.import_module(module_path)
                except ImportError as e:
                    diagnostic_info = self._get_diagnostic_info()
                    raise ImportError(
                        f"Failed to import module {module_path}: {e}\n"
                        f"{diagnostic_info}"
                    )
                # Look for a class that has a 'run' method
                found_classes = []
                runnable_candidates = []
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type):
                        found_classes.append(attr_name)
                        if (
                            hasattr(attr, "execute")
                            and callable(getattr(attr, "execute"))
                        ) or (hasattr(attr, "run") and callable(getattr(attr, "run"))):
                            # Found a runnable class (debug)
                            # Skip BaseRunnable itself - we want concrete implementations
                            if attr_name != "BaseRunnable":
                                runnable_candidates.append(attr)

                # Return the first concrete runnable found
                if runnable_candidates:
                    return runnable_candidates[0]

                # All classes in module (debug): {found_classes}
                diagnostic_info = self._get_diagnostic_info()
                available_attrs = [
                    attr for attr in dir(module) if not attr.startswith("_")
                ]
                raise ValueError(
                    f"No runnable class found in {module_path}\n"
                    f"Available attributes: {available_attrs}\n"
                    f"All classes: {found_classes}\n"
                    f"{diagnostic_info}"
                )
            else:
                # Simple module name format (no dots or paths)
                try:
                    module = importlib.import_module(module_path)
                except ImportError as e:
                    diagnostic_info = self._get_diagnostic_info()
                    raise ImportError(
                        f"Failed to import module {module_path}: {e}\n"
                        f"{diagnostic_info}"
                    )
                # Look for a class that has a 'run' method
                found_classes = []
                runnable_candidates = []
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type):
                        found_classes.append(attr_name)
                        if (
                            hasattr(attr, "execute")
                            and callable(getattr(attr, "execute"))
                        ) or (hasattr(attr, "run") and callable(getattr(attr, "run"))):
                            # Found a runnable class (debug)
                            # Skip BaseRunnable itself - we want concrete implementations
                            if attr_name != "BaseRunnable":
                                runnable_candidates.append(attr)

                # Return the first concrete runnable found
                if runnable_candidates:
                    return runnable_candidates[0]

                # All classes in module (debug): {found_classes}
                diagnostic_info = self._get_diagnostic_info()
                available_attrs = [
                    attr for attr in dir(module) if not attr.startswith("_")
                ]
                raise ValueError(
                    f"No runnable class found in {module_path}\n"
                    f"Available attributes: {available_attrs}\n"
                    f"All classes: {found_classes}\n"
                    f"{diagnostic_info}"
                )

        except Exception as e:
            raise ImportError(f"Failed to load runnable from {module_path}: {e}")

    def _execute_runnable(self, runnable: Any, run: Run) -> None:
        """Execute a runnable instance through its lifecycle."""
        stages = ["prepare", "execute", "score", "cleanup"]
        run.logger.info(f"Executing runnable with stages: {stages}")

        last_exception = None
        skip_score = False

        for stage in stages:
            # Skip score stage if execute failed
            if stage == "score" and skip_score:
                continue

            run.logger.debug(f"Checking stage '{stage}'...")
            if hasattr(runnable, stage):
                method = getattr(runnable, stage)
                if callable(method):
                    run.logger.info(f"Executing {stage}() method...")
                    try:
                        method()
                        run.logger.info(f"Completed {stage}() method")
                    except Exception as e:
                        # Log stage error but continue to cleanup
                        if hasattr(runnable, "run") and hasattr(
                            runnable.run, "_log_event"
                        ):
                            runnable.run.log_event(f"{stage}_error", {"error": str(e)})
                        if stage == "cleanup":
                            # Don't re-raise cleanup errors
                            continue
                        else:
                            # Save the exception but continue to cleanup
                            last_exception = e
                            if stage == "execute":
                                # Skip score stage after execute failure
                                skip_score = True

        # Re-raise the last exception after cleanup has run
        if last_exception:
            raise last_exception


class BaseRunnable:
    """Base class for experiment runnables."""

    def __init__(self, experiment: Experiment, run: Run, config: Dict[str, Any]):
        self.experiment = experiment
        self.run = run
        self.config = config

    def prepare(self) -> None:
        """Override to add preparation logic."""
        pass

    def execute(self) -> None:
        """Override to add main execution logic."""
        raise NotImplementedError("Subclasses must implement execute()")

    def run(self) -> None:
        """Deprecated: Use execute() instead. Kept for backward compatibility."""
        return self.execute()

    def score(self) -> None:
        """Override to add scoring/evaluation logic."""
        pass

    def cleanup(self) -> None:
        """Override to add cleanup logic."""
        pass
