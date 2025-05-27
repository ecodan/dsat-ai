"""
Annotation-based experiment framework using decorators.
"""

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional, TypeVar, Union

from .runs import Run, RunType

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])

# Global run context for decorator access
_current_run: Optional[Run] = None
_default_run_type: RunType = None  # Will be set by CLI or manually


def set_default_run_type(run_type: Union[RunType, str]) -> None:
    """Set the default run type for decorators."""
    global _default_run_type
    if isinstance(run_type, str):
        _default_run_type = RunType(run_type)
    else:
        _default_run_type = run_type


def get_default_run_type() -> RunType:
    """Get the default run type."""
    return _default_run_type or RunType.TRIAL


def set_current_run(run: Run) -> None:
    """Set the current run context for decorators."""
    global _current_run
    _current_run = run


def get_current_run() -> Optional[Run]:
    """Get the current run context."""
    return _current_run


@contextmanager
def run_context(run: Run):
    """Context manager for setting current run."""
    global _current_run
    previous_run = _current_run
    _current_run = run
    try:
        yield run
    finally:
        _current_run = previous_run


def experiment(name: Optional[str] = None):
    """
    Decorator to mark a function as an experiment.

    Args:
        name: Optional experiment name (defaults to function name)

    Note: Run type (trial vs milestone) is controlled by CLI flags or set_default_run_type()
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            experiment_name = name or func.__name__

            # Check if we already have a run context
            if _current_run is not None:
                return func(*args, **kwargs)

            # Create experiment and run
            from .experiment import Experiment

            # Try to extract experiment path from kwargs or use default
            project_root = kwargs.pop("project_root", ".")
            exp = Experiment(project_root, experiment_name)

            # Use the globally configured run type
            run_type_enum = get_default_run_type()
            run = exp.create_run(run_type_enum)

            try:
                with run_context(run):
                    result = func(*args, **kwargs)
                    run.finish()
                    return result
            except Exception as e:
                run.log_event("experiment_error", {"error": str(e)})
                run.finish()
                raise

        # Add metadata to function
        wrapper._scryptorum_experiment = True
        wrapper._experiment_name = name or func.__name__

        return wrapper

    return decorator


def metric(name: Optional[str] = None, metric_type: str = "custom"):
    """
    Decorator to automatically log function return value as a metric.

    Args:
        name: Metric name (defaults to function name)
        metric_type: Type of metric (accuracy, f1, loss, etc.)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if _current_run is not None:
                metric_name = name or func.__name__

                # Handle different return types
                if isinstance(result, (int, float)):
                    _current_run.log_metric(metric_name, result, metric_type)
                elif isinstance(result, dict) and "value" in result:
                    # Support returning dict with metadata
                    result_copy = result.copy()
                    value = result_copy.pop("value")
                    _current_run.log_metric(
                        metric_name, value, metric_type, **result_copy
                    )
                elif isinstance(result, tuple) and len(result) == 2:
                    # Support returning (value, metadata) tuple
                    value, metadata = result
                    _current_run.log_metric(metric_name, value, metric_type, **metadata)

            return result

        wrapper._scryptorum_metric = True
        wrapper._metric_name = name or func.__name__
        wrapper._metric_type = metric_type

        return wrapper

    return decorator


def timer(name: Optional[str] = None):
    """
    Decorator to automatically time function execution.

    Args:
        name: Timer name (defaults to function name)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or func.__name__

            start_time = time.time() * 1000  # Convert to milliseconds
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if _current_run is not None:
                    duration_ms = (time.time() * 1000) - start_time
                    _current_run.log_timing(timer_name, duration_ms)

        wrapper._scryptorum_timer = True
        wrapper._timer_name = name or func.__name__

        return wrapper

    return decorator


def llm_call(
    model: Optional[str] = None, log_input: bool = True, log_output: bool = True
):
    """
    Decorator to automatically log LLM calls.

    Args:
        model: Model identifier (if not provided, tries to extract from kwargs)
        log_input: Whether to log input data
        log_output: Whether to log output data
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time() * 1000

            # Extract model from kwargs if not provided
            model_name = model or kwargs.get("model", "unknown")

            # Capture input if requested
            input_data = None
            if log_input:
                # Try to extract meaningful input from args/kwargs
                if args:
                    input_data = args[0] if len(args) == 1 else args
                elif "prompt" in kwargs:
                    input_data = kwargs["prompt"]
                elif "input" in kwargs:
                    input_data = kwargs["input"]
                else:
                    input_data = kwargs

            try:
                result = func(*args, **kwargs)

                if _current_run is not None:
                    duration_ms = (time.time() * 1000) - start_time
                    output_data = result if log_output else None

                    _current_run.log_llm_call(
                        model=model_name,
                        input_data=input_data,
                        output_data=output_data,
                        duration_ms=duration_ms,
                    )

                return result
            except Exception as e:
                if _current_run is not None:
                    duration_ms = (time.time() * 1000) - start_time
                    _current_run.log_llm_call(
                        model=model_name,
                        input_data=input_data,
                        output_data=None,
                        duration_ms=duration_ms,
                        error=str(e),
                    )
                raise

        wrapper._scryptorum_llm_call = True
        wrapper._model = model
        wrapper._log_input = log_input
        wrapper._log_output = log_output

        return wrapper

    return decorator


def batch_processor(batch_size: int = 10, parallel: bool = False):
    """
    Decorator for batch processing with automatic logging.

    Args:
        batch_size: Size of each batch
        parallel: Whether to process batches in parallel
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(items, *args, **kwargs):
            if not hasattr(items, "__iter__"):
                raise ValueError("First argument must be iterable")

            results = []
            total_items = len(list(items)) if hasattr(items, "__len__") else None

            if _current_run is not None:
                _current_run.log_event(
                    "batch_processing_started",
                    {
                        "function": func.__name__,
                        "batch_size": batch_size,
                        "total_items": total_items,
                        "parallel": parallel,
                    },
                )

            # Process in batches
            items_list = list(items)
            for i in range(0, len(items_list), batch_size):
                batch = items_list[i : i + batch_size]

                if parallel:
                    # TODO: Implement parallel processing
                    batch_results = [func(item, *args, **kwargs) for item in batch]
                else:
                    batch_results = [func(item, *args, **kwargs) for item in batch]

                results.extend(batch_results)

                if _current_run is not None:
                    _current_run.log_event(
                        "batch_completed",
                        {
                            "batch_start": i,
                            "batch_size": len(batch),
                            "total_processed": len(results),
                        },
                    )

            if _current_run is not None:
                _current_run.log_event(
                    "batch_processing_finished", {"total_results": len(results)}
                )

            return results

        wrapper._scryptorum_batch = True
        wrapper._batch_size = batch_size
        wrapper._parallel = parallel

        return wrapper

    return decorator


# Convenience context managers
def time_operation(operation_name: str):
    """Context manager for timing operations."""
    if _current_run is None:
        return nullcontext()

    from .runs import TimerContext

    return TimerContext(_current_run, operation_name)


class nullcontext:
    """Null context manager for when no run is active."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
