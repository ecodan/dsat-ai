"""
Tests for decorator functionality.
"""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from dsat.scryptorum.core.decorators import (
    experiment,
    metric,
    timer,
    llm_call,
    batch_processor,
    set_default_run_type,
    get_default_run_type,
    set_current_run,
    get_current_run,
    run_context,
    time_operation,
)
from dsat.scryptorum.core.runs import RunType
from dsat.scryptorum.core.experiment import Experiment
from test.conftest import verify_jsonl_file, assert_log_entry_structure


class TestRunTypeConfiguration:
    """Test run type configuration for decorators."""

    def test_set_get_default_run_type_enum(self):
        """Test setting run type with enum."""
        set_default_run_type(RunType.MILESTONE)
        assert get_default_run_type() == RunType.MILESTONE

        set_default_run_type(RunType.TRIAL)
        assert get_default_run_type() == RunType.TRIAL

    def test_set_get_default_run_type_string(self):
        """Test setting run type with string."""
        set_default_run_type("milestone")
        assert get_default_run_type() == RunType.MILESTONE

        set_default_run_type("trial")
        assert get_default_run_type() == RunType.TRIAL

    def test_default_run_type_fallback(self):
        """Test that default run type falls back to TRIAL."""
        # Reset to None
        import dsat.scryptorum.core.decorators as decorators

        decorators._default_run_type = None

        assert get_default_run_type() == RunType.TRIAL


class TestRunContext:
    """Test run context management."""

    def test_set_get_current_run(self, trial_run):
        """Test setting and getting current run."""
        assert get_current_run() is None

        set_current_run(trial_run)
        assert get_current_run() == trial_run

        set_current_run(None)
        assert get_current_run() is None

    def test_run_context_manager(self, trial_run):
        """Test run context manager."""
        assert get_current_run() is None

        with run_context(trial_run) as run:
            assert run == trial_run
            assert get_current_run() == trial_run

        assert get_current_run() is None

    def test_run_context_nested(self, trial_run, milestone_run):
        """Test nested run contexts."""
        set_current_run(trial_run)

        with run_context(milestone_run):
            assert get_current_run() == milestone_run

        # Should restore previous context
        assert get_current_run() == trial_run

        set_current_run(None)


class TestExperimentDecorator:
    """Test @experiment decorator."""

    def test_experiment_decorator_basic(self, test_project_root: Path):
        """Test basic experiment decorator functionality."""
        set_default_run_type(RunType.TRIAL)

        @experiment(name="test_experiment")
        def run_test():
            return "success"

        # Mock project_root to avoid current directory issues
        with patch.dict(
            "os.environ", {"SCRYPTORUM_PROJECT_ROOT": str(test_project_root)}
        ):
            result = run_test(project_root=test_project_root)

        assert result == "success"

        # Verify experiment was created
        exp_path = test_project_root / "experiments" / "test_experiment"
        assert exp_path.exists()

        # Verify run was created and finished
        trial_run_path = exp_path / "runs" / "trial_run"
        assert trial_run_path.exists()

        log_entries = verify_jsonl_file(trial_run_path / "run.jsonl")
        start_events = [e for e in log_entries if e["event_type"] == "run_started"]
        finish_events = [e for e in log_entries if e["event_type"] == "run_finished"]

        assert len(start_events) == 1
        assert len(finish_events) == 1

    def test_experiment_decorator_with_milestone(self, test_project_root: Path):
        """Test experiment decorator with milestone run type."""
        set_default_run_type(RunType.MILESTONE)

        @experiment(name="milestone_experiment")
        def run_milestone():
            return "milestone_success"

        result = run_milestone(project_root=test_project_root)
        assert result == "milestone_success"

        # Verify milestone run was created
        exp_path = test_project_root / "experiments" / "milestone_experiment"
        runs_path = exp_path / "runs"

        # Should have a versioned run directory (not trial_run)
        run_dirs = [
            d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith("ms-")
        ]
        assert len(run_dirs) == 1

        run_dir = run_dirs[0]
        assert (run_dir / "artifacts").exists()
        assert (run_dir / "code_snapshot").exists()

    def test_experiment_decorator_default_name(self, test_project_root: Path):
        """Test experiment decorator with default name (function name)."""
        set_default_run_type(RunType.TRIAL)

        @experiment()
        def my_test_function():
            return "named_from_function"

        result = my_test_function(project_root=test_project_root)
        assert result == "named_from_function"

        # Verify experiment named after function
        exp_path = test_project_root / "experiments" / "my_test_function"
        assert exp_path.exists()

    def test_experiment_decorator_exception_handling(self, test_project_root: Path):
        """Test experiment decorator handles exceptions properly."""
        set_default_run_type(RunType.TRIAL)

        @experiment(name="failing_experiment")
        def failing_function():
            raise ValueError("Test exception")

        with pytest.raises(ValueError, match="Test exception"):
            failing_function(project_root=test_project_root)

        # Verify run was created and finished despite exception
        exp_path = test_project_root / "experiments" / "failing_experiment"
        trial_run_path = exp_path / "runs" / "trial_run"

        log_entries = verify_jsonl_file(trial_run_path / "run.jsonl")
        error_events = [e for e in log_entries if e["event_type"] == "experiment_error"]
        finish_events = [e for e in log_entries if e["event_type"] == "run_finished"]

        assert len(error_events) == 1
        assert len(finish_events) == 1
        assert "Test exception" in error_events[0]["error"]

    def test_experiment_decorator_existing_context(self, trial_run):
        """Test experiment decorator with existing run context."""

        @experiment(name="nested_experiment")
        def nested_function():
            return "nested_result"

        # Set existing context
        with run_context(trial_run):
            result = nested_function()

        assert result == "nested_result"

        # Should not create new experiment when context exists
        # This is tested by checking that the function runs without project_root


class TestMetricDecorator:
    """Test @metric decorator."""

    def test_metric_decorator_basic(self, trial_run):
        """Test basic metric decorator functionality."""

        @metric(name="test_accuracy", metric_type="accuracy")
        def calculate_accuracy():
            return 0.85

        with run_context(trial_run):
            result = calculate_accuracy()

        assert result == 0.85

        # Verify metric was logged
        metric_entries = verify_jsonl_file(trial_run.metrics_file, expected_entries=1)
        entry = metric_entries[0]

        assert entry["name"] == "test_accuracy"
        assert entry["value"] == 0.85
        assert entry["type"] == "accuracy"

    def test_metric_decorator_default_name(self, trial_run):
        """Test metric decorator with default name."""

        @metric(metric_type="f1")
        def calculate_f1_score():
            return 0.82

        with run_context(trial_run):
            result = calculate_f1_score()

        metric_entries = verify_jsonl_file(trial_run.metrics_file, expected_entries=1)
        assert metric_entries[0]["name"] == "calculate_f1_score"

    def test_metric_decorator_dict_return(self, trial_run):
        """Test metric decorator with dict return value."""

        @metric(name="complex_metric", metric_type="custom")
        def calculate_complex_metric():
            return {"value": 0.75, "dataset_size": 1000, "error_count": 25}

        with run_context(trial_run):
            result = calculate_complex_metric()

        assert result == {"value": 0.75, "dataset_size": 1000, "error_count": 25}

        metric_entries = verify_jsonl_file(trial_run.metrics_file, expected_entries=1)
        entry = metric_entries[0]

        assert entry["name"] == "complex_metric"
        assert entry["value"] == 0.75
        assert entry["dataset_size"] == 1000
        assert entry["error_count"] == 25

    def test_metric_decorator_tuple_return(self, trial_run):
        """Test metric decorator with tuple return value."""

        @metric(name="tuple_metric", metric_type="precision")
        def calculate_precision():
            return 0.88, {"confidence": 0.95, "samples": 500}

        with run_context(trial_run):
            result = calculate_precision()

        assert result == (0.88, {"confidence": 0.95, "samples": 500})

        metric_entries = verify_jsonl_file(trial_run.metrics_file, expected_entries=1)
        entry = metric_entries[0]

        assert entry["value"] == 0.88
        assert entry["confidence"] == 0.95
        assert entry["samples"] == 500

    def test_metric_decorator_no_context(self):
        """Test metric decorator without run context."""

        @metric(name="no_context_metric")
        def calculate_metric():
            return 0.5

        # Should work without context, just not log anything
        result = calculate_metric()
        assert result == 0.5


class TestTimerDecorator:
    """Test @timer decorator."""

    def test_timer_decorator_basic(self, trial_run):
        """Test basic timer decorator functionality."""

        @timer(name="test_operation")
        def slow_operation():
            time.sleep(0.01)  # 10ms
            return "completed"

        with run_context(trial_run):
            result = slow_operation()

        assert result == "completed"

        # Verify timing was logged
        timing_entries = verify_jsonl_file(trial_run.timings_file, expected_entries=1)
        entry = timing_entries[0]

        assert entry["operation"] == "test_operation"
        assert entry["duration_ms"] >= 10  # Should be at least 10ms
        assert entry["duration_ms"] < 1000  # Should be reasonable

    def test_timer_decorator_default_name(self, trial_run):
        """Test timer decorator with default name."""

        @timer()
        def my_timed_function():
            time.sleep(0.005)
            return "done"

        with run_context(trial_run):
            result = my_timed_function()

        timing_entries = verify_jsonl_file(trial_run.timings_file, expected_entries=1)
        assert timing_entries[0]["operation"] == "my_timed_function"

    def test_timer_decorator_exception_handling(self, trial_run):
        """Test timer decorator logs timing even on exception."""

        @timer(name="failing_operation")
        def failing_operation():
            time.sleep(0.005)
            raise RuntimeError("Operation failed")

        with run_context(trial_run):
            with pytest.raises(RuntimeError):
                failing_operation()

        # Timing should still be logged
        timing_entries = verify_jsonl_file(trial_run.timings_file, expected_entries=1)
        assert timing_entries[0]["operation"] == "failing_operation"

    def test_timer_decorator_no_context(self):
        """Test timer decorator without run context."""

        @timer(name="no_context_timer")
        def operation():
            return "result"

        # Should work without context, just not log timing
        result = operation()
        assert result == "result"


class TestLLMCallDecorator:
    """Test @llm_call decorator."""

    def test_llm_call_decorator_basic(self, trial_run):
        """Test basic LLM call decorator functionality."""

        @llm_call(model="gpt-4")
        def call_llm(prompt):
            # Simulate LLM call
            time.sleep(0.01)
            return f"Response to: {prompt}"

        with run_context(trial_run):
            result = call_llm("Test prompt")

        assert result == "Response to: Test prompt"

        # Verify LLM call was logged
        log_entries = verify_jsonl_file(trial_run.log_file)
        llm_events = [e for e in log_entries if e["event_type"] == "llm_call"]
        assert len(llm_events) == 1

        event = llm_events[0]
        assert event["model"] == "gpt-4"
        assert event["input"] == "Test prompt"
        assert event["output"] == "Response to: Test prompt"
        assert event["duration_ms"] >= 10

    def test_llm_call_decorator_model_from_kwargs(self, trial_run):
        """Test LLM call decorator extracting model from kwargs."""

        @llm_call()
        def call_llm_with_model(**kwargs):
            return f"Called {kwargs['model']}"

        with run_context(trial_run):
            result = call_llm_with_model(model="claude-3", temperature=0.7)

        log_entries = verify_jsonl_file(trial_run.log_file)
        llm_events = [e for e in log_entries if e["event_type"] == "llm_call"]
        assert llm_events[0]["model"] == "claude-3"

    def test_llm_call_decorator_no_input_logging(self, trial_run):
        """Test LLM call decorator with input logging disabled."""

        @llm_call(model="gpt-3.5", log_input=False)
        def call_llm(prompt):
            return "Response"

        with run_context(trial_run):
            call_llm("Secret prompt")

        log_entries = verify_jsonl_file(trial_run.log_file)
        llm_events = [e for e in log_entries if e["event_type"] == "llm_call"]
        assert llm_events[0]["input"] is None

    def test_llm_call_decorator_no_output_logging(self, trial_run):
        """Test LLM call decorator with output logging disabled."""

        @llm_call(model="gpt-4", log_output=False)
        def call_llm(prompt):
            return "Secret response"

        with run_context(trial_run):
            call_llm("Test prompt")

        log_entries = verify_jsonl_file(trial_run.log_file)
        llm_events = [e for e in log_entries if e["event_type"] == "llm_call"]
        assert llm_events[0]["output"] is None

    def test_llm_call_decorator_exception_handling(self, trial_run):
        """Test LLM call decorator handles exceptions."""

        @llm_call(model="gpt-4")
        def failing_llm_call(prompt):
            time.sleep(0.005)
            raise ConnectionError("API unavailable")

        with run_context(trial_run):
            with pytest.raises(ConnectionError):
                failing_llm_call("Test prompt")

        log_entries = verify_jsonl_file(trial_run.log_file)
        llm_events = [e for e in log_entries if e["event_type"] == "llm_call"]
        assert len(llm_events) == 1

        event = llm_events[0]
        assert event["output"] is None
        assert "API unavailable" in event["error"]


class TestBatchProcessorDecorator:
    """Test @batch_processor decorator."""

    def test_batch_processor_basic(self, trial_run):
        """Test basic batch processor functionality."""

        @batch_processor(batch_size=3)
        def process_item(item):
            return item * 2

        items = [1, 2, 3, 4, 5, 6, 7]

        with run_context(trial_run):
            results = process_item(items)

        assert results == [2, 4, 6, 8, 10, 12, 14]

        # Verify batch processing events
        log_entries = verify_jsonl_file(trial_run.log_file)
        start_events = [
            e for e in log_entries if e["event_type"] == "batch_processing_started"
        ]
        complete_events = [
            e for e in log_entries if e["event_type"] == "batch_completed"
        ]
        finish_events = [
            e for e in log_entries if e["event_type"] == "batch_processing_finished"
        ]

        assert len(start_events) == 1
        assert len(complete_events) == 3  # 7 items / 3 batch_size = 3 batches
        assert len(finish_events) == 1

        start_event = start_events[0]
        assert start_event["batch_size"] == 3
        assert start_event["total_items"] == 7

    def test_batch_processor_exact_batches(self, trial_run):
        """Test batch processor with exact batch divisions."""

        @batch_processor(batch_size=2)
        def double_item(item):
            return item * 2

        items = [1, 2, 3, 4]  # Exactly 2 batches

        with run_context(trial_run):
            results = double_item(items)

        assert results == [2, 4, 6, 8]

        log_entries = verify_jsonl_file(trial_run.log_file)
        complete_events = [
            e for e in log_entries if e["event_type"] == "batch_completed"
        ]
        assert len(complete_events) == 2


class TestTimeOperationContext:
    """Test time_operation context manager."""

    def test_time_operation_with_context(self, trial_run):
        """Test time_operation context manager with run context."""
        with run_context(trial_run):
            with time_operation("test_context_operation"):
                time.sleep(0.01)

        timing_entries = verify_jsonl_file(trial_run.timings_file, expected_entries=1)
        assert timing_entries[0]["operation"] == "test_context_operation"

    def test_time_operation_without_context(self):
        """Test time_operation context manager without run context."""
        # Should not raise exception
        with time_operation("no_context_operation"):
            time.sleep(0.005)


class TestDecoratorMetadata:
    """Test decorator metadata functionality."""

    def test_experiment_decorator_metadata(self):
        """Test that experiment decorator adds metadata to function."""

        @experiment(name="metadata_test")
        def test_function():
            pass

        assert hasattr(test_function, "_scryptorum_experiment")
        assert test_function._scryptorum_experiment is True
        assert test_function._experiment_name == "metadata_test"

    def test_metric_decorator_metadata(self):
        """Test that metric decorator adds metadata to function."""

        @metric(name="test_metric", metric_type="accuracy")
        def metric_function():
            pass

        assert hasattr(metric_function, "_scryptorum_metric")
        assert metric_function._scryptorum_metric is True
        assert metric_function._metric_name == "test_metric"
        assert metric_function._metric_type == "accuracy"

    def test_timer_decorator_metadata(self):
        """Test that timer decorator adds metadata to function."""

        @timer(name="test_timer")
        def timer_function():
            pass

        assert hasattr(timer_function, "_scryptorum_timer")
        assert timer_function._scryptorum_timer is True
        assert timer_function._timer_name == "test_timer"

    def test_llm_call_decorator_metadata(self):
        """Test that LLM call decorator adds metadata to function."""

        @llm_call(model="gpt-4", log_input=False)
        def llm_function():
            pass

        assert hasattr(llm_function, "_scryptorum_llm_call")
        assert llm_function._scryptorum_llm_call is True
        assert llm_function._model == "gpt-4"
        assert llm_function._log_input is False
