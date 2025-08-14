"""
Tests for execution runner functionality.
"""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from dsat.scryptorum.execution.runner import Runner, BaseRunnable
from dsat.scryptorum.core.experiment import Experiment
from dsat.scryptorum.core.runs import RunType
from test.conftest import verify_jsonl_file, verify_json_file


class TestBaseRunnable:
    """Test BaseRunnable base class."""

    def test_base_runnable_init(self, test_experiment, trial_run):
        """Test BaseRunnable initialization."""
        config = {"model": "gpt-4", "batch_size": 32}

        runnable = BaseRunnable(test_experiment, trial_run, config)

        assert runnable.experiment == test_experiment
        assert runnable.run == trial_run
        assert runnable.config == config

    def test_base_runnable_lifecycle_methods(self, test_experiment, trial_run):
        """Test BaseRunnable lifecycle methods."""
        runnable = BaseRunnable(test_experiment, trial_run, {})

        # These should not raise exceptions
        runnable.prepare()
        runnable.score()
        runnable.cleanup()

        # execute() should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            runnable.execute()


class TestRunner:
    """Test Runner class functionality."""

    def test_runner_init(self, test_project_root: Path):
        """Test Runner initialization."""
        runner = Runner(test_project_root)
        assert runner.project_root == test_project_root

    def test_run_function_basic(self, test_project_root: Path):
        """Test running a simple function as experiment."""
        runner = Runner(test_project_root)

        def simple_experiment():
            return "function_result"

        result = runner.run_function("function_test", simple_experiment, RunType.TRIAL)

        assert result == "function_result"

        # Verify experiment was created
        exp_path = test_project_root / "experiments" / "function_test"
        assert exp_path.exists()

        # Verify run was created and finished
        trial_run_path = exp_path / "runs" / "trial_run"
        assert trial_run_path.exists()

        log_entries = verify_jsonl_file(trial_run_path / "run.jsonl")
        finish_events = [e for e in log_entries if e["event_type"] == "run_finished"]
        assert len(finish_events) == 1

    def test_run_function_with_args(self, test_project_root: Path):
        """Test running function with arguments."""
        runner = Runner(test_project_root)

        def function_with_args(x, y, multiplier=1):
            return (x + y) * multiplier

        result = runner.run_function(
            "args_test",
            function_with_args,
            RunType.TRIAL,
            None,  # run_id
            5,
            3,  # positional args
            multiplier=2,  # keyword args
        )

        assert result == 16  # (5 + 3) * 2

    def test_run_function_exception_handling(self, test_project_root: Path):
        """Test function execution with exception."""
        runner = Runner(test_project_root)

        def failing_function():
            raise ValueError("Function failed")

        with pytest.raises(ValueError, match="Function failed"):
            runner.run_function("failing_test", failing_function)

        # Verify run was still created and finished
        exp_path = test_project_root / "experiments" / "failing_test"
        trial_run_path = exp_path / "runs" / "trial_run"

        log_entries = verify_jsonl_file(trial_run_path / "run.jsonl")
        error_events = [e for e in log_entries if e["event_type"] == "execution_error"]
        finish_events = [e for e in log_entries if e["event_type"] == "run_finished"]

        assert len(error_events) == 1
        assert len(finish_events) == 1
        assert "Function failed" in error_events[0]["error"]


class TestRunnableExecution:
    """Test runnable class execution."""

    def create_test_runnable_class(self):
        """Create a test runnable class for testing."""

        class TestRunnable(BaseRunnable):
            def __init__(self, experiment, run, config):
                super().__init__(experiment, run, config)
                self.prepare_called = False
                self.run_called = False
                self.score_called = False
                self.cleanup_called = False

            def prepare(self):
                self.prepare_called = True
                self.run.log_event("prepare_completed", {"status": "success"})

                # Log configuration
                model = self.config.get("model", "default")
                self.run.log_event("model_configured", {"model": model})

            def execute(self):
                self.run_called = True
                self.run.log_event("execution_started", {})

                # Simulate some work
                for i in range(3):
                    self.run.log_metric(f"iteration_{i}", 0.1 * i, "progress")
                    time.sleep(0.001)  # Small delay

                self.run.log_event("execution_completed", {"iterations": 3})

            def score(self):
                self.score_called = True

                # Calculate final metrics
                accuracy = 0.85
                self.run.log_metric("final_accuracy", accuracy, "accuracy")
                self.run.log_event("scoring_completed", {"accuracy": accuracy})

            def cleanup(self):
                self.cleanup_called = True
                self.run.log_event("cleanup_completed", {})

        return TestRunnable

    def test_run_experiment_with_runnable_class(self, test_project_root: Path):
        """Test running experiment with runnable class."""
        runner = Runner(test_project_root)
        TestRunnable = self.create_test_runnable_class()

        config = {"model": "gpt-4", "batch_size": 16}

        run = runner.run_experiment(
            "runnable_test",
            runnable_class=TestRunnable,
            run_type=RunType.TRIAL,
            config=config,
        )

        # Verify experiment structure
        exp_path = test_project_root / "experiments" / "runnable_test"
        assert exp_path.exists()

        # Verify all lifecycle methods were called
        log_entries = verify_jsonl_file(run.log_file)
        event_types = {e["event_type"] for e in log_entries}

        expected_events = {
            "run_started",
            "prepare_completed",
            "model_configured",
            "execution_started",
            "execution_completed",
            "scoring_completed",
            "cleanup_completed",
            "run_finished",
        }
        assert expected_events.issubset(event_types)

        # Verify metrics were logged
        metric_entries = verify_jsonl_file(run.metrics_file)
        metric_names = {e["name"] for e in metric_entries}

        expected_metrics = {
            "iteration_0",
            "iteration_1",
            "iteration_2",
            "final_accuracy",
        }
        assert expected_metrics.issubset(metric_names)

    def test_run_experiment_milestone_with_artifacts(self, test_project_root: Path):
        """Test milestone run with artifact preservation."""

        class ArtifactRunnable(BaseRunnable):
            def execute(self):
                # Create some artifacts
                artifacts = {
                    "model_weights": [0.1, 0.2, 0.3],
                    "predictions": [1, 0, 1, 1, 0],
                    "metadata": {"version": "1.0", "timestamp": "2024-01-01"},
                }
                self.run.preserve_artifacts(artifacts)

        runner = Runner(test_project_root)

        run = runner.run_experiment(
            "artifact_test", runnable_class=ArtifactRunnable, run_type=RunType.MILESTONE
        )

        # Verify artifacts were preserved
        assert run.artifacts_dir.exists()
        artifact_files = list(run.artifacts_dir.glob("*.json"))
        assert len(artifact_files) == 3

        # Verify artifact contents
        model_weights_file = run.artifacts_dir / "model_weights.json"
        weights_data = verify_json_file(model_weights_file)
        assert weights_data == [0.1, 0.2, 0.3]

    def test_runnable_execution_stage_exception(self, test_project_root: Path):
        """Test runnable execution with exception in stage."""

        class FailingRunnable(BaseRunnable):
            def prepare(self):
                self.run.log_event("prepare_started", {})

            def execute(self):
                raise RuntimeError("Execution failed")

            def score(self):
                self.run.log_event("score_called", {})

            def cleanup(self):
                self.run.log_event("cleanup_called", {})

        runner = Runner(test_project_root)

        with pytest.raises(RuntimeError, match="Execution failed"):
            runner.run_experiment(
                "failing_runnable_test", runnable_class=FailingRunnable
            )

        # Verify prepare was called but not score, and cleanup was still called
        exp_path = test_project_root / "experiments" / "failing_runnable_test"
        trial_run_path = exp_path / "runs" / "trial_run"

        log_entries = verify_jsonl_file(trial_run_path / "run.jsonl")
        event_types = [e["event_type"] for e in log_entries]

        assert "prepare_started" in event_types
        assert "execute_error" in event_types  # Error should be logged
        assert "score_called" not in event_types  # Should not reach score
        assert "cleanup_called" in event_types  # Cleanup should still run

    def test_runnable_cleanup_exception(self, test_project_root: Path):
        """Test that cleanup exceptions don't propagate."""

        class CleanupFailingRunnable(BaseRunnable):
            def execute(self):
                self.run.log_event("run_completed", {})

            def cleanup(self):
                raise Exception("Cleanup failed")

        runner = Runner(test_project_root)

        # Should not raise exception despite cleanup failure
        run = runner.run_experiment(
            "cleanup_failing_test", runnable_class=CleanupFailingRunnable
        )

        # Verify run completed successfully
        log_entries = verify_jsonl_file(run.log_file)
        event_types = [e["event_type"] for e in log_entries]
        assert "run_completed" in event_types
        assert "run_finished" in event_types


class TestModuleLoading:
    """Test dynamic module loading functionality."""

    def test_load_runnable_class_from_file(self, test_project_root: Path):
        """Test loading runnable class from file path."""
        # Create a module file with runnable class
        module_content = """
from dsat.scryptorum.execution.runner import BaseRunnable

class FileRunnable(BaseRunnable):
    def execute(self):
        self.run._log_event("file_runnable_executed", {})
        self.run.log_metric("file_metric", 0.9, "test")
"""
        module_file = test_project_root / "file_runnable.py"
        module_file.write_text(module_content)

        runner = Runner(test_project_root)

        run = runner.run_experiment(
            "file_module_test", runnable_module=str(module_file)
        )

        # Verify the class was loaded and executed
        log_entries = verify_jsonl_file(run.log_file)
        event_types = [e["event_type"] for e in log_entries]
        assert "file_runnable_executed" in event_types

        metric_entries = verify_jsonl_file(run.metrics_file)
        assert len(metric_entries) == 1
        assert metric_entries[0]["name"] == "file_metric"

    def test_load_runnable_class_invalid_file(self, test_project_root: Path):
        """Test loading from invalid file path."""
        runner = Runner(test_project_root)

        with pytest.raises(ImportError, match="Failed to load runnable"):
            runner.run_experiment(
                "invalid_file_test", runnable_module="nonexistent_file.py"
            )

    def test_load_runnable_class_no_runnable(self, test_project_root: Path):
        """Test loading from file with no runnable class."""
        # Create a module file without runnable class
        module_content = """
def some_function():
    pass

class NotARunnable:
    pass
"""
        module_file = test_project_root / "no_runnable.py"
        module_file.write_text(module_content)

        runner = Runner(test_project_root)

        with pytest.raises(ImportError, match="No runnable class found"):
            runner.run_experiment("no_runnable_test", runnable_module=str(module_file))

    def test_load_runnable_class_skips_base_runnable(self, test_project_root: Path):
        """Test that BaseRunnable itself is skipped in favor of concrete implementations."""
        # Create a module file with both BaseRunnable and concrete implementation
        module_content = """
from dsat.scryptorum.execution.runner import BaseRunnable

class MyRunnable(BaseRunnable):
    def execute(self):
        self.run._log_event("concrete_runnable_executed", {})
"""
        module_file = test_project_root / "concrete_runnable.py"
        module_file.write_text(module_content)

        runner = Runner(test_project_root)

        run = runner.run_experiment(
            "concrete_runnable_test", runnable_module=str(module_file)
        )

        # Verify the concrete class was loaded and executed (not BaseRunnable)
        log_entries = verify_jsonl_file(run.log_file)
        event_types = [e["event_type"] for e in log_entries]
        assert "concrete_runnable_executed" in event_types

    def test_python_path_auto_addition(self, test_project_root: Path):
        """Test that ./src directory is automatically added to Python path."""
        import sys
        from pathlib import Path
        
        # Create src directory structure
        src_dir = test_project_root / "src"
        package_dir = src_dir / "test_package"
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py
        (package_dir / "__init__.py").write_text("")
        
        # Create runnable module
        module_content = """
from dsat.scryptorum.execution.runner import BaseRunnable

class SrcRunnable(BaseRunnable):
    def execute(self):
        self.run._log_event("src_runnable_executed", {})
"""
        (package_dir / "runnable.py").write_text(module_content)
        
        # Change to test project root to simulate real usage
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(test_project_root)
            runner = Runner(test_project_root)

            run = runner.run_experiment(
                "src_path_test", runnable_module="test_package.runnable"
            )

            # Verify the module was loaded from src
            log_entries = verify_jsonl_file(run.log_file)
            event_types = [e["event_type"] for e in log_entries]
            assert "src_runnable_executed" in event_types
            
        finally:
            os.chdir(original_cwd)

    def test_run_experiment_no_runnable_warning(self, test_project_root: Path):
        """Test running experiment without runnable logs warning."""
        runner = Runner(test_project_root)

        run = runner.run_experiment("no_runnable_test")

        # Verify warning was logged
        log_entries = verify_jsonl_file(run.log_file)
        warning_events = [e for e in log_entries if e["event_type"] == "warning"]
        assert len(warning_events) == 1
        assert "No runnable specified" in warning_events[0]["message"]


class TestRunnerIntegration:
    """Test runner integration scenarios."""

    def test_full_experiment_lifecycle(self, test_project_root: Path):
        """Test complete experiment lifecycle with all features."""

        class CompleteRunnable(BaseRunnable):
            def prepare(self):
                # Setup model configuration
                model_config = self.config.get("model", {})
                self.run.log_event("model_setup", model_config)

                # Log initial timing
                self.run.log_timing("setup", 50.0)

            def execute(self):
                # Simulate LLM calls
                prompts = ["Analyze sentiment", "Classify text", "Generate summary"]

                for i, prompt in enumerate(prompts):
                    self.run.log_llm_call(
                        model="gpt-4",
                        input_data=prompt,
                        output_data=f"Response {i}",
                        duration_ms=200.0 + i * 50,
                    )

                # Log processing timing
                self.run.log_timing("llm_calls", 750.0)

            def score(self):
                # Calculate multiple metrics
                self.run.log_metric("accuracy", 0.85, "accuracy")
                self.run.log_metric("f1_score", 0.82, "f1")
                self.run.log_metric("precision", 0.88, "precision")
                self.run.log_metric("recall", 0.77, "recall")

                # Create artifacts
                results = {
                    "predictions": [1, 0, 1, 1, 0, 1, 0],
                    "probabilities": [0.9, 0.1, 0.8, 0.95, 0.2, 0.85, 0.15],
                    "ground_truth": [1, 0, 1, 1, 1, 1, 0],
                }
                self.run.preserve_artifacts({"evaluation_results": results})

        runner = Runner(test_project_root)

        config = {"model": {"name": "gpt-4", "temperature": 0.7}, "batch_size": 32}

        run = runner.run_experiment(
            "complete_test",
            runnable_class=CompleteRunnable,
            run_type=RunType.MILESTONE,
            run_id="complete_v1",
            config=config,
        )

        # Verify all logs were created correctly
        assert run.run_id == "complete_v1"

        # Check metrics
        metric_entries = verify_jsonl_file(run.metrics_file)
        metric_names = {e["name"] for e in metric_entries}
        expected_metrics = {"accuracy", "f1_score", "precision", "recall"}
        assert expected_metrics == metric_names

        # Check timings
        timing_entries = verify_jsonl_file(run.timings_file)
        timing_ops = {e["operation"] for e in timing_entries}
        expected_timings = {"setup", "llm_calls"}
        assert expected_timings == timing_ops

        # Check LLM calls
        log_entries = verify_jsonl_file(run.log_file)
        llm_events = [e for e in log_entries if e["event_type"] == "llm_call"]
        assert len(llm_events) == 3

        # Check artifacts (milestone run)
        artifacts_file = run.artifacts_dir / "evaluation_results.json"
        assert artifacts_file.exists()

        # Verify run completed successfully
        finish_events = [e for e in log_entries if e["event_type"] == "run_finished"]
        assert len(finish_events) == 1
