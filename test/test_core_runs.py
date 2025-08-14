"""
Tests for core run functionality.
"""

import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from dsat.scryptorum.core.runs import Run, RunType, TimerContext
from test.conftest import (
    verify_jsonl_file,
    verify_json_file,
    assert_log_entry_structure,
)


class TestRunCreation:
    """Test run creation and basic functionality."""

    def test_trial_run_creation(self, trial_run: Run):
        """Test trial run creation and initial state."""
        assert trial_run.run_type == RunType.TRIAL
        assert trial_run.run_id == "trial_run"
        assert trial_run.start_time is not None
        assert trial_run.end_time is None

        # Verify directory structure
        assert trial_run.run_dir.exists()
        assert trial_run.run_dir.name == "trial_run"

        # Verify log files exist
        assert trial_run.log_file.exists()
        assert trial_run.metrics_file.exists()
        assert trial_run.timings_file.exists()

        # Trial runs should not have artifact directories
        assert (
            not hasattr(trial_run, "artifacts_dir")
            or not trial_run.artifacts_dir.exists()
        )
        assert (
            not hasattr(trial_run, "code_snapshot_dir")
            or not trial_run.code_snapshot_dir.exists()
        )

    def test_milestone_run_creation(self, milestone_run: Run):
        """Test milestone run creation and initial state."""
        assert milestone_run.run_type == RunType.MILESTONE
        assert milestone_run.run_id.startswith("ms-")
        assert milestone_run.start_time is not None
        assert milestone_run.end_time is None

        # Verify directory structure
        assert milestone_run.run_dir.exists()
        assert milestone_run.run_dir.name.startswith("ms-")

        # Verify log files exist
        assert milestone_run.log_file.exists()
        assert milestone_run.metrics_file.exists()
        assert milestone_run.timings_file.exists()

        # Milestone runs should have artifact directories
        assert milestone_run.artifacts_dir.exists()
        assert milestone_run.code_snapshot_dir.exists()

    def test_run_initialization_logging(self, trial_run: Run):
        """Test that run initialization creates proper log entries."""
        # Check run started event
        log_entries = verify_jsonl_file(trial_run.log_file, expected_entries=1)
        start_entry = log_entries[0]

        assert_log_entry_structure(
            start_entry, "run_started", ["run_id", "run_type", "start_time"]
        )
        assert start_entry["run_id"] == trial_run.run_id
        assert start_entry["run_type"] == trial_run.run_type.value


class TestRunLogging:
    """Test run logging functionality."""

    def test_log_metric(self, trial_run: Run, sample_data):
        """Test metric logging."""
        metrics = sample_data["metrics"]

        for metric in metrics:
            trial_run.log_metric(metric["name"], metric["value"], metric["type"])

        # Verify metrics file
        metric_entries = verify_jsonl_file(
            trial_run.metrics_file, expected_entries=len(metrics)
        )

        for i, entry in enumerate(metric_entries):
            expected = metrics[i]
            assert entry["name"] == expected["name"]
            assert entry["value"] == expected["value"]
            assert entry["type"] == expected["type"]
            assert "timestamp" in entry
            assert "run_id" in entry

        # Verify events logged to main log
        log_entries = verify_jsonl_file(trial_run.log_file)
        metric_events = [e for e in log_entries if e["event_type"] == "metric_logged"]
        assert len(metric_events) == len(metrics)

    def test_log_timing(self, trial_run: Run, sample_data):
        """Test timing logging."""
        timings = sample_data["timings"]

        for timing in timings:
            trial_run.log_timing(timing["operation"], timing["duration_ms"])

        # Verify timings file
        timing_entries = verify_jsonl_file(
            trial_run.timings_file, expected_entries=len(timings)
        )

        for i, entry in enumerate(timing_entries):
            expected = timings[i]
            assert entry["operation"] == expected["operation"]
            assert entry["duration_ms"] == expected["duration_ms"]
            assert "timestamp" in entry
            assert "run_id" in entry

        # Verify events logged to main log
        log_entries = verify_jsonl_file(trial_run.log_file)
        timing_events = [e for e in log_entries if e["event_type"] == "timing_logged"]
        assert len(timing_events) == len(timings)

    def test_log_llm_call(self, trial_run: Run, sample_data):
        """Test LLM call logging."""
        llm_calls = sample_data["llm_calls"]

        for call in llm_calls:
            trial_run.log_llm_call(
                call["model"], call["input"], call["output"], call["duration_ms"]
            )

        # Verify LLM calls logged to main log
        log_entries = verify_jsonl_file(trial_run.log_file)
        llm_events = [e for e in log_entries if e["event_type"] == "llm_call"]
        assert len(llm_events) == len(llm_calls)

        for i, event in enumerate(llm_events):
            expected = llm_calls[i]
            assert event["model"] == expected["model"]
            assert event["input"] == expected["input"]
            assert event["output"] == expected["output"]
            assert event["duration_ms"] == expected["duration_ms"]

    def test_log_metric_with_metadata(self, trial_run: Run):
        """Test metric logging with additional metadata."""
        trial_run.log_metric(
            "accuracy",
            0.85,
            "classification",
            dataset_size=1000,
            error_count=50,
            model_version="v1",
        )

        metric_entries = verify_jsonl_file(trial_run.metrics_file, expected_entries=1)
        entry = metric_entries[0]

        assert entry["dataset_size"] == 1000
        assert entry["error_count"] == 50
        assert entry["model_version"] == "v1"

    def test_log_timing_with_metadata(self, trial_run: Run):
        """Test timing logging with additional metadata."""
        trial_run.log_timing(
            "model_inference", 1250.5, batch_size=32, model_type="transformer"
        )

        timing_entries = verify_jsonl_file(trial_run.timings_file, expected_entries=1)
        entry = timing_entries[0]

        assert entry["batch_size"] == 32
        assert entry["model_type"] == "transformer"


class TestRunArtifacts:
    """Test run artifact handling."""

    def test_trial_run_preserve_artifacts(self, trial_run: Run, sample_data):
        """Test that trial runs only log artifact metadata."""
        artifacts = sample_data["artifacts"]
        trial_run.preserve_artifacts(artifacts)

        # Verify only metadata is logged, no files created
        log_entries = verify_jsonl_file(trial_run.log_file)
        artifact_events = [
            e for e in log_entries if e["event_type"] == "artifacts_logged"
        ]
        assert len(artifact_events) == 1

        event = artifact_events[0]
        assert event["artifact_count"] == len(artifacts)
        assert set(event["artifact_types"]) == set(artifacts.keys())

        # No artifact files should be created for trial runs
        if hasattr(trial_run, "artifacts_dir"):
            assert (
                not trial_run.artifacts_dir.exists()
                or len(list(trial_run.artifacts_dir.glob("*"))) == 0
            )

    def test_milestone_run_preserve_artifacts(self, milestone_run: Run, sample_data):
        """Test that milestone runs save artifacts to disk."""
        artifacts = sample_data["artifacts"]
        milestone_run.preserve_artifacts(artifacts)

        # Verify artifacts are saved to disk
        assert milestone_run.artifacts_dir.exists()
        artifact_files = list(milestone_run.artifacts_dir.glob("*.json"))
        assert len(artifact_files) == len(artifacts)

        # Verify artifact contents
        for artifact_name, artifact_data in artifacts.items():
            artifact_file = milestone_run.artifacts_dir / f"{artifact_name}.json"
            saved_data = verify_json_file(artifact_file)
            assert saved_data == artifact_data

        # Verify preservation event logged
        log_entries = verify_jsonl_file(milestone_run.log_file)
        preservation_events = [
            e for e in log_entries if e["event_type"] == "artifacts_preserved"
        ]
        assert len(preservation_events) == 1

        event = preservation_events[0]
        assert event["artifact_count"] == len(artifacts)
        assert len(event["preserved_artifacts"]) == len(artifacts)

    def test_milestone_run_code_snapshot(self, milestone_run: Run, temp_dir: Path):
        """Test code snapshot functionality for milestone runs."""
        # Create some test files
        test_file1 = temp_dir / "test_script.py"
        test_file2 = temp_dir / "config.json"
        test_dir = temp_dir / "utils"
        test_dir.mkdir()
        test_file3 = test_dir / "helpers.py"

        test_file1.write_text("print('Hello, world!')")
        test_file2.write_text('{"key": "value"}')
        test_file3.write_text("def helper(): pass")

        # Create snapshot
        milestone_run.snapshot_code([test_file1, test_file2, test_dir])

        # Verify files are copied
        assert (milestone_run.code_snapshot_dir / "test_script.py").exists()
        assert (milestone_run.code_snapshot_dir / "config.json").exists()
        assert (milestone_run.code_snapshot_dir / "utils").exists()
        assert (milestone_run.code_snapshot_dir / "utils" / "helpers.py").exists()

        # Verify content is preserved
        copied_script = milestone_run.code_snapshot_dir / "test_script.py"
        assert copied_script.read_text() == "print('Hello, world!')"

        # Verify snapshot event logged
        log_entries = verify_jsonl_file(milestone_run.log_file)
        snapshot_events = [
            e for e in log_entries if e["event_type"] == "code_snapshot_created"
        ]
        assert len(snapshot_events) == 1

    def test_trial_run_code_snapshot_skipped(self, trial_run: Run, temp_dir: Path):
        """Test that trial runs skip code snapshots."""
        test_file = temp_dir / "test.py"
        test_file.write_text("print('test')")

        trial_run.snapshot_code([test_file])

        # Verify snapshot was skipped
        log_entries = verify_jsonl_file(trial_run.log_file)
        skip_events = [
            e for e in log_entries if e["event_type"] == "code_snapshot_skipped"
        ]
        assert len(skip_events) == 1
        assert skip_events[0]["reason"] == "trial_run_type"


class TestRunFinish:
    """Test run finishing functionality."""

    def test_run_finish(self, trial_run: Run):
        """Test run finishing updates state and logs."""
        # Initially not finished
        assert trial_run.end_time is None

        # Finish the run
        trial_run.finish()

        # Verify state updated
        assert trial_run.end_time is not None
        assert trial_run.end_time >= trial_run.start_time

        # Verify finish event logged
        log_entries = verify_jsonl_file(trial_run.log_file)
        finish_events = [e for e in log_entries if e["event_type"] == "run_finished"]
        assert len(finish_events) == 1

        event = finish_events[0]
        assert "end_time" in event
        assert "duration_seconds" in event
        assert event["duration_seconds"] >= 0


class TestTimerContext:
    """Test timer context manager."""

    def test_timer_context_basic(self, trial_run: Run):
        """Test basic timer context functionality."""
        operation_name = "test_operation"

        with TimerContext(trial_run, operation_name):
            time.sleep(0.01)  # Sleep for 10ms

        # Verify timing was logged
        timing_entries = verify_jsonl_file(trial_run.timings_file, expected_entries=1)
        entry = timing_entries[0]

        assert entry["operation"] == operation_name
        assert entry["duration_ms"] >= 10  # Should be at least 10ms
        assert entry["duration_ms"] < 1000  # Should be reasonable

    def test_timer_context_with_metadata(self, trial_run: Run):
        """Test timer context with metadata."""
        with TimerContext(
            trial_run, "complex_operation", batch_size=64, model_type="bert"
        ):
            time.sleep(0.005)

        timing_entries = verify_jsonl_file(trial_run.timings_file, expected_entries=1)
        entry = timing_entries[0]

        assert entry["batch_size"] == 64
        assert entry["model_type"] == "bert"

    def test_timer_context_exception_handling(self, trial_run: Run):
        """Test that timer context still logs timing on exception."""
        try:
            with TimerContext(trial_run, "failing_operation"):
                time.sleep(0.005)
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # Timing should still be logged
        timing_entries = verify_jsonl_file(trial_run.timings_file, expected_entries=1)
        assert timing_entries[0]["operation"] == "failing_operation"


class TestThreadSafety:
    """Test thread safety of run logging."""

    def test_concurrent_metric_logging(self, trial_run: Run):
        """Test that concurrent metric logging is thread-safe."""

        def log_metrics(start_num):
            for i in range(10):
                trial_run.log_metric(f"metric_{start_num}_{i}", 0.5 + i * 0.01)

        # Run concurrent logging
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(log_metrics, i) for i in range(4)]
            for future in futures:
                future.result()

        # Verify all metrics were logged
        metric_entries = verify_jsonl_file(trial_run.metrics_file, expected_entries=40)

        # Verify no corruption (all entries are valid JSON)
        assert len(metric_entries) == 40

        # Verify metric names are correct
        metric_names = {entry["name"] for entry in metric_entries}
        expected_names = {f"metric_{i}_{j}" for i in range(4) for j in range(10)}
        assert metric_names == expected_names

    def test_concurrent_timing_logging(self, trial_run: Run):
        """Test that concurrent timing logging is thread-safe."""

        def log_timings(start_num):
            for i in range(5):
                trial_run.log_timing(f"operation_{start_num}_{i}", 100.0 + i * 10)

        # Run concurrent logging
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(log_timings, i) for i in range(3)]
            for future in futures:
                future.result()

        # Verify all timings were logged
        timing_entries = verify_jsonl_file(trial_run.timings_file, expected_entries=15)
        assert len(timing_entries) == 15

    def test_concurrent_event_logging(self, trial_run: Run):
        """Test that concurrent event logging is thread-safe."""

        def log_events(thread_id):
            for i in range(5):
                trial_run.log_event(f"test_event_{thread_id}", {"iteration": i})

        # Run concurrent logging
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(log_events, i) for i in range(3)]
            for future in futures:
                future.result()

        # Verify all events were logged (plus the initial run_started event)
        log_entries = verify_jsonl_file(trial_run.log_file, expected_entries=16)

        # Count test events (excluding run_started)
        test_events = [
            e for e in log_entries if e["event_type"].startswith("test_event_")
        ]
        assert len(test_events) == 15
