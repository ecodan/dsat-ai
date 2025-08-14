"""
Tests for core experiment functionality.
"""

import json
import pytest
from pathlib import Path

from dsat.scryptorum.core.experiment import Experiment, create_project
from dsat.scryptorum.core.runs import RunType
from test.conftest import verify_json_file, verify_jsonl_file


class TestCreateProject:
    """Test project creation functionality."""

    def test_create_project_structure(self, temp_dir: Path):
        """Test that create_project creates the correct directory structure."""
        project_name = "test_project"
        project_root = create_project(project_name, temp_dir)

        # Verify project root exists
        assert project_root.exists()
        assert project_root.name == project_name

        # Verify required directories
        expected_dirs = ["experiments", "data", "models", "artifacts"]
        for dir_name in expected_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Missing directory: {dir_name}"
            assert dir_path.is_dir()

        # Verify project metadata file
        project_config = verify_json_file(project_root / "project.json")
        assert project_config["name"] == project_name
        assert project_config["framework"] == "scryptorum"
        assert project_config["version"] == "0.1.0"

    def test_create_project_with_custom_path(self, temp_dir: Path):
        """Test creating a project in a custom path."""
        custom_path = temp_dir / "custom" / "location"
        custom_path.mkdir(parents=True)

        project_root = create_project("my_project", custom_path)
        assert project_root == custom_path / "my_project"
        assert project_root.exists()


class TestExperiment:
    """Test Experiment class functionality."""

    def test_experiment_creation(self, test_project_root: Path):
        """Test experiment creation and directory structure."""
        experiment = Experiment(test_project_root, "sentiment_analysis")

        # Verify experiment path
        expected_path = test_project_root / "experiments" / "sentiment_analysis"
        assert experiment.experiment_path.resolve() == expected_path.resolve()
        assert experiment.experiment_name == "sentiment_analysis"

        # Verify experiment directories
        expected_dirs = ["runs", "data", "config"]
        for dir_name in expected_dirs:
            dir_path = experiment.experiment_path / dir_name
            assert dir_path.exists(), f"Missing experiment directory: {dir_name}"

        # Verify experiment metadata
        metadata = verify_json_file(experiment.experiment_path / "experiment.json")
        assert metadata["name"] == "sentiment_analysis"
        assert metadata["project_root"] == str(test_project_root)
        assert "created_at" in metadata

    def test_experiment_properties(self, test_experiment: Experiment):
        """Test experiment property accessors."""
        data_dir = test_experiment.data_dir
        config_dir = test_experiment.config_dir

        assert data_dir == test_experiment.experiment_path / "data"
        assert config_dir == test_experiment.experiment_path / "config"
        assert data_dir.exists()
        assert config_dir.exists()

    def test_create_trial_run(self, test_experiment: Experiment):
        """Test trial run creation."""
        run = test_experiment.create_run(RunType.TRIAL)

        assert run.run_type == RunType.TRIAL
        assert run.run_id == "trial_run"
        assert run.run_dir.exists()

        # Verify trial run structure
        assert (run.run_dir / "run.jsonl").exists()
        assert (run.run_dir / "metrics.jsonl").exists()
        assert (run.run_dir / "timings.jsonl").exists()

        # Trial runs should not have artifact directories
        assert not (run.run_dir / "artifacts").exists()
        assert not (run.run_dir / "code_snapshot").exists()

    def test_create_milestone_run(self, test_experiment: Experiment):
        """Test milestone run creation."""
        run = test_experiment.create_run(RunType.MILESTONE)

        assert run.run_type == RunType.MILESTONE
        assert run.run_id.startswith("ms-")
        assert run.run_dir.exists()

        # Verify milestone run structure
        assert (run.run_dir / "run.jsonl").exists()
        assert (run.run_dir / "metrics.jsonl").exists()
        assert (run.run_dir / "timings.jsonl").exists()
        assert (run.run_dir / "artifacts").exists()
        assert (run.run_dir / "code_snapshot").exists()

    def test_create_milestone_run_with_custom_id(self, test_experiment: Experiment):
        """Test milestone run creation with custom ID."""
        custom_id = "experiment_v1"
        run = test_experiment.create_run(RunType.MILESTONE, custom_id)

        assert run.run_id == custom_id
        assert run.run_dir.name == custom_id

    def test_trial_run_reset_behavior(self, test_experiment: Experiment):
        """Test that trial runs reset the directory on each creation."""
        # Create first trial run and add some data
        run1 = test_experiment.create_run(RunType.TRIAL)
        run1.log_metric("test_metric", 0.5)
        run1.finish()

        # Verify file exists
        metrics_file = run1.run_dir / "metrics.jsonl"
        assert metrics_file.exists()

        # Create second trial run - should reset directory
        run2 = test_experiment.create_run(RunType.TRIAL)

        # Directory should be reset (empty metrics file)
        metrics_entries = verify_jsonl_file(metrics_file, expected_entries=0)
        assert len(metrics_entries) == 0

    def test_list_runs_empty(self, test_experiment: Experiment):
        """Test listing runs when no runs exist."""
        runs = test_experiment.list_runs()
        assert runs == []

    def test_list_runs_with_data(self, test_experiment: Experiment):
        """Test listing runs with actual runs."""
        # Create multiple runs
        trial_run = test_experiment.create_run(RunType.TRIAL)
        milestone_run1 = test_experiment.create_run(RunType.MILESTONE)
        milestone_run2 = test_experiment.create_run(RunType.MILESTONE)

        # Finish runs to ensure proper metadata
        trial_run.finish()
        milestone_run1.finish()
        milestone_run2.finish()

        runs = test_experiment.list_runs()
        assert len(runs) == 3

        # Check that all runs are represented
        run_ids = {run["run_id"] for run in runs}
        expected_ids = {trial_run.run_id, milestone_run1.run_id, milestone_run2.run_id}
        assert run_ids == expected_ids

        # Check run types
        run_types = {run["run_type"] for run in runs}
        assert "trial" in run_types
        assert "milestone" in run_types

    def test_load_run_trial(self, test_experiment: Experiment):
        """Test loading an existing trial run."""
        # Create and finish a trial run
        original_run = test_experiment.create_run(RunType.TRIAL)
        original_run.log_metric("test_metric", 0.7)
        original_run.finish()

        # Load the run
        loaded_run = test_experiment.load_run("trial_run")
        assert loaded_run is not None
        assert loaded_run.run_type == RunType.TRIAL
        assert loaded_run.run_id == "trial_run"

    def test_load_run_milestone(self, test_experiment: Experiment):
        """Test loading an existing milestone run."""
        # Create and finish a milestone run
        original_run = test_experiment.create_run(RunType.MILESTONE)
        run_id = original_run.run_id
        original_run.log_metric("test_metric", 0.8)
        original_run.finish()

        # Load the run
        loaded_run = test_experiment.load_run(run_id)
        assert loaded_run is not None
        assert loaded_run.run_type == RunType.MILESTONE
        assert loaded_run.run_id == run_id

    def test_load_nonexistent_run(self, test_experiment: Experiment):
        """Test loading a run that doesn't exist."""
        loaded_run = test_experiment.load_run("nonexistent_run")
        assert loaded_run is None
