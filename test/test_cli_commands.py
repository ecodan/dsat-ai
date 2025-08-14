"""
Tests for CLI commands functionality.
"""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from dsat.scryptorum.cli.commands import (
    create_project_command,
    create_experiment_command,
    list_experiments_command,
    list_runs_command,
    run_experiment_command,
    main,
)
from dsat.scryptorum.core.runs import RunType
from test.conftest import verify_json_file, verify_jsonl_file


class MockArgs:
    """Mock arguments object for testing CLI commands."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestCreateProjectCommand:
    """Test create-project CLI command."""

    def test_create_project_basic(self, temp_dir: Path):
        """Test basic project creation via CLI."""
        args = MockArgs(name="cli_test_project", parent_dir=str(temp_dir))

        # Capture stdout
        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            create_project_command(args)

        output = captured_output.getvalue()
        project_path = temp_dir / "cli_test_project"

        # Verify output message
        assert (
            f"Created scryptorum project 'cli_test_project' at {project_path}" in output
        )
        assert "Created example experiment" in output

        # Verify project structure
        assert project_path.exists()
        assert (project_path / "project.json").exists()
        assert (project_path / "experiments").exists()
        assert (project_path / "examples").exists()

        # Verify example file created
        example_file = project_path / "examples" / "example_experiment.py"
        assert example_file.exists()

        example_content = example_file.read_text()
        assert '@experiment(name="example_experiment")' in example_content
        assert "from dsat.scryptorum import" in example_content

    def test_create_project_error_handling(self, temp_dir: Path):
        """Test project creation error handling."""
        # Try to create project in non-existent directory with no permissions
        invalid_path = "/invalid/nonexistent/path"
        args = MockArgs(name="test_project", parent_dir=invalid_path)

        captured_output = StringIO()
        with (
            patch("sys.stderr", captured_output),
            pytest.raises(SystemExit) as exc_info,
        ):
            create_project_command(args)

        assert exc_info.value.code == 1
        assert "Error creating project" in captured_output.getvalue()


class TestCreateExperimentCommand:
    """Test create-experiment CLI command."""

    def test_create_experiment_basic(self, test_project_root: Path):
        """Test basic experiment creation via CLI."""
        args = MockArgs(
            name="cli_sentiment_analysis", project_root=str(test_project_root)
        )

        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            create_experiment_command(args)

        output = captured_output.getvalue()
        exp_path = test_project_root / "experiments" / "cli_sentiment_analysis"

        # Verify output message (handle macOS symlink differences)
        expected_path = str(exp_path.resolve()).replace("/private", "")
        actual_output = output.replace("/private", "")
        assert f"Created experiment 'cli_sentiment_analysis' in {expected_path}" in actual_output

        # Verify experiment structure
        assert exp_path.exists()
        assert (exp_path / "experiment.json").exists()
        assert (exp_path / "runs").exists()
        assert (exp_path / "data").exists()
        assert (exp_path / "config").exists()

    def test_create_experiment_nonexistent_project(self, temp_dir: Path):
        """Test experiment creation with nonexistent project."""
        nonexistent_project = temp_dir / "nonexistent"
        args = MockArgs(name="test_exp", project_root=str(nonexistent_project))

        captured_output = StringIO()
        with (
            patch("sys.stderr", captured_output),
            pytest.raises(SystemExit) as exc_info,
        ):
            create_experiment_command(args)

        assert exc_info.value.code == 1
        assert "Project root does not exist" in captured_output.getvalue()


class TestListExperimentsCommand:
    """Test list-experiments CLI command."""

    def test_list_experiments_empty(self, test_project_root: Path):
        """Test listing experiments when none exist."""
        args = MockArgs(project_root=str(test_project_root))

        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            list_experiments_command(args)

        output = captured_output.getvalue()
        assert f"Experiments in {test_project_root}:" in output

    def test_list_experiments_with_data(self, test_project_root: Path):
        """Test listing experiments with actual experiments."""
        from dsat.scryptorum.core.experiment import Experiment

        # Create some experiments
        exp1 = Experiment(test_project_root, "sentiment_analysis")
        exp2 = Experiment(test_project_root, "text_classification")

        # Create some runs in experiments
        run1 = exp1.create_run()
        run1.finish()
        run2 = exp2.create_run(RunType.MILESTONE)  # Use milestone to avoid reset
        run2.finish()
        run3 = exp2.create_run(RunType.MILESTONE)  # Use milestone to avoid reset
        run3.finish()

        args = MockArgs(project_root=str(test_project_root))

        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            list_experiments_command(args)

        output = captured_output.getvalue()
        assert "sentiment_analysis (1 runs)" in output
        assert "text_classification (2 runs)" in output

    def test_list_experiments_no_experiments_dir(self, temp_dir: Path):
        """Test listing experiments when experiments directory doesn't exist."""
        # Create a project directory without experiments subdirectory
        project_dir = temp_dir / "no_experiments_project"
        project_dir.mkdir()

        args = MockArgs(project_root=str(project_dir))

        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            list_experiments_command(args)

        output = captured_output.getvalue()
        assert "No experiments directory found" in output


class TestListRunsCommand:
    """Test list-runs CLI command."""

    def test_list_runs_empty(self, test_experiment):
        """Test listing runs when no runs exist."""
        args = MockArgs(
            experiment="test_experiment", project_root=str(test_experiment.project_root)
        )

        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            list_runs_command(args)

        output = captured_output.getvalue()
        assert "No runs found in experiment 'test_experiment'" in output

    def test_list_runs_with_data(self, test_experiment):
        """Test listing runs with actual runs."""
        from dsat.scryptorum.core.runs import RunType

        # Create some runs
        trial_run = test_experiment.create_run(RunType.TRIAL)
        milestone_run = test_experiment.create_run(RunType.MILESTONE)

        trial_run.finish()
        milestone_run.finish()

        args = MockArgs(
            experiment="test_experiment", project_root=str(test_experiment.project_root)
        )

        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            list_runs_command(args)

        output = captured_output.getvalue()
        assert "Runs in experiment 'test_experiment':" in output
        assert "trial_run (trial)" in output
        assert f"{milestone_run.run_id} (milestone)" in output


class TestRunExperimentCommand:
    """Test run experiment CLI command."""

    def test_run_experiment_script_trial(self, test_project_root: Path):
        """Test running experiment from script in trial mode."""
        # Create a test script
        script_content = """
from dsat.scryptorum import experiment, metric

@experiment(name="script_test")
def main():
    return calculate_result()

@metric(name="test_metric", metric_type="accuracy")
def calculate_result():
    return 0.95

if __name__ == "__main__":
    main()
"""
        script_path = test_project_root / "test_script.py"
        script_path.write_text(script_content)

        args = MockArgs(
            experiment="script_test",
            project_root=str(test_project_root),
            script=str(script_path),
            module=None,
            milestone=False,
            run_id=None,
        )

        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            run_experiment_command(args)

        output = captured_output.getvalue()
        assert "Executed experiment script" in output
        assert "Experiment completed with run type: trial" in output

        # Verify experiment was created and run
        exp_path = test_project_root / "experiments" / "script_test"
        trial_run_path = exp_path / "runs" / "trial_run"
        assert trial_run_path.exists()

        # Verify the metric was logged
        metrics_file = trial_run_path / "metrics.jsonl"
        metric_entries = verify_jsonl_file(metrics_file, expected_entries=1)
        assert metric_entries[0]["name"] == "test_metric"
        assert metric_entries[0]["value"] == 0.95

    def test_run_experiment_script_milestone(self, test_project_root: Path):
        """Test running experiment from script in milestone mode."""
        script_content = """
from dsat.scryptorum import experiment

@experiment(name="milestone_test")
def main():
    return "milestone_success"

if __name__ == "__main__":
    main()
"""
        script_path = test_project_root / "milestone_script.py"
        script_path.write_text(script_content)

        args = MockArgs(
            experiment="milestone_test",
            project_root=str(test_project_root),
            script=str(script_path),
            module=None,
            milestone=True,
            run_id="custom_milestone_v1",
        )

        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            run_experiment_command(args)

        output = captured_output.getvalue()
        assert "Experiment completed with run type: milestone" in output

        # Verify milestone run was created with custom ID
        exp_path = test_project_root / "experiments" / "milestone_test"
        milestone_run_path = exp_path / "runs" / "custom_milestone_v1"
        assert milestone_run_path.exists()
        assert (milestone_run_path / "artifacts").exists()
        assert (milestone_run_path / "code_snapshot").exists()

    def test_run_experiment_nonexistent_script(self, test_project_root: Path):
        """Test running experiment with nonexistent script."""
        args = MockArgs(
            experiment="test",
            project_root=str(test_project_root),
            script="nonexistent_script.py",
            module=None,
            milestone=False,
            run_id=None,
        )

        captured_output = StringIO()
        with (
            patch("sys.stderr", captured_output),
            pytest.raises(SystemExit) as exc_info,
        ):
            run_experiment_command(args)

        assert exc_info.value.code == 1
        assert "Script not found" in captured_output.getvalue()

    def test_run_experiment_no_script_or_module(self, test_project_root: Path):
        """Test running experiment without script or module."""
        args = MockArgs(
            experiment="test",
            project_root=str(test_project_root),
            script=None,
            module=None,
            milestone=False,
            run_id=None,
        )

        captured_output = StringIO()
        with (
            patch("sys.stderr", captured_output),
            pytest.raises(SystemExit) as exc_info,
        ):
            run_experiment_command(args)

        assert exc_info.value.code == 1
        assert "Must specify either --module or --script" in captured_output.getvalue()


class TestMainCLI:
    """Test main CLI entry point."""

    def test_main_no_command(self):
        """Test main CLI with no command shows help."""
        with (
            patch("sys.argv", ["scryptorum"]),
            patch("scryptorum.cli.commands.main") as mock_main,
            pytest.raises(SystemExit),
        ):

            # Create a mock args object that doesn't have 'func'
            mock_args = MagicMock()
            del mock_args.func  # Remove func attribute

            with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
                main()

    def test_main_create_project(self, temp_dir: Path):
        """Test main CLI with create-project command."""
        test_args = ["scryptorum", "create-project", "test_project", str(temp_dir)]

        with patch("sys.argv", test_args):
            main()

        # Verify project was created
        project_path = temp_dir / "test_project"
        assert project_path.exists()
        assert (project_path / "project.json").exists()

    def test_main_create_experiment(self, test_project_root: Path):
        """Test main CLI with create-experiment command."""
        test_args = [
            "scryptorum",
            "create-experiment",
            "main_test_experiment",
            "-p",
            str(test_project_root),
        ]

        with patch("sys.argv", test_args):
            main()

        # Verify experiment was created
        exp_path = test_project_root / "experiments" / "main_test_experiment"
        assert exp_path.exists()
        assert (exp_path / "experiment.json").exists()

    def test_main_list_experiments(self, test_project_root: Path):
        """Test main CLI with list-experiments command."""
        # Create an experiment first
        from dsat.scryptorum.core.experiment import Experiment

        exp = Experiment(test_project_root, "listed_experiment")

        test_args = ["scryptorum", "list-experiments", "-p", str(test_project_root)]

        captured_output = StringIO()
        with patch("sys.argv", test_args), patch("sys.stdout", captured_output):
            main()

        output = captured_output.getvalue()
        assert "listed_experiment" in output

    def test_main_run_script(self, test_project_root: Path):
        """Test main CLI with run command."""
        # Create a test script
        script_content = """
from dsat.scryptorum import experiment

@experiment(name="main_run_test")
def main():
    return "success"

if __name__ == "__main__":
    main()
"""
        script_path = test_project_root / "main_test_script.py"
        script_path.write_text(script_content)

        test_args = [
            "scryptorum",
            "run",
            "main_run_test",
            "-s",
            str(script_path),
            "-p",
            str(test_project_root),
        ]

        captured_output = StringIO()
        with patch("sys.argv", test_args), patch("sys.stdout", captured_output):
            main()

        output = captured_output.getvalue()
        assert "Executed experiment script" in output

        # Verify experiment was created
        exp_path = test_project_root / "experiments" / "main_run_test"
        assert exp_path.exists()
