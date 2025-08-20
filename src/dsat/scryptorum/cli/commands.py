"""
Command-line interface for scryptorum project management.
"""

import argparse
import sys
import traceback
from pathlib import Path

from ..core.experiment import create_project, Experiment, resolve_project_root
from ..core.project_config import (
    ScryptorumConfig,
    find_scryptorum_project,
    resolve_experiments_dir,
)

# Removed logging_utils import - CLI commands use print for immediate feedback


def _create_sample_experiment_file(current_dir: Path, project_name: str) -> None:
    """Create a sample experiment file in the root directory."""
    sample_filename = "sample_experiment.py"
    sample_file_path = current_dir / sample_filename

    # Don't overwrite existing files
    if sample_file_path.exists():
        print(f"Sample file already exists: {sample_file_path}")
        return

    sample_content = f'''"""
Sample experiment for {project_name}.

This demonstrates both decorator and class-based approaches to running experiments.
"""

from dsat.scryptorum import experiment, metric, timer, llm_call
from dsat.scryptorum.execution.runner import BaseRunnable


@experiment(name="sample_experiment")
def run_sample_experiment():
    """Sample experiment using decorators."""
    
    # Your experiment logic here
    data = prepare_data()
    results = process_data(data)
    accuracy = evaluate_results(results)
    
    return accuracy


@timer("data_preparation")
def prepare_data():
    """Prepare experimental data."""
    # Simulate data preparation
    import time
    time.sleep(0.1)
    return list(range(100))


@timer("data_processing")
def process_data(data):
    """Process the data."""
    return [x * 2 for x in data]


@llm_call(model="gpt-4")
def call_llm(prompt: str) -> str:
    """Example LLM call (replace with actual implementation)."""
    # This would be your actual LLM call
    return f"Response to: {{prompt}}"


@metric(name="accuracy", metric_type="accuracy")
def evaluate_results(results):
    """Evaluate experiment results."""
    # Simulate evaluation
    return 0.95


class SampleExperimentRunnable(BaseRunnable):
    """Sample experiment using class-based approach."""
    
    def prepare(self):
        """Prepare experiment resources."""
        self.run.log_event("preparation_started", {{}})
        self.run.logger.info("Preparing experiment...")
        
    def execute(self):
        """Execute the main experiment."""
        self.run.log_event("execution_started", {{}})
        self.run.logger.info("Running experiment...")
        
    def score(self):
        """Evaluate experiment results."""
        self.run.log_event("scoring_started", {{}})
        self.run.logger.info("Scoring results...")
        
    def cleanup(self):
        """Clean up experiment resources."""
        self.run.log_event("cleanup_started", {{}})
        self.run.logger.info("Cleaning up...")


if __name__ == "__main__":
    # Option 1: Run using decorator
    print("Running sample experiment with decorators...")
    result = run_sample_experiment()
    print(f"Experiment completed with accuracy: {{result}}")
    
    # Option 2: Run using Runner (uses proper experiment logging)
    # from dsat.scryptorum.execution.runner import Runner
    # print("\\nRunning sample experiment with Runner...")
    # runner = Runner(".")
    # runner.run_experiment("sample_experiment", SampleExperimentRunnable)
'''

    with open(sample_file_path, "w") as f:
        f.write(sample_content)

    print(f"Created sample experiment file: {sample_file_path}")


def create_project_command(args) -> None:
    """Create a new scryptorum project."""
    try:
        project_path = create_project(args.name, getattr(args, "parent_dir", None))
        print(f"Created scryptorum project '{args.name}' at {project_path}")

        # Create initial example files
        example_dir = project_path / "examples"
        example_dir.mkdir(exist_ok=True)

        example_content = '''"""
Example scryptorum experiment using decorators.
"""

from dsat.scryptorum import experiment, metric, timer, llm_call


@experiment(name="example_experiment")
def run_example_experiment():
    """Simple example experiment."""
    
    # Your experiment logic here
    data = prepare_data()
    results = process_data(data)
    accuracy = evaluate_results(results)
    
    return accuracy


@timer("data_preparation")
def prepare_data():
    """Prepare experimental data."""
    # Simulate data preparation
    import time
    time.sleep(0.1)
    return list(range(100))


@timer("data_processing")
def process_data(data):
    """Process the data."""
    return [x * 2 for x in data]


@llm_call(model="gpt-4")
def call_llm(prompt: str) -> str:
    """Example LLM call (replace with actual implementation)."""
    # This would be your actual LLM call
    return f"Response to: {prompt}"


@metric(name="accuracy", metric_type="accuracy")
def evaluate_results(results):
    """Evaluate experiment results."""
    # Simulate evaluation
    return 0.95


if __name__ == "__main__":
    # Run the experiment
    result = run_example_experiment()
    print(f"Experiment completed with accuracy: {result}")
'''

        with open(example_dir / "example_experiment.py", "w") as f:
            f.write(example_content)

        print(f"Created example experiment at {example_dir / 'example_experiment.py'}")

    except Exception as e:
        print(f"Error creating project: {e}", file=sys.stderr)
        sys.exit(1)


def init_command(args) -> None:
    """Initialize scryptorum in an existing Python project."""
    try:
        current_dir = Path.cwd()
        config = ScryptorumConfig()

        if config.exists():
            print(f"Scryptorum already initialized in {current_dir}")
            return

        # Resolve experiments directory
        experiments_dir = resolve_experiments_dir(
            getattr(args, "experiments_dir", None)
        )
        project_name = getattr(args, "project_name", None) or current_dir.name

        # Create the scryptorum project directory structure
        project_root = experiments_dir / project_name
        if not project_root.exists():
            project_root_created = create_project(project_name, experiments_dir)
            print(f"Created scryptorum project at {project_root_created}")
        else:
            print(f"Using existing scryptorum project at {project_root}")

        # Create .scryptorum config file
        config.create(experiments_dir, project_name)
        print(f"Initialized scryptorum in {current_dir}")
        print(f"Project: {project_name}")
        print(f"Experiments directory: {experiments_dir}")
        print(f"Config file: {config.config_path}")

        # Create sample experiment file if requested
        if getattr(args, "sample", False):
            _create_sample_experiment_file(current_dir, project_name)

    except Exception as e:
        print(f"Error initializing scryptorum: {e}", file=sys.stderr)
        sys.exit(1)


def create_experiment_command(args) -> None:
    """Create a new experiment in an existing project."""
    try:
        # Try to auto-detect scryptorum project first
        auto_project_root = find_scryptorum_project()

        if auto_project_root:
            project_root = auto_project_root
            print(f"Using auto-detected scryptorum project: {project_root}")
        elif hasattr(args, "project_name") and args.project_name:
            project_root = resolve_project_root(
                args.project_name, getattr(args, "parent_dir", None)
            )
        elif hasattr(args, "project_root") and args.project_root:
            project_root = Path(args.project_root)
        else:
            print(
                "No scryptorum project found. Run 'scryptorum init' first or specify project location.",
                file=sys.stderr,
            )
            sys.exit(1)

        if not project_root.exists():
            print(f"Project root does not exist: {project_root}", file=sys.stderr)
            sys.exit(1)

        experiment = Experiment(project_root, args.name)
        print(f"Created experiment '{args.name}' in {experiment.experiment_path}")

    except Exception as e:
        print(f"Error creating experiment: {e}", file=sys.stderr)
        sys.exit(1)


def list_experiments_command(args) -> None:
    """List experiments in a project."""
    try:
        # Try to auto-detect scryptorum project first
        auto_project_root = find_scryptorum_project()

        if auto_project_root:
            project_root = auto_project_root
        elif hasattr(args, "project_name") and args.project_name:
            project_root = resolve_project_root(
                args.project_name, getattr(args, "parent_dir", None)
            )
        elif hasattr(args, "project_root") and args.project_root:
            project_root = Path(args.project_root)
        else:
            print(
                "No scryptorum project found. Run 'scryptorum init' first or specify project location.",
                file=sys.stderr,
            )
            sys.exit(1)

        experiments_dir = project_root / "experiments"
        if not experiments_dir.exists():
            print(f"No experiments directory found at {experiments_dir}")
            return

        print(f"Experiments in {project_root}:")
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                experiment = Experiment(project_root, exp_dir.name)
                runs = experiment.list_runs()
                print(f"  {exp_dir.name} ({len(runs)} runs)")

    except Exception as e:
        print(f"Error listing experiments: {e}", file=sys.stderr)
        sys.exit(1)


def list_runs_command(args) -> None:
    """List runs in an experiment."""
    try:
        # Try to auto-detect scryptorum project first
        auto_project_root = find_scryptorum_project()

        if auto_project_root:
            project_root = auto_project_root
        elif hasattr(args, "project_name") and args.project_name:
            project_root = resolve_project_root(
                args.project_name, getattr(args, "parent_dir", None)
            )
        elif hasattr(args, "project_root") and args.project_root:
            project_root = Path(args.project_root)
        else:
            print(
                "No scryptorum project found. Run 'scryptorum init' first or specify project location.",
                file=sys.stderr,
            )
            sys.exit(1)

        experiment = Experiment(project_root, args.experiment)
        runs = experiment.list_runs()

        if not runs:
            print(f"No runs found in experiment '{args.experiment}'")
            return

        print(f"Runs in experiment '{args.experiment}':")
        for run in runs:
            print(f"  {run['run_id']} ({run['run_type']}) - {run['start_time']}")

    except Exception as e:
        print(f"Error listing runs: {e}", file=sys.stderr)
        sys.exit(1)


def sample_command(args) -> None:
    """Create a sample experiment file in the current directory."""
    try:
        current_dir = Path.cwd()

        # Determine project name from .scryptorum config if it exists
        config = ScryptorumConfig()
        if config.exists():
            project_name = config.get_project_name()
        else:
            project_name = current_dir.name

        _create_sample_experiment_file(current_dir, project_name)

    except Exception as e:
        print(f"Error creating sample file: {e}", file=sys.stderr)
        sys.exit(1)


def run_experiment_command(args) -> None:
    """Run an experiment with configurable run type."""
    try:
        from ..core.decorators import set_default_run_type
        from ..core.runs import RunType
        from ..execution.runner import Runner

        # Set the run type based on CLI flag
        run_type = RunType.MILESTONE if args.milestone else RunType.TRIAL
        set_default_run_type(run_type)

        # Try to auto-detect scryptorum project first
        auto_project_root = find_scryptorum_project()

        if auto_project_root:
            project_root = auto_project_root
        elif hasattr(args, "project_name") and args.project_name:
            project_root = resolve_project_root(
                args.project_name, getattr(args, "parent_dir", None)
            )
        elif hasattr(args, "project_root") and args.project_root:
            project_root = Path(args.project_root)
        else:
            print(
                "No scryptorum project found. Run 'scryptorum init' first or specify project location.",
                file=sys.stderr,
            )
            sys.exit(1)

        runner = Runner(project_root)

        if args.module:
            # Run using module/class
            runner.run_experiment(
                experiment_name=args.experiment,
                runnable_module=args.module,
                run_type=run_type,
                run_id=args.run_id,
            )
        elif args.script:
            # Run a script file through the Runner
            script_path = Path(args.script).resolve()
            if not script_path.exists():
                print(f"Script not found: {script_path}", file=sys.stderr)
                sys.exit(1)

            # Use the Runner to execute the script
            runner.run_experiment(
                experiment_name=args.experiment,
                runnable_module=str(script_path),
                run_type=run_type,
                run_id=args.run_id,
            )
        else:
            print("Must specify either --module or --script", file=sys.stderr)
            sys.exit(1)

        print(f"Experiment completed with run type: {run_type.value}")

    except Exception as e:
        print(f"Error running experiment: {e}", file=sys.stderr)
        print("Stack trace:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scryptorum - Modern LLM Experiment Framework"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create project command
    create_project_parser = subparsers.add_parser(
        "create-project", help="Create a new scryptorum project"
    )
    create_project_parser.add_argument("name", help="Project name")
    create_project_parser.add_argument(
        "--parent-dir",
        help="Parent directory for projects (overrides SCRYPTORUM_PROJECTS_DIR env var, default: current directory)",
    )
    create_project_parser.set_defaults(func=create_project_command)

    # Init command
    init_parser = subparsers.add_parser(
        "init", help="Initialize scryptorum in an existing Python project"
    )
    init_parser.add_argument(
        "--experiments-dir",
        help="Directory for scryptorum experiments (overrides SCRYPTORUM_EXPERIMENTS_DIR env var, default: ./experiments)",
    )
    init_parser.add_argument(
        "--project-name",
        help="Project name (default: current directory name)",
    )
    init_parser.add_argument(
        "--sample",
        action="store_true",
        help="Create a sample experiment Python file in the root directory",
    )
    init_parser.set_defaults(func=init_command)

    # Create experiment command
    create_exp_parser = subparsers.add_parser(
        "create-experiment", help="Create a new experiment"
    )
    create_exp_parser.add_argument("name", help="Experiment name")

    # Support both project name (with parent dir resolution) or direct project root
    # Now optional - will auto-detect if .scryptorum exists
    project_group = create_exp_parser.add_mutually_exclusive_group(required=False)
    project_group.add_argument(
        "--project-name", help="Project name (will resolve using parent directory)"
    )
    project_group.add_argument(
        "--project-root", help="Direct path to project root directory"
    )
    create_exp_parser.add_argument(
        "--parent-dir",
        help="Parent directory for projects (used with --project-name, overrides SCRYPTORUM_PROJECTS_DIR env var)",
    )
    create_exp_parser.set_defaults(func=create_experiment_command)

    # List experiments command
    list_exp_parser = subparsers.add_parser(
        "list-experiments", help="List experiments in project"
    )

    # Support both project name (with parent dir resolution) or direct project root
    # Now optional - will auto-detect if .scryptorum exists
    project_group = list_exp_parser.add_mutually_exclusive_group(required=False)
    project_group.add_argument(
        "--project-name", help="Project name (will resolve using parent directory)"
    )
    project_group.add_argument(
        "--project-root", help="Direct path to project root directory"
    )
    list_exp_parser.add_argument(
        "--parent-dir",
        help="Parent directory for projects (used with --project-name, overrides SCRYPTORUM_PROJECTS_DIR env var)",
    )
    list_exp_parser.set_defaults(func=list_experiments_command)

    # List runs command
    list_runs_parser = subparsers.add_parser(
        "list-runs", help="List runs in experiment"
    )
    list_runs_parser.add_argument("experiment", help="Experiment name")

    # Support both project name (with parent dir resolution) or direct project root
    # Now optional - will auto-detect if .scryptorum exists
    project_group = list_runs_parser.add_mutually_exclusive_group(required=False)
    project_group.add_argument(
        "--project-name", help="Project name (will resolve using parent directory)"
    )
    project_group.add_argument(
        "--project-root", help="Direct path to project root directory"
    )
    list_runs_parser.add_argument(
        "--parent-dir",
        help="Parent directory for projects (used with --project-name, overrides SCRYPTORUM_PROJECTS_DIR env var)",
    )
    list_runs_parser.set_defaults(func=list_runs_command)

    # Run experiment command
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("experiment", help="Experiment name")

    # Support both project name (with parent dir resolution) or direct project root
    # Now optional - will auto-detect if .scryptorum exists
    project_group = run_parser.add_mutually_exclusive_group(required=False)
    project_group.add_argument(
        "--project-name", help="Project name (will resolve using parent directory)"
    )
    project_group.add_argument(
        "--project-root", help="Direct path to project root directory"
    )
    run_parser.add_argument(
        "--parent-dir",
        help="Parent directory for projects (used with --project-name, overrides SCRYPTORUM_PROJECTS_DIR env var)",
    )
    run_parser.add_argument(
        "-m", "--module", help="Python module containing runnable class"
    )
    run_parser.add_argument("-s", "--script", help="Python script file to execute")
    run_parser.add_argument(
        "--milestone",
        action="store_true",
        help="Create milestone run (full versioning) instead of trial run",
    )
    run_parser.add_argument("--run-id", help="Custom run ID (milestone runs only)")
    run_parser.set_defaults(func=run_experiment_command)

    # Sample command
    sample_parser = subparsers.add_parser(
        "sample", help="Create a sample experiment file in the current directory"
    )
    sample_parser.set_defaults(func=sample_command)

    # Parse arguments and execute
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
