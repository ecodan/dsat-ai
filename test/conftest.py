"""
Pytest configuration and shared fixtures for scryptorum tests.
"""

import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Generator
from unittest.mock import Mock

import pytest

from dsat.scryptorum.core.experiment import Experiment, create_project
from dsat.scryptorum.core.runs import Run, RunType
from dsat.agents.agent import AgentConfig


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_project_root(temp_dir: Path) -> Path:
    """Create a test project structure."""
    project_root = create_project("test_project", temp_dir)
    return project_root


@pytest.fixture
def test_experiment(test_project_root: Path) -> Experiment:
    """Create a test experiment."""
    return Experiment(test_project_root, "test_experiment")


@pytest.fixture
def trial_run(test_experiment: Experiment) -> Run:
    """Create a trial run for testing."""
    return test_experiment.create_run(RunType.TRIAL)


@pytest.fixture
def milestone_run(test_experiment: Experiment) -> Run:
    """Create a milestone run for testing."""
    return test_experiment.create_run(RunType.MILESTONE)


@pytest.fixture
def temp_prompts_dir():
    """Create temporary directory for prompts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_config():
    """Return a sample AgentConfig for testing."""
    return AgentConfig(
        agent_name="test_agent",
        model_provider="test_provider",
        model_family="test_family",
        model_version="test-model-v1",
        prompt="test_prompt:v1",
        model_parameters={"temperature": 0.7},
        provider_auth={"api_key": "test-key"},
        prepend_datetime=False
    )


@pytest.fixture
def logger():
    """Return a mock logger."""
    return Mock(spec=logging.Logger)


@pytest.fixture
def anthropic_config():
    """Return Anthropic agent config."""
    return AgentConfig(
        agent_name="claude_agent",
        model_provider="anthropic",
        model_family="claude",
        model_version="claude-3-5-haiku-latest",
        prompt="assistant:v1",
        provider_auth={"api_key": "sk-test-key"}
    )


@pytest.fixture
def google_config():
    """Return Google Vertex AI agent config."""
    return AgentConfig(
        agent_name="vertex_agent",
        model_provider="google",
        model_family="gemini",
        model_version="gemini-2.0-flash",
        prompt="assistant:v1",
        provider_auth={"project_id": "test-project", "location": "us-central1"}
    )


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "metrics": [
            {"name": "accuracy", "value": 0.85, "type": "accuracy"},
            {"name": "f1_score", "value": 0.82, "type": "f1"},
            {"name": "loss", "value": 0.23, "type": "loss"},
        ],
        "timings": [
            {"operation": "data_loading", "duration_ms": 1250.5},
            {"operation": "model_inference", "duration_ms": 890.2},
            {"operation": "evaluation", "duration_ms": 150.0},
        ],
        "llm_calls": [
            {
                "model": "gpt-4",
                "input": "What is the sentiment of: Great product!",
                "output": "positive",
                "duration_ms": 1200.0,
            },
            {
                "model": "gpt-3.5-turbo",
                "input": "Analyze this text: Terrible service",
                "output": "negative",
                "duration_ms": 800.0,
            },
        ],
        "artifacts": {
            "model_predictions": [0.9, 0.1, 0.8, 0.2],
            "ground_truth": [1, 0, 1, 0],
            "config": {"batch_size": 32, "learning_rate": 0.001},
        },
    }


def verify_jsonl_file(file_path: Path, expected_entries: int = None) -> list:
    """Verify JSONL file exists and return parsed entries."""
    assert file_path.exists(), f"JSONL file does not exist: {file_path}"

    entries = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if expected_entries is not None:
        assert (
            len(entries) == expected_entries
        ), f"Expected {expected_entries} entries, got {len(entries)}"

    return entries


def verify_json_file(file_path: Path) -> dict:
    """Verify JSON file exists and return parsed content."""
    assert file_path.exists(), f"JSON file does not exist: {file_path}"

    with open(file_path, "r") as f:
        return json.load(f)


def assert_log_entry_structure(
    entry: dict, event_type: str, required_fields: list = None
):
    """Assert that a log entry has the correct structure."""
    assert "timestamp" in entry
    assert "event_type" in entry
    assert entry["event_type"] == event_type

    if required_fields:
        for field in required_fields:
            assert field in entry, f"Missing required field: {field}"
