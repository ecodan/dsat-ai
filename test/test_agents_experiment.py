"""
Tests for agent experiment functionality.
"""

import json
import pytest
from pathlib import Path

from src.agents.agent_experiment import AgentExperiment, AgentRun
from src.agents.agent import AgentConfig
from dsat.scryptorum.core.runs import RunType
from test.conftest import verify_json_file, verify_jsonl_file
from test.echo_agent import EchoAgent, create_echo_agent_config


class TestAgentRun:
    """Test AgentRun enhanced logging functionality."""

    def test_agent_run_creation(self, temp_dir: Path):
        """Test AgentRun creation and basic functionality."""
        experiment_path = temp_dir / "test_experiment"
        experiment_path.mkdir(parents=True)
        
        run = AgentRun(experiment_path, RunType.TRIAL)
        
        assert run.run_type == RunType.TRIAL
        assert run.run_id == "trial_run"
        assert run.run_dir.exists()

    def test_log_agent_created(self, temp_dir: Path):
        """Test logging agent creation."""
        experiment_path = temp_dir / "test_experiment"
        experiment_path.mkdir(parents=True)
        
        run = AgentRun(experiment_path, RunType.TRIAL)
        
        config_data = {
            "model_provider": "anthropic",
            "model_family": "claude",
            "model_version": "claude-3-5-haiku-latest",
            "prompt": "assistant:v1"
        }
        
        run.log_agent_created("test_agent", config_data)
        
        # Verify the event was logged
        log_entries = verify_jsonl_file(run.log_file, expected_entries=2)  # run_started + agent_created
        agent_event = next(entry for entry in log_entries if entry["event_type"] == "agent_created")
        
        assert agent_event["agent_name"] == "test_agent"
        assert agent_event["model_provider"] == "anthropic"
        assert agent_event["model_family"] == "claude"
        assert agent_event["prompt"] == "assistant:v1"

    def test_log_agent_invoke(self, temp_dir: Path):
        """Test logging agent invocation."""
        experiment_path = temp_dir / "test_experiment"
        experiment_path.mkdir(parents=True)
        
        run = AgentRun(experiment_path, RunType.TRIAL)
        
        run.log_agent_invoke(
            agent_name="test_agent",
            user_prompt="Hello, world!",
            system_prompt="You are a helpful assistant",
            response="Hello! How can I help you?",
            duration_ms=150.5
        )
        
        # Verify the event was logged
        log_entries = verify_jsonl_file(run.log_file, expected_entries=2)  # run_started + agent_invoke
        invoke_event = next(entry for entry in log_entries if entry["event_type"] == "agent_invoke")
        
        assert invoke_event["agent_name"] == "test_agent"
        assert invoke_event["user_prompt"] == "Hello, world!"
        assert invoke_event["system_prompt"] == "You are a helpful assistant"
        assert invoke_event["response"] == "Hello! How can I help you?"
        assert invoke_event["duration_ms"] == 150.5

    def test_enhanced_llm_call_logging(self, temp_dir: Path):
        """Test enhanced LLM call logging with agent information."""
        experiment_path = temp_dir / "test_experiment"
        experiment_path.mkdir(parents=True)
        
        run = AgentRun(experiment_path, RunType.TRIAL)
        
        run.log_llm_call(
            model="claude-3-5-haiku-latest",
            input_data="Test input",
            output_data="Test output",
            duration_ms=200.0,
            agent_name="test_agent",
            prompt="assistant:v1"
        )
        
        # Verify the event was logged
        log_entries = verify_jsonl_file(run.log_file, expected_entries=2)  # run_started + llm_call
        llm_event = next(entry for entry in log_entries if entry["event_type"] == "llm_call")
        
        assert llm_event["model"] == "claude-3-5-haiku-latest"
        assert llm_event["agent_name"] == "test_agent"
        assert llm_event["prompt_name"] == "assistant"
        assert llm_event["prompt_version"] == "v1"


class TestAgentExperiment:
    """Test AgentExperiment functionality."""

    def test_agent_experiment_creation(self, temp_dir: Path):
        """Test AgentExperiment creation and directory structure."""
        experiment = AgentExperiment(temp_dir, "agent_test_experiment")

        # Verify basic experiment functionality
        expected_path = temp_dir / "experiments" / "agent_test_experiment"
        assert experiment.experiment_path.resolve() == expected_path.resolve()
        assert experiment.experiment_name == "agent_test_experiment"

        # Verify experiment directories exist
        expected_dirs = ["runs", "data", "config"]
        for dir_name in expected_dirs:
            dir_path = experiment.experiment_path / dir_name
            assert dir_path.exists(), f"Missing experiment directory: {dir_name}"

        # Verify experiment metadata includes agent information
        metadata = verify_json_file(experiment.experiment_path / "experiment.json")
        assert metadata["name"] == "agent_test_experiment"
        assert metadata["agents_enabled"] is True
        assert "agent_configs" in metadata

    def test_default_agent_config_creation(self, temp_dir: Path):
        """Test that AgentExperiment creates default agent config."""
        experiment = AgentExperiment(temp_dir, "default_config_test")

        # Should have created a default agent config
        agent_configs = experiment.list_agent_configs()
        assert len(agent_configs) == 1
        assert "default_config_test_agent" in agent_configs

        # Verify the default config
        config = experiment.load_agent_config("default_config_test_agent")
        assert config is not None
        assert config.agent_name == "default_config_test_agent"
        assert config.model_provider == "anthropic"
        assert config.prompt == "default:v1"

    def test_create_custom_agent_config(self, temp_dir: Path):
        """Test creating custom agent configuration."""
        experiment = AgentExperiment(temp_dir, "custom_config_test")

        # Create custom config with overrides
        config_file = experiment.create_agent_config(
            "custom_agent",
            model_provider="google",
            model_family="gemini",
            prompt="custom_prompt:v2",
            model_parameters={"temperature": 0.3}
        )

        assert config_file.exists()

        # Load and verify
        config = experiment.load_agent_config("custom_agent")
        assert config.agent_name == "custom_agent"
        assert config.model_provider == "google" 
        assert config.model_family == "gemini"
        assert config.prompt == "custom_prompt:v2"
        assert config.model_parameters["temperature"] == 0.3

    def test_create_agent(self, temp_dir: Path):
        """Test creating agent from configuration."""
        experiment = AgentExperiment(temp_dir, "agent_creation_test")

        # Create an echo agent config
        experiment.create_agent_config(
            "test_agent", 
            model_provider="echo",
            model_family="test",
            model_version="echo-v1",
            prompt="test_prompt:v1"
        )

        # Create agent - should return an EchoAgent
        agent = experiment.create_agent("test_agent")

        # Verify it's actually an EchoAgent
        assert isinstance(agent, EchoAgent)
        assert agent.config.agent_name == "test_agent"
        assert agent.model == "echo-v1"

    def test_create_agent_with_overrides(self, temp_dir: Path):
        """Test creating agent with configuration overrides."""
        experiment = AgentExperiment(temp_dir, "agent_override_test")

        # Create an echo agent config
        experiment.create_agent_config(
            "test_agent", 
            model_provider="echo",
            model_family="test",
            model_version="echo-v1",
            prompt="original:v1"
        )

        # Create agent with overrides
        agent = experiment.create_agent(
            "test_agent", 
            prompt="overridden:v2",
            model_parameters={"temperature": 0.8}
        )

        # Verify the agent was created with overridden config
        assert isinstance(agent, EchoAgent)
        assert agent.config.prompt == "overridden:v2"
        assert agent.config.model_parameters["temperature"] == 0.8

    def test_create_agent_nonexistent_config(self, temp_dir: Path):
        """Test creating agent with non-existent configuration raises error."""
        experiment = AgentExperiment(temp_dir, "nonexistent_test")

        with pytest.raises(ValueError, match="Agent configuration 'nonexistent' not found"):
            experiment.create_agent("nonexistent")

    def test_list_agent_configs(self, temp_dir: Path):
        """Test listing agent configurations."""
        experiment = AgentExperiment(temp_dir, "list_configs_test")

        # Should start with default config
        configs = experiment.list_agent_configs()
        assert "list_configs_test_agent" in configs

        # Create additional configs
        experiment.create_agent_config("agent_alpha")
        experiment.create_agent_config("agent_beta")

        configs = experiment.list_agent_configs()
        assert len(configs) == 3
        assert "agent_alpha" in configs
        assert "agent_beta" in configs

    def test_update_agent_config(self, temp_dir: Path):
        """Test updating agent configuration."""
        experiment = AgentExperiment(temp_dir, "update_test")

        # Create config to update
        experiment.create_agent_config("update_agent")

        # Update the config
        updates = {
            "model_provider": "google",
            "prompt": "updated_prompt:v3"
        }

        result = experiment.update_agent_config("update_agent", updates)
        assert result is True

        # Verify updates
        config = experiment.load_agent_config("update_agent")
        assert config.model_provider == "google"
        assert config.prompt == "updated_prompt:v3"

    def test_delete_agent_config(self, temp_dir: Path):
        """Test deleting agent configuration."""
        experiment = AgentExperiment(temp_dir, "delete_test")

        # Create config to delete
        experiment.create_agent_config("delete_agent")
        assert "delete_agent" in experiment.list_agent_configs()

        # Delete the config
        result = experiment.delete_agent_config("delete_agent")
        assert result is True

        # Verify deletion
        assert "delete_agent" not in experiment.list_agent_configs()
        config = experiment.load_agent_config("delete_agent")
        assert config is None

    def test_create_agent_run(self, temp_dir: Path):
        """Test that AgentExperiment creates AgentRun instances."""
        experiment = AgentExperiment(temp_dir, "agent_run_test")

        # Create trial run
        trial_run = experiment.create_run(RunType.TRIAL)
        assert isinstance(trial_run, AgentRun)
        assert trial_run.run_type == RunType.TRIAL

        # Create milestone run
        milestone_run = experiment.create_run(RunType.MILESTONE)
        assert isinstance(milestone_run, AgentRun)
        assert milestone_run.run_type == RunType.MILESTONE

        # Verify agent config snapshot directory exists for milestone runs
        assert (milestone_run.run_dir / "agent_configs").exists()

    def test_milestone_run_agent_config_snapshot(self, temp_dir: Path):
        """Test that milestone runs snapshot agent configurations."""
        experiment = AgentExperiment(temp_dir, "snapshot_test")

        # Create some agent configs
        experiment.create_agent_config("agent1", prompt="test1:v1")
        experiment.create_agent_config("agent2", prompt="test2:v2")

        # Create milestone run
        run = experiment.create_run(RunType.MILESTONE)

        # Verify configs were snapshotted
        snapshot_dir = run.run_dir / "agent_configs"
        assert snapshot_dir.exists()

        # Check that config files were copied
        config_files = list(snapshot_dir.glob("*_agent_config.json"))
        assert len(config_files) >= 2  # At least the 2 we created (plus default)

        # Verify we can find our configs
        config_names = [f.stem.replace("_agent_config", "") for f in config_files]
        assert "agent1" in config_names
        assert "agent2" in config_names


class TestAgentExperimentIntegration:
    """Integration tests for AgentExperiment with scryptorum decorators."""

    def test_agent_creation_logging_in_run_context(self, temp_dir: Path):
        """Test that agent creation works properly within experiment context."""
        experiment = AgentExperiment(temp_dir, "logging_test")

        # Create an echo agent config
        experiment.create_agent_config(
            "logged_agent", 
            model_provider="echo",
            model_family="test",
            model_version="echo-v1",
            prompt="test:v1"
        )
        
        # Create and use the agent
        agent = experiment.create_agent("logged_agent")
        
        # Test that the agent works
        response = agent.invoke("Hello test")
        assert "Echo: Hello test" in response
        assert isinstance(agent, EchoAgent)
        
        # Verify agent was created with correct config
        assert agent.config.agent_name == "logged_agent"
        assert agent.config.model_provider == "echo"
        assert agent.config.prompt == "test:v1"

    def test_agent_experiment_vs_base_experiment(self, temp_dir: Path):
        """Test that AgentExperiment provides enhanced functionality over base Experiment."""
        from dsat.scryptorum.core.experiment import Experiment

        # Create base experiment
        base_experiment = Experiment(temp_dir / "base", "base_test")

        # Create agent experiment
        agent_experiment = AgentExperiment(temp_dir / "agent", "agent_test")

        # Base experiment should not have agent methods
        assert not hasattr(base_experiment, 'create_agent')
        assert not hasattr(base_experiment, 'list_agent_configs')

        # Agent experiment should have agent methods
        assert hasattr(agent_experiment, 'create_agent')
        assert hasattr(agent_experiment, 'list_agent_configs')
        assert hasattr(agent_experiment, 'create_agent_config')

        # Both should have base functionality
        assert hasattr(base_experiment, 'create_run')
        assert hasattr(agent_experiment, 'create_run')

        # Agent experiment runs should be AgentRun instances
        base_run = base_experiment.create_run()
        agent_run = agent_experiment.create_run()

        assert type(base_run).__name__ == 'Run'
        assert isinstance(agent_run, AgentRun)