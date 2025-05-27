"""
Tests for core.config module - agent configuration management.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.scryptorum.core.config import (
    AgentConfig,
    ConfigManager,
    create_default_agent_config,
)


class TestAgentConfig:
    """Test AgentConfig dataclass functionality."""

    def test_agent_config_creation(self):
        """Test creating an AgentConfig with all fields."""
        config = AgentConfig(
            agent_name="test_agent",
            llm_family="openai",
            llm_model_name="gpt-4",
            llm_meta={"temperature": 0.7, "max_tokens": 1000},
            prompt_name="test_prompt",
            prompt_version="v1",
            custom_configs={"retry_attempts": 3},
        )

        assert config.agent_name == "test_agent"
        assert config.llm_family == "openai"
        assert config.llm_model_name == "gpt-4"
        assert config.llm_meta["temperature"] == 0.7
        assert config.prompt_name == "test_prompt"
        assert config.prompt_version == "v1"
        assert config.custom_configs["retry_attempts"] == 3

    def test_agent_config_to_dict(self):
        """Test converting AgentConfig to dictionary."""
        config = AgentConfig(
            agent_name="test_agent",
            llm_family="anthropic",
            llm_model_name="claude-3-sonnet",
            llm_meta={"temperature": 0.5},
            prompt_name="conversation_prompt",
            prompt_version="v2",
            custom_configs={"logging": True},
        )

        config_dict = config.to_dict()

        expected = {
            "agent_name": "test_agent",
            "llm_family": "anthropic",
            "llm_model_name": "claude-3-sonnet",
            "llm_meta": {"temperature": 0.5},
            "prompt_name": "conversation_prompt",
            "prompt_version": "v2",
            "custom_configs": {"logging": True},
        }

        assert config_dict == expected

    def test_agent_config_from_dict(self):
        """Test creating AgentConfig from dictionary."""
        config_data = {
            "agent_name": "dict_agent",
            "llm_family": "google",
            "llm_model_name": "gemini-pro",
            "llm_meta": {"temperature": 0.3, "top_p": 0.9},
            "prompt_name": "analysis_prompt",
            "prompt_version": "v2",
            "custom_configs": {"cache_responses": True, "retry_attempts": 5},
        }

        config = AgentConfig.from_dict(config_data)

        assert config.agent_name == "dict_agent"
        assert config.llm_family == "google"
        assert config.llm_model_name == "gemini-pro"
        assert config.llm_meta["temperature"] == 0.3
        assert config.llm_meta["top_p"] == 0.9
        assert config.prompt_name == "analysis_prompt"
        assert config.prompt_version == "v2"
        assert config.custom_configs["cache_responses"] is True
        assert config.custom_configs["retry_attempts"] == 5

    def test_agent_config_file_save_load(self):
        """Test saving and loading AgentConfig to/from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.json"

            # Create and save config
            original_config = AgentConfig(
                agent_name="file_test_agent",
                llm_family="meta",
                llm_model_name="llama-2-70b",
                llm_meta={"temperature": 0.8, "max_tokens": 2000},
                prompt_name="creative_prompt",
                prompt_version="v3",
                custom_configs={"streaming": True},
            )

            original_config.save_to_file(config_file)

            # Verify file exists and has correct content
            assert config_file.exists()

            with open(config_file) as f:
                saved_data = json.load(f)

            assert saved_data["agent_name"] == "file_test_agent"
            assert saved_data["prompt_name"] == "creative_prompt"

            # Load config from file
            loaded_config = AgentConfig.load_from_file(config_file)

            assert loaded_config.agent_name == original_config.agent_name
            assert loaded_config.llm_family == original_config.llm_family
            assert loaded_config.llm_model_name == original_config.llm_model_name
            assert loaded_config.llm_meta == original_config.llm_meta
            assert loaded_config.prompt_name == original_config.prompt_name
            assert loaded_config.prompt_version == original_config.prompt_version
            assert loaded_config.custom_configs == original_config.custom_configs


class TestCreateDefaultAgentConfig:
    """Test the create_default_agent_config function."""

    def test_create_default_config(self):
        """Test creating a default agent configuration."""
        agent_name = "default_test_agent"
        config = create_default_agent_config(agent_name)

        assert config.agent_name == agent_name
        assert config.llm_family == "openai"
        assert config.llm_model_name == "gpt-4"
        assert config.prompt_name == "default_prompt"
        assert config.prompt_version == "v1"

        # Check llm_meta structure
        assert "provider" in config.llm_meta
        assert "temperature" in config.llm_meta
        assert "max_tokens" in config.llm_meta
        assert config.llm_meta["provider"] == "openai"
        assert config.llm_meta["temperature"] == 0.7

        # Check custom_configs structure
        assert "retry_attempts" in config.custom_configs
        assert "logging_enabled" in config.custom_configs
        assert config.custom_configs["retry_attempts"] == 3
        assert config.custom_configs["logging_enabled"] is True

    def test_default_config_prompt_name_included(self):
        """Test that default config includes prompt_name field."""
        config = create_default_agent_config("test_agent")

        # Ensure prompt_name is present and has expected value
        assert hasattr(config, "prompt_name")
        assert config.prompt_name == "default_prompt"

        # Ensure it's included in dict representation
        config_dict = config.to_dict()
        assert "prompt_name" in config_dict
        assert config_dict["prompt_name"] == "default_prompt"


class TestConfigManager:
    """Test ConfigManager functionality."""

    def test_config_manager_initialization(self):
        """Test ConfigManager initialization creates directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            assert manager.config_dir == config_dir
            assert config_dir.exists()
            assert config_dir.is_dir()

    def test_create_agent_config_default(self):
        """Test creating agent config with default values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            config_file = manager.create_agent_config("test_agent")

            expected_file = config_dir / "test_agent_config.json"
            assert config_file == expected_file
            assert config_file.exists()

            # Load and verify content
            with open(config_file) as f:
                data = json.load(f)

            assert data["agent_name"] == "test_agent"
            assert data["llm_family"] == "openai"
            assert data["prompt_name"] == "default_prompt"

    def test_create_agent_config_custom(self):
        """Test creating agent config with custom values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            custom_config = AgentConfig(
                agent_name="custom_agent",
                llm_family="anthropic",
                llm_model_name="claude-3-haiku",
                llm_meta={"temperature": 0.2},
                prompt_name="specialized_prompt",
                prompt_version="v2",
                custom_configs={"fast_mode": True},
            )

            config_file = manager.create_agent_config("custom_agent", custom_config)

            # Verify file content
            with open(config_file) as f:
                data = json.load(f)

            assert data["agent_name"] == "custom_agent"
            assert data["llm_family"] == "anthropic"
            assert data["prompt_name"] == "specialized_prompt"
            assert data["prompt_version"] == "v2"

    def test_load_agent_config(self):
        """Test loading existing agent configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            # Create a config first
            original_config = AgentConfig(
                agent_name="load_test_agent",
                llm_family="google",
                llm_model_name="gemini-pro",
                llm_meta={"temperature": 0.6},
                prompt_name="test_prompt",
                prompt_version="v2",
                custom_configs={"debug": True},
            )

            manager.create_agent_config("load_test_agent", original_config)

            # Load the config
            loaded_config = manager.load_agent_config("load_test_agent")

            assert loaded_config is not None
            assert loaded_config.agent_name == "load_test_agent"
            assert loaded_config.llm_family == "google"
            assert loaded_config.prompt_name == "test_prompt"
            assert loaded_config.custom_configs["debug"] is True

    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration returns None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            config = manager.load_agent_config("nonexistent_agent")
            assert config is None

    def test_list_agent_configs(self):
        """Test listing all available agent configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            # Initially empty
            configs = manager.list_agent_configs()
            assert configs == []

            # Create several configs
            manager.create_agent_config("agent_alpha")
            manager.create_agent_config("agent_beta")
            manager.create_agent_config("agent_gamma")

            configs = manager.list_agent_configs()
            assert len(configs) == 3
            assert "agent_alpha" in configs
            assert "agent_beta" in configs
            assert "agent_gamma" in configs
            assert configs == sorted(configs)  # Should be sorted

    def test_update_agent_config(self):
        """Test updating existing agent configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            # Create initial config
            manager.create_agent_config("update_test_agent")

            # Update the config
            updates = {
                "llm_family": "anthropic",
                "llm_model_name": "claude-3-opus",
                "prompt_name": "updated_prompt",
            }

            result = manager.update_agent_config("update_test_agent", updates)
            assert result is True

            # Verify updates
            updated_config = manager.load_agent_config("update_test_agent")
            assert updated_config.llm_family == "anthropic"
            assert updated_config.llm_model_name == "claude-3-opus"
            assert updated_config.prompt_name == "updated_prompt"

    def test_update_nonexistent_config(self):
        """Test updating non-existent configuration returns False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            result = manager.update_agent_config(
                "nonexistent", {"llm_family": "openai"}
            )
            assert result is False

    def test_delete_agent_config(self):
        """Test deleting agent configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            # Create a config to delete
            manager.create_agent_config("delete_test_agent")
            assert "delete_test_agent" in manager.list_agent_configs()

            # Delete the config
            result = manager.delete_agent_config("delete_test_agent")
            assert result is True

            # Verify deletion
            assert "delete_test_agent" not in manager.list_agent_configs()
            config = manager.load_agent_config("delete_test_agent")
            assert config is None

    def test_delete_nonexistent_config(self):
        """Test deleting non-existent configuration returns False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            result = manager.delete_agent_config("nonexistent")
            assert result is False


class TestIntegrationWithExperiment:
    """Test config integration with Experiment class."""

    def test_experiment_creates_default_config_with_prompt_name(self):
        """Test that experiment creates default config including prompt_name."""
        from src.scryptorum.core.experiment import Experiment

        with tempfile.TemporaryDirectory() as temp_dir:
            experiment = Experiment(temp_dir, "config_test_experiment")

            # Check that default config was created
            configs = experiment.config.list_agent_configs()
            assert len(configs) == 1
            assert "config_test_experiment_agent" in configs

            # Load and verify the default config includes prompt_name
            default_config = experiment.config.load_agent_config(
                "config_test_experiment_agent"
            )
            assert default_config is not None
            assert hasattr(default_config, "prompt_name")
            assert default_config.prompt_name == "default_prompt"
            assert default_config.prompt_version == "v1"

    def test_experiment_create_custom_config_with_prompt_name(self):
        """Test creating custom config through experiment with prompt_name override."""
        from src.scryptorum.core.experiment import Experiment

        with tempfile.TemporaryDirectory() as temp_dir:
            experiment = Experiment(temp_dir, "custom_config_test")

            # Create custom config with prompt_name override
            config_file = experiment.create_agent_config(
                "custom_agent",
                llm_family="anthropic",
                prompt_name="custom_analysis_prompt",
                prompt_version="v3",
            )

            assert config_file.exists()

            # Load and verify
            custom_config = experiment.config.load_agent_config("custom_agent")
            assert custom_config.agent_name == "custom_agent"
            assert custom_config.llm_family == "anthropic"
            assert custom_config.prompt_name == "custom_analysis_prompt"
            assert custom_config.prompt_version == "v3"
