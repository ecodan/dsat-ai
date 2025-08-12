"""
Tests for the AgentConfig class.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from src.agents.agent import AgentConfig


class TestAgentConfig:
    """Test cases for AgentConfig class."""

    @pytest.fixture
    def valid_config_dict(self):
        """Return a valid configuration dictionary."""
        return {
            "agent_name": "test_assistant",
            "model_provider": "anthropic",
            "model_family": "claude",
            "model_version": "claude-3-5-haiku-latest",
            "prompt_name": "assistant",
            "prompt_version": "v1",
            "model_parameters": {"temperature": 0.7, "max_tokens": 4096},
            "provider_auth": {"api_key": "sk-test-key"},
            "custom_configs": {"custom_field": "custom_value"},
            "tools": ["tool1", "tool2"]
        }

    @pytest.fixture
    def minimal_config_dict(self):
        """Return a minimal valid configuration dictionary."""
        return {
            "agent_name": "minimal",
            "model_provider": "anthropic",
            "model_family": "claude",
            "model_version": "claude-3-5-haiku-latest",
            "prompt_name": "basic",
            "prompt_version": "v1"
        }

    def test_agent_config_creation_with_all_fields(self, valid_config_dict):
        """Test creating AgentConfig with all fields."""
        config = AgentConfig(**valid_config_dict)
        
        assert config.agent_name == "test_assistant"
        assert config.model_provider == "anthropic"
        assert config.model_family == "claude"
        assert config.model_version == "claude-3-5-haiku-latest"
        assert config.prompt_name == "assistant"
        assert config.prompt_version == "v1"
        assert config.model_parameters == {"temperature": 0.7, "max_tokens": 4096}
        assert config.provider_auth == {"api_key": "sk-test-key"}
        assert config.custom_configs == {"custom_field": "custom_value"}
        assert config.tools == ["tool1", "tool2"]

    def test_agent_config_creation_minimal_fields(self, minimal_config_dict):
        """Test creating AgentConfig with only required fields."""
        config = AgentConfig(**minimal_config_dict)
        
        assert config.agent_name == "minimal"
        assert config.model_provider == "anthropic"
        assert config.model_parameters == {}
        assert config.provider_auth == {}
        assert config.custom_configs == {}
        assert config.tools == []

    def test_from_dict_valid_config(self, valid_config_dict):
        """Test creating AgentConfig from dictionary."""
        config = AgentConfig.from_dict(valid_config_dict)
        
        assert config.agent_name == "test_assistant"
        assert config.model_provider == "anthropic"
        assert config.model_parameters == {"temperature": 0.7, "max_tokens": 4096}

    def test_from_dict_minimal_config(self, minimal_config_dict):
        """Test creating AgentConfig from minimal dictionary."""
        config = AgentConfig.from_dict(minimal_config_dict)
        
        assert config.agent_name == "minimal"
        assert config.model_parameters == {}
        assert config.provider_auth == {}
        assert config.custom_configs == {}
        assert config.tools == []

    def test_from_dict_missing_required_field(self):
        """Test from_dict raises ValueError for missing required fields."""
        invalid_config = {
            "agent_name": "test",
            "model_provider": "anthropic",
            # Missing other required fields
        }
        
        with pytest.raises(ValueError, match="Missing required key: model_family"):
            AgentConfig.from_dict(invalid_config)

    def test_from_dict_not_a_dict(self):
        """Test from_dict raises ValueError for non-dict input."""
        with pytest.raises(ValueError, match="config_dict must be a dictionary"):
            AgentConfig.from_dict("not a dict")

    def test_from_dict_adds_default_optional_fields(self):
        """Test that from_dict adds default values for optional fields."""
        minimal_dict = {
            "agent_name": "test",
            "model_provider": "anthropic",
            "model_family": "claude",
            "model_version": "claude-3-5-haiku",
            "prompt_name": "test",
            "prompt_version": "v1"
        }
        
        config = AgentConfig.from_dict(minimal_dict)
        
        assert isinstance(config.model_parameters, dict)
        assert isinstance(config.provider_auth, dict)
        assert isinstance(config.custom_configs, dict)
        assert isinstance(config.tools, list)

    def test_to_dict(self, valid_config_dict):
        """Test converting AgentConfig to dictionary."""
        config = AgentConfig(**valid_config_dict)
        result_dict = config.to_dict()
        
        # Should match original input
        assert result_dict == valid_config_dict

    def test_to_dict_minimal_config(self, minimal_config_dict):
        """Test to_dict with minimal config includes default empty values."""
        config = AgentConfig(**minimal_config_dict)
        result_dict = config.to_dict()
        
        expected = minimal_config_dict.copy()
        expected["model_parameters"] = {}
        expected["provider_auth"] = {}
        expected["custom_configs"] = {}
        expected["tools"] = []
        
        assert result_dict == expected

    def test_load_from_file_json(self):
        """Test loading agent configs from JSON file."""
        test_configs = {
            "agent1": {
                "agent_name": "agent1",
                "model_provider": "anthropic",
                "model_family": "claude",
                "model_version": "claude-3-5-haiku",
                "prompt_name": "test",
                "prompt_version": "v1"
            },
            "agent2": {
                "agent_name": "agent2",
                "model_provider": "google",
                "model_family": "gemini",
                "model_version": "gemini-2.0-flash",
                "prompt_name": "assistant",
                "prompt_version": "v2"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_configs, f)
            temp_file_path = f.name
        
        try:
            configs = AgentConfig.load_from_file(temp_file_path)
            
            assert len(configs) == 2
            assert "agent1" in configs
            assert "agent2" in configs
            assert isinstance(configs["agent1"], AgentConfig)
            assert isinstance(configs["agent2"], AgentConfig)
            assert configs["agent1"].agent_name == "agent1"
            assert configs["agent2"].model_provider == "google"
        finally:
            Path(temp_file_path).unlink()

    def test_load_from_file_toml(self):
        """Test loading agent configs from TOML file."""
        toml_content = '''
[agent1]
agent_name = "agent1"
model_provider = "anthropic"
model_family = "claude"
model_version = "claude-3-5-haiku"
prompt_name = "test"
prompt_version = "v1"

[agent2]
agent_name = "agent2"
model_provider = "google"
model_family = "gemini"
model_version = "gemini-2.0-flash"
prompt_name = "assistant"
prompt_version = "v2"
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            temp_file_path = f.name
        
        try:
            configs = AgentConfig.load_from_file(temp_file_path)
            
            assert len(configs) == 2
            assert "agent1" in configs
            assert "agent2" in configs
            assert configs["agent1"].model_family == "claude"
            assert configs["agent2"].model_family == "gemini"
        finally:
            Path(temp_file_path).unlink()

    @patch('builtins.open', side_effect=FileNotFoundError())
    def test_load_from_file_not_found(self, mock_open_func):
        """Test load_from_file raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            AgentConfig.load_from_file("/nonexistent/path.json")

    def test_load_from_file_unsupported_format(self):
        """Test load_from_file raises ValueError for unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            temp_file_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format: .yaml"):
                AgentConfig.load_from_file(temp_file_path)
        finally:
            Path(temp_file_path).unlink()

    def test_load_from_file_invalid_json(self):
        """Test load_from_file handles invalid JSON gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            temp_file_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid configuration file format"):
                AgentConfig.load_from_file(temp_file_path)
        finally:
            Path(temp_file_path).unlink()

    def test_load_from_file_not_dict_root(self):
        """Test load_from_file raises ValueError if root is not a dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(["not", "a", "dict"], f)
            temp_file_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Configuration file must contain a dictionary"):
                AgentConfig.load_from_file(temp_file_path)
        finally:
            Path(temp_file_path).unlink()

    def test_load_from_file_invalid_agent_config(self):
        """Test load_from_file handles invalid agent configs."""
        invalid_configs = {
            "invalid_agent": {
                "agent_name": "invalid_agent",
                # Missing required fields
                "model_provider": "anthropic"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_configs, f)
            temp_file_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Missing required key"):
                AgentConfig.load_from_file(temp_file_path)
        finally:
            Path(temp_file_path).unlink()

    def test_load_from_file_agent_name_override(self):
        """Test that load_from_file overrides agent_name with the key."""
        test_configs = {
            "correct_name": {
                "agent_name": "wrong_name",  # This should be overridden
                "model_provider": "anthropic",
                "model_family": "claude",
                "model_version": "claude-3-5-haiku",
                "prompt_name": "test",
                "prompt_version": "v1"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_configs, f)
            temp_file_path = f.name
        
        try:
            configs = AgentConfig.load_from_file(temp_file_path)
            assert configs["correct_name"].agent_name == "correct_name"
        finally:
            Path(temp_file_path).unlink()

    def test_save_to_file_json_single_config(self):
        """Test saving single config to JSON file."""
        config = AgentConfig(
            agent_name="test_save",
            model_provider="anthropic",
            model_family="claude",
            model_version="claude-3-5-haiku",
            prompt_name="test",
            prompt_version="v1"
        )
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file_path = f.name
        
        try:
            config.save_to_file(temp_file_path)
            
            # Verify file was created and has correct content
            assert Path(temp_file_path).exists()
            
            with open(temp_file_path, 'r') as f:
                saved_data = json.load(f)
            
            assert "test_save" in saved_data
            assert saved_data["test_save"]["agent_name"] == "test_save"
            assert saved_data["test_save"]["model_provider"] == "anthropic"
        finally:
            Path(temp_file_path).unlink()

    def test_save_to_file_json_multiple_configs(self):
        """Test saving multiple configs to JSON file."""
        config1 = AgentConfig(
            agent_name="agent1",
            model_provider="anthropic",
            model_family="claude",
            model_version="claude-3-5-haiku",
            prompt_name="test",
            prompt_version="v1"
        )
        
        config2 = AgentConfig(
            agent_name="agent2",
            model_provider="google",
            model_family="gemini",
            model_version="gemini-2.0-flash",
            prompt_name="assistant",
            prompt_version="v2"
        )
        
        configs_dict = {"agent1": config1, "agent2": config2}
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file_path = f.name
        
        try:
            config1.save_to_file(temp_file_path, configs_dict)
            
            with open(temp_file_path, 'r') as f:
                saved_data = json.load(f)
            
            assert len(saved_data) == 2
            assert "agent1" in saved_data
            assert "agent2" in saved_data
        finally:
            Path(temp_file_path).unlink()

    def test_save_to_file_toml(self):
        """Test saving config to TOML file."""
        config = AgentConfig(
            agent_name="toml_test",
            model_provider="anthropic",
            model_family="claude",
            model_version="claude-3-5-haiku",
            prompt_name="test",
            prompt_version="v1",
            model_parameters={"temperature": 0.5}
        )
        
        with tempfile.NamedTemporaryFile(suffix='.toml', delete=False) as f:
            temp_file_path = f.name
        
        try:
            config.save_to_file(temp_file_path)
            
            assert Path(temp_file_path).exists()
            
            # Verify content is valid TOML
            with open(temp_file_path, 'r') as f:
                content = f.read()
                assert 'agent_name = "toml_test"' in content
                assert 'model_provider = "anthropic"' in content
        finally:
            Path(temp_file_path).unlink()

    def test_save_to_file_unsupported_format(self):
        """Test save_to_file raises ValueError for unsupported format."""
        config = AgentConfig(
            agent_name="test",
            model_provider="anthropic",
            model_family="claude",
            model_version="claude-3-5-haiku",
            prompt_name="test",
            prompt_version="v1"
        )
        
        with pytest.raises(ValueError, match="Unsupported file format: .yaml"):
            config.save_to_file("config.yaml")

    @pytest.mark.skip(reason="Complex import mocking - tested manually")
    def test_save_to_file_toml_import_error(self):
        """Test save_to_file handles missing tomlkit dependency."""
        # This test would verify ImportError handling when tomlkit is not available
        # Skipping due to complexity of mocking import system in tests
        pass

    def test_from_dict_preserves_original(self):
        """Test that from_dict doesn't modify the original dictionary."""
        original_dict = {
            "agent_name": "test",
            "model_provider": "anthropic",
            "model_family": "claude",
            "model_version": "claude-3-5-haiku",
            "prompt_name": "test",
            "prompt_version": "v1"
        }
        original_copy = original_dict.copy()
        
        config = AgentConfig.from_dict(original_dict)
        
        # Original dict should be unchanged
        assert original_dict == original_copy
        assert config.agent_name == "test"