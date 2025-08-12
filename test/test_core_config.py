"""
Tests for core.config module - generic configuration management.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.scryptorum.core.config import ConfigManager


class TestConfigManager:
    """Test generic ConfigManager functionality."""

    def test_config_manager_initialization(self):
        """Test ConfigManager initialization creates directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            assert manager.config_dir == config_dir
            assert config_dir.exists()
            assert config_dir.is_dir()

    def test_save_and_load_config(self):
        """Test saving and loading generic configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            # Test config data
            config_data = {
                "name": "test_config",
                "settings": {
                    "debug": True,
                    "timeout": 30
                },
                "features": ["feature1", "feature2"]
            }

            # Save config
            config_file = manager.save_config("test_config", config_data)
            expected_file = config_dir / "test_config.json"
            assert config_file == expected_file
            assert config_file.exists()

            # Load config
            loaded_data = manager.load_config("test_config")
            assert loaded_data == config_data

    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration returns None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            config = manager.load_config("nonexistent_config")
            assert config is None

    def test_list_configs(self):
        """Test listing all available configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            # Initially empty
            configs = manager.list_configs()
            assert configs == []

            # Create several configs
            manager.save_config("alpha", {"type": "alpha"})
            manager.save_config("beta", {"type": "beta"})
            manager.save_config("gamma", {"type": "gamma"})

            configs = manager.list_configs()
            assert len(configs) == 3
            assert "alpha" in configs
            assert "beta" in configs
            assert "gamma" in configs
            assert configs == sorted(configs)  # Should be sorted

    def test_list_configs_with_pattern(self):
        """Test listing configs with specific pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            # Create different types of files
            manager.save_config("config1", {"data": "test"})
            manager.save_config("config2", {"data": "test"})
            
            # Create a non-JSON file manually
            (config_dir / "readme.txt").write_text("test")

            # List only JSON configs
            configs = manager.list_configs("*.json")
            assert len(configs) == 2
            assert "config1" in configs
            assert "config2" in configs

    def test_delete_config(self):
        """Test deleting configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            # Create a config to delete
            manager.save_config("delete_test", {"data": "test"})
            assert "delete_test" in manager.list_configs()

            # Delete the config
            result = manager.delete_config("delete_test")
            assert result is True

            # Verify deletion
            assert "delete_test" not in manager.list_configs()
            config = manager.load_config("delete_test")
            assert config is None

    def test_delete_nonexistent_config(self):
        """Test deleting non-existent configuration returns False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            result = manager.delete_config("nonexistent")
            assert result is False


