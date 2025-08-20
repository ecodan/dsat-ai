"""
Configuration management for experiments and agents.
"""

from pathlib import Path
from typing import Dict, Any, Optional

# Scryptorum config management - no agent dependencies


class ConfigManager:
    """Manages experiment configuration files (generic JSON/TOML configs)."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config_name: str, config_data: Dict[str, Any]) -> Path:
        """Save a configuration to JSON file."""
        config_file = self.config_dir / f"{config_name}.json"

        import json

        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        return config_file

    def load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Load a configuration from JSON file."""
        config_file = self.config_dir / f"{config_name}.json"
        if not config_file.exists():
            return None

        import json

        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def list_configs(self, pattern: str = "*.json") -> list[str]:
        """List all available configuration files."""
        configs = []
        for config_file in self.config_dir.glob(pattern):
            # Extract config name from filename
            config_name = config_file.stem
            configs.append(config_name)
        return sorted(configs)

    def delete_config(self, config_name: str) -> bool:
        """Delete a configuration file."""
        config_file = self.config_dir / f"{config_name}.json"
        if config_file.exists():
            config_file.unlink()
            return True
        return False
