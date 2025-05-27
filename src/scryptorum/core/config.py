"""
Configuration management for experiments and agents.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class AgentConfig:
    """Agent configuration schema."""

    agent_name: str
    llm_family: str  # e.g., "openai", "anthropic", "google", "meta"
    llm_model_name: str  # e.g., "gpt-4", "claude-3-sonnet", "gemini-pro"
    llm_meta: Dict[str, Any]  # Provider-specific details, model params, etc.
    prompt_name: str  # Name/identifier of the prompt template
    prompt_version: str  # Prompt template version/identifier
    custom_configs: Dict[str, Any]  # Additional custom configuration

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create from dictionary."""
        return cls(**data)

    def save_to_file(self, file_path: Path) -> None:
        """Save config to JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: Path) -> "AgentConfig":
        """Load config from JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def create_default_agent_config(agent_name: str) -> AgentConfig:
    """Create a default agent configuration template."""
    return AgentConfig(
        agent_name=agent_name,
        llm_family="openai",  # Default to OpenAI
        llm_model_name="gpt-4",
        llm_meta={
            "provider": "openai",
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "api_version": "v1",
            "timeout": 30,
        },
        prompt_name="default_prompt",
        prompt_version="v1",
        custom_configs={
            "retry_attempts": 3,
            "rate_limit_rpm": 60,
            "logging_enabled": True,
            "cache_responses": False,
        },
    )


class ConfigManager:
    """Manages experiment configuration files."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def create_agent_config(
        self, agent_name: str, config: Optional[AgentConfig] = None
    ) -> Path:
        """Create an agent configuration file."""
        if config is None:
            config = create_default_agent_config(agent_name)

        config_file = self.config_dir / f"{agent_name}_config.json"
        config.save_to_file(config_file)
        return config_file

    def load_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Load an agent configuration."""
        config_file = self.config_dir / f"{agent_name}_config.json"
        if not config_file.exists():
            return None

        return AgentConfig.load_from_file(config_file)

    def list_agent_configs(self) -> list[str]:
        """List all available agent configurations."""
        configs = []
        for config_file in self.config_dir.glob("*_config.json"):
            # Extract agent name from filename
            agent_name = config_file.stem.replace("_config", "")
            configs.append(agent_name)
        return sorted(configs)

    def update_agent_config(self, agent_name: str, updates: Dict[str, Any]) -> bool:
        """Update an existing agent configuration."""
        config = self.load_agent_config(agent_name)
        if config is None:
            return False

        # Update fields
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Save updated config
        self.create_agent_config(agent_name, config)
        return True

    def delete_agent_config(self, agent_name: str) -> bool:
        """Delete an agent configuration."""
        config_file = self.config_dir / f"{agent_name}_config.json"
        if config_file.exists():
            config_file.unlink()
            return True
        return False
