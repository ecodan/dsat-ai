"""
Agent-aware experiment management that extends scryptorum.

This module provides AgentExperiment, which extends the base scryptorum Experiment
class with agent-specific functionality like agent creation, configuration management,
and enhanced logging.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from dsat.scryptorum.core.experiment import Experiment
from dsat.scryptorum.core.runs import Run, RunType
from .agent import Agent, AgentConfig


class AgentRun(Run):
    """Enhanced Run class with agent-specific logging capabilities."""

    def log_agent_created(self, agent_name: str, config_data: Dict[str, any]) -> None:
        """Log agent creation with configuration details."""
        self.log_event(
            "agent_created",
            {
                "agent_name": agent_name,
                "model_provider": config_data.get("model_provider"),
                "model_family": config_data.get("model_family"),
                "model_version": config_data.get("model_version"),
                "prompt": config_data.get("prompt"),
            },
        )

    def log_agent_invoke(
        self,
        agent_name: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        response: Optional[str] = None,
        duration_ms: Optional[float] = None,
        **metadata,
    ) -> None:
        """Log agent invocation with full context."""
        invoke_data = {
            "agent_name": agent_name,
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "response": response,
            "duration_ms": duration_ms,
            **metadata,
        }
        self.log_event("agent_invoke", invoke_data)

    def log_llm_call(
        self,
        model: str,
        input_data: any,
        output_data: any,
        duration_ms: Optional[float] = None,
        agent_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        prompt_version: Optional[str] = None,
        prompt: Optional[str] = None,
        **metadata,
    ) -> None:
        """Enhanced LLM call logging with agent information."""
        llm_data = {
            "model": model,
            "input": input_data,
            "output": output_data,
            "duration_ms": duration_ms,
            **metadata,
        }

        # Add agent information if available
        if agent_name:
            llm_data["agent_name"] = agent_name

        # Handle prompt information - either separate fields or combined format
        if prompt:
            # Parse "name:version" format
            if ":" in prompt:
                prompt_name, prompt_version = prompt.split(":", 1)
            else:
                prompt_name, prompt_version = prompt, "latest"

        if prompt_name:
            llm_data["prompt_name"] = prompt_name
        if prompt_version:
            llm_data["prompt_version"] = prompt_version

        self.log_event("llm_call", llm_data)

    def snapshot_agent_configs(self, config_dir: Path) -> None:
        """Snapshot agent configurations to the run directory."""
        snapshot_dir = self.run_dir / "agent_configs"
        snapshot_dir.mkdir(exist_ok=True)

        # Copy all agent config files
        for config_file in config_dir.glob("*_agent_config.json"):
            dest_file = snapshot_dir / config_file.name
            dest_file.write_text(config_file.read_text())


class AgentExperiment(Experiment):
    """
    Agent-aware experiment that extends scryptorum.Experiment with agent capabilities.

    Provides:
    - Agent creation from configurations
    - Agent configuration management
    - Automatic agent config snapshotting
    - Enhanced logging for agent usage
    """

    def __init__(self, project_root: Union[str, Path], experiment_name: str):
        super().__init__(project_root, experiment_name)
        # Initialize with agent-specific setup
        self._create_default_agent_configs()

    def _create_default_agent_configs(self) -> None:
        """Create default agent configuration files if none exist."""
        # Check if any agent configs already exist
        existing_configs = list(self.config_dir.glob("*_agent_config.json"))

        if not existing_configs:
            # Create a default agent config
            default_agent_name = f"{self.experiment_name}_agent"
            default_config = self._create_default_agent_config(default_agent_name)
            config_file = self.create_agent_config(default_agent_name, default_config)

            # Log the creation (fall back to print since no run context yet)
            print(f"Created default agent config: {config_file}")

    def _create_default_agent_config(self, agent_name: str) -> AgentConfig:
        """Create a default agent configuration template."""
        return AgentConfig(
            agent_name=agent_name,
            model_provider="anthropic",  # Default to Anthropic
            model_family="claude",
            model_version="claude-3-5-haiku-latest",
            prompt="default:v1",
            model_parameters={
                "temperature": 0.7,
                "max_tokens": 4096,
            },
            provider_auth={"api_key": "your-api-key-here"},
            custom_configs={
                "retry_attempts": 3,
                "rate_limit_rpm": 60,
                "logging_enabled": True,
                "cache_responses": False,
            },
            tools=[],
        )

    def create_agent_config(
        self, agent_name: str, config: Optional[AgentConfig] = None, **config_overrides
    ) -> Path:
        """Create a new agent configuration with optional overrides."""
        if config is None:
            config = self._create_default_agent_config(agent_name)

        # Apply any overrides
        if config_overrides:
            # Create a copy with overrides applied
            config_dict = config.to_dict()

            for key, value in config_overrides.items():
                if key in config_dict:
                    config_dict[key] = value
                elif key in ["model_parameters", "provider_auth", "custom_configs"]:
                    if isinstance(value, dict):
                        config_dict[key].update(value)
                    else:
                        config_dict[key] = value

            config = AgentConfig.from_dict(config_dict)

        # Save to agent-specific config file
        config_file = self.config_dir / f"{agent_name}_agent_config.json"
        config.save_to_file(config_file, {agent_name: config})
        return config_file

    def load_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Load an agent configuration."""
        config_file = self.config_dir / f"{agent_name}_agent_config.json"
        if not config_file.exists():
            return None

        configs = AgentConfig.load_from_file(config_file)
        return configs.get(agent_name)

    def list_agent_configs(self) -> List[str]:
        """List all available agent configurations."""
        configs = []
        for config_file in self.config_dir.glob("*_agent_config.json"):
            # Extract agent name from filename
            agent_name = config_file.stem.replace("_agent_config", "")
            configs.append(agent_name)
        return sorted(configs)

    def create_agent(self, agent_name: str, **overrides):
        """
        Create an agent instance from experiment configuration.

        Args:
            agent_name: Name of the agent configuration to load
            **overrides: Optional configuration overrides (e.g., prompt="custom:v2")

        Returns:
            Agent instance

        Raises:
            ValueError: If agent configuration is not found
        """
        # Load agent configuration
        config = self.load_agent_config(agent_name)
        if config is None:
            raise ValueError(f"Agent configuration '{agent_name}' not found")

        # Apply any overrides
        if overrides:
            # Create a copy of the config with overrides
            config_dict = {
                "agent_name": config.agent_name,
                "model_provider": config.model_provider,
                "model_family": config.model_family,
                "model_version": config.model_version,
                "prompt": config.prompt,
                "model_parameters": config.model_parameters.copy(),
                "provider_auth": config.provider_auth.copy(),
                "custom_configs": config.custom_configs.copy(),
                "tools": config.tools.copy(),
            }

            # Apply overrides
            for key, value in overrides.items():
                if key in config_dict:
                    config_dict[key] = value
                elif key in ["model_parameters", "provider_auth", "custom_configs"]:
                    if isinstance(value, dict):
                        config_dict[key].update(value)
                    else:
                        config_dict[key] = value

            config = AgentConfig.from_dict(config_dict)

        # Create agent using the Agent factory
        agent = Agent.create(config)

        # Log agent creation in current run context if available
        from dsat.scryptorum.core.decorators import get_current_run

        current_run = get_current_run()
        if current_run:
            current_run.log_agent_created(
                agent_name,
                {
                    "model_provider": config.model_provider,
                    "model_family": config.model_family,
                    "model_version": config.model_version,
                    "prompt": config.prompt,
                },
            )

        return agent

    def update_agent_config(self, agent_name: str, updates: Dict[str, any]) -> bool:
        """Update an existing agent configuration."""
        config = self.load_agent_config(agent_name)
        if config is None:
            return False

        # Create updated config
        config_dict = config.to_dict()
        for key, value in updates.items():
            if key in config_dict:
                config_dict[key] = value

        updated_config = AgentConfig.from_dict(config_dict)
        self.create_agent_config(agent_name, updated_config)
        return True

    def delete_agent_config(self, agent_name: str) -> bool:
        """Delete an agent configuration."""
        config_file = self.config_dir / f"{agent_name}_agent_config.json"
        if config_file.exists():
            config_file.unlink()
            return True
        return False

    def create_run(
        self, run_type: RunType = RunType.TRIAL, run_id: Optional[str] = None
    ):
        """Create an AgentRun with agent config snapshotting for milestone runs."""
        # Create AgentRun instead of base Run
        run = AgentRun(self.experiment_path, run_type, run_id)

        # For milestone runs, automatically snapshot agent configs
        if run_type == RunType.MILESTONE:
            run.snapshot_agent_configs(self.config_dir)

        return run

    def _update_experiment_metadata(self) -> None:
        """Create or update experiment metadata with agent information."""
        super()._update_experiment_metadata()

        # Add agent-specific metadata
        metadata_file = self.experiment_path / "experiment.json"

        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Add agent information
        metadata["agents_enabled"] = True
        metadata["agent_configs"] = self.list_agent_configs()

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
