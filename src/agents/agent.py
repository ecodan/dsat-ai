import logging
import os
import json
from pathlib import Path
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .anthropic_agent import ClaudeLLMAgent
    from .vertex_agent import GoogleVertexAIAgent

from .prompts import PromptManager


@dataclass
class AgentConfig:
    """
    Agent Configuration System
    
    Agents are defined declaratively with configurations that can be stored in JSON or TOML files.
    The agent config file will be a dict and can have one or more agents defined by key=name value=config.
    
    This configuration system allows for:
    - Declarative agent setup with specific parameters
    - Storage in JSON or YAML files with multiple agents per file
    - Easy integration with different LLM providers
    - Flexible model parameters and authentication
    - Future support for MCP tools
    
    Required Fields:
    ---------------
    agent_name : str
        Unique name of agent within project context
    model_provider : str
        The hosting provider of model (defines which Agent sub-class is used)
        Examples: "anthropic", "openai", "google", "azure"
    model_family : str
        The overall model family - important when using a multi-model host
        Examples: "claude", "gpt", "gemini"
    model_version : str
        The specific model+version
        Examples: "claude-3-5-haiku-latest", "gpt-4o", "gemini-2.5-flash"
    prompt : str
        Prompt template name and version in format "name:version" or "name:latest"
        Examples: "assistant:v1", "assistant:2", "assistant:latest"
    
    Optional Fields:
    ---------------
    model_parameters : dict, optional
        Settings specific to the model (temperature, max_tokens, etc.)
    provider_auth : dict, optional
        Any authentication details needed for the host
        Examples: {"api_key": "sk-...", "project_id": "my-project"}
    custom_configs : dict, optional
        Additional custom configuration for this agent
    tools : list, optional
        FUTURE: List of all MCP tools available to the agent
    
    Usage:
    ------
    # Create from dictionary
    config = AgentConfig.from_dict({
        "agent_name": "my_assistant",
        "model_provider": "anthropic",
        "model_family": "claude",
        "model_version": "claude-3-5-haiku-latest",
        "prompt": "assistant:v1"
    })
    
    # Create agent using config
    agent = Agent.create_from_config(config)
    
    # Load from file
    configs = AgentConfig.load_from_file("agents.json")
    agent = Agent.create_from_config(configs["my_assistant"])
    """
    agent_name: str
    model_provider: str
    model_family: str
    model_version: str
    prompt: str
    model_parameters: Optional[Dict[str, Any]] = field(default_factory=dict)
    provider_auth: Optional[Dict[str, str]] = field(default_factory=dict)
    custom_configs: Optional[Dict[str, Any]] = field(default_factory=dict)
    tools: Optional[List[str]] = field(default_factory=list)  # FUTURE: MCP tools

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """
        Create an AgentConfig instance from a dictionary.
        
        :param config_dict: Dictionary containing configuration parameters
        :return: AgentConfig instance
        :raises ValueError: If required keys are missing or config_dict is invalid
        """
        if not isinstance(config_dict, dict):
            raise ValueError("config_dict must be a dictionary")
            
        required_keys = ["agent_name", "model_provider", "model_family", "model_version", "prompt"]
        for key in required_keys:
            if key not in config_dict:
                raise ValueError(f"Missing required key: {key} in config_dict")

        # Create a copy to avoid modifying the original
        config_data = config_dict.copy()
        
        # Ensure optional fields have proper defaults
        config_data.setdefault("model_parameters", {})
        config_data.setdefault("provider_auth", {})
        config_data.setdefault("custom_configs", {})
        config_data.setdefault("tools", [])
        
        return cls(**config_data)
    
    def parse_prompt(self) -> tuple[str, str]:
        """
        Parse the prompt field into name and version components.
        
        :return: Tuple of (prompt_name, prompt_version)
        """
        if ':' not in self.prompt:
            raise ValueError(f"Invalid prompt format: '{self.prompt}'. Expected format: 'name:version' or 'name:latest'")
        
        prompt_name, prompt_version = self.prompt.split(':', 1)
        return prompt_name.strip(), prompt_version.strip()
    
    @property
    def prompt_name(self) -> str:
        """
        Get the prompt name from the prompt field.
        
        :return: Prompt name
        """
        prompt_name, _ = self.parse_prompt()
        return prompt_name
    
    @property
    def prompt_version(self) -> str:
        """
        Get the prompt version from the prompt field.
        
        :return: Prompt version
        """
        _, prompt_version = self.parse_prompt()
        return prompt_version
        
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> Dict[str, 'AgentConfig']:
        """
        Load agent configurations from a JSON or TOML file.
        
        The file should contain a dictionary where keys are agent names and
        values are agent configuration dictionaries.
        
        :param file_path: Path to the configuration file
        :return: Dictionary mapping agent names to AgentConfig instances
        :raises FileNotFoundError: If the file doesn't exist
        :raises ValueError: If the file format is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif file_path.suffix.lower() in ['.toml', '.tml']:
                try:
                    import tomlkit
                    with open(file_path, 'r') as f:
                        data = tomlkit.load(f)
                except ImportError:
                    raise ImportError("tomlkit package is required to load TOML files. Install with: pip install tomlkit")
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: .json, .toml")
                
        except (json.JSONDecodeError, Exception) as e:
            raise ValueError(f"Invalid configuration file format: {e}")
            
        if not isinstance(data, dict):
            raise ValueError("Configuration file must contain a dictionary of agent configurations")
            
        # Convert each agent config dictionary to AgentConfig instance
        agent_configs = {}
        for agent_name, config_dict in data.items():
            if not isinstance(config_dict, dict):
                raise ValueError(f"Agent '{agent_name}' configuration must be a dictionary")
                
            # Ensure agent_name matches the key
            config_dict['agent_name'] = agent_name
            agent_configs[agent_name] = cls.from_dict(config_dict)
            
        return agent_configs

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the AgentConfig instance to a dictionary.
        
        :return: Dictionary representation of the AgentConfig suitable for serialization
        """
        return {
            "agent_name": self.agent_name,
            "model_provider": self.model_provider,
            "model_family": self.model_family,
            "model_version": self.model_version,
            "prompt": self.prompt,
            "model_parameters": self.model_parameters,
            "provider_auth": self.provider_auth,
            "custom_configs": self.custom_configs,
            "tools": self.tools
        }
        
    def save_to_file(self, file_path: Union[str, Path], configs: Dict[str, 'AgentConfig'] = None) -> None:
        """
        Save agent configuration(s) to a file.
        
        :param file_path: Path where to save the configuration file
        :param configs: Optional dictionary of multiple agent configs to save.
                       If None, saves only this config using its agent_name as key.
        """
        file_path = Path(file_path)
        
        if configs is None:
            configs = {self.agent_name: self}
            
        data = {name: config.to_dict() for name, config in configs.items()}
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif file_path.suffix.lower() in ['.toml', '.tml']:
            try:
                import tomlkit
                doc = tomlkit.document()
                for name, config_dict in data.items():
                    doc[name] = config_dict
                    
                with open(file_path, 'w') as f:
                    f.write(tomlkit.dumps(doc))
            except ImportError:
                raise ImportError("tomlkit package is required to save TOML files. Install with: pip install tomlkit")
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: .json, .toml")


class Agent(metaclass=ABCMeta):
    """
    Base class for all agents.
    """

    def __init__(self, config: AgentConfig, logger: logging.Logger, prompts_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the agent with configuration and optional logger.
        :param config: Agent configuration
        :param logger: Optional logger instance
        :param prompts_dir: Directory containing prompt TOML files. If None, defaults to ./prompts
        """
        self.config = config
        self.logger = logger
        
        # Initialize prompt manager
        if prompts_dir is None:
            prompts_dir = Path("prompts")
        elif isinstance(prompts_dir, str):
            prompts_dir = Path(prompts_dir)
        
        self.prompt_manager = PromptManager(prompts_dir)
        self._system_prompt = None  # Cached system prompt

    def get_system_prompt(self) -> Optional[str]:
        """
        Load system prompt from prompt manager based on config.
        Caches the prompt after first load.
        
        :return: System prompt text or None if not found
        """
        if self._system_prompt is not None:
            return self._system_prompt
            
        prompt_name = self.config.prompt_name
        prompt_version = self.config.prompt_version
        
        # Handle "latest" version or None
        if prompt_version == "latest" or prompt_version is None:
            prompt_version = None
            
        self._system_prompt = self.prompt_manager.get_prompt(prompt_name, prompt_version)
        
        if self._system_prompt is None:
            self.logger.warning(f"System prompt not found: {prompt_name}:{prompt_version or 'latest'}")
            
        return self._system_prompt

    @abstractmethod
    def invoke(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send the prompts to the LLM and return the response.
        
        :param user_prompt: Specific user prompt
        :param system_prompt: Optional system prompt override. If None, loads from config via prompt manager.
        :return: Text of response
        """
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """
        Return the model name.
        :return: model name
        """
        pass

    @classmethod
    def create(cls, config: AgentConfig, logger: logging.Logger = None, prompts_dir: Optional[Union[str, Path]] = None) -> 'Agent':
        """
        Factory method to create an agent instance from an AgentConfig.
        
        :param config: AgentConfig instance with all necessary configuration
        :param logger: Optional logger instance, will create default if None
        :param prompts_dir: Directory containing prompt TOML files. If None, defaults to ./prompts
        :return: Agent instance
        """
        if logger is None:
            logger = logging.getLogger(__name__)
            
        provider = config.model_provider.lower()
        
        if provider == "anthropic":
            # Check if anthropic is available
            try:
                from .anthropic_agent import ClaudeLLMAgent, ANTHROPIC_AVAILABLE
                if not ANTHROPIC_AVAILABLE:
                    raise ImportError("anthropic package is required for Anthropic provider")
            except ImportError:
                raise ImportError("anthropic package is required for Anthropic provider. Install with: pip install anthropic")
            
            # Get API key from provider_auth
            api_key = config.provider_auth.get("api_key")
            if not api_key:
                raise ValueError("api_key is required in provider_auth for Anthropic provider")
            
            return ClaudeLLMAgent(config=config, api_key=api_key, logger=logger, prompts_dir=prompts_dir)

        elif provider == "google":
            # Check if vertex AI is available
            try:
                from .vertex_agent import GoogleVertexAIAgent, VERTEX_AI_AVAILABLE
                if not VERTEX_AI_AVAILABLE:
                    raise ImportError("google-cloud-aiplatform package is required for Google provider")
            except ImportError:
                raise ImportError("google-cloud-aiplatform package is required for Google provider. Install with: pip install google-cloud-aiplatform")
            
            # Get required auth parameters
            project_id = config.provider_auth.get("project_id")
            location = config.provider_auth.get("location", "us-central1")
            
            if not project_id:
                raise ValueError("project_id is required in provider_auth for Google provider")
            
            return GoogleVertexAIAgent(config=config, project_id=project_id, location=location, logger=logger, prompts_dir=prompts_dir)

        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: anthropic, google")
