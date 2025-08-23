import logging
import os
import json
from pathlib import Path
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING, AsyncGenerator

if TYPE_CHECKING:
    from ..cli.memory import ConversationMessage

# Entry points for plugin discovery
try:
    from importlib.metadata import entry_points

    ENTRY_POINTS_AVAILABLE = True
except ImportError:
    try:
        from importlib_metadata import entry_points

        ENTRY_POINTS_AVAILABLE = True
    except ImportError:
        ENTRY_POINTS_AVAILABLE = False

from .prompts import PromptManager
from .agent_logger import AgentCallLogger


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
    - Model definition separation via _models section and model_id references
    - Future support for MCP tools

    Required Fields:
    ---------------
    agent_name : str
        Unique name of agent within project context
    model_provider : str
        The hosting provider of model (defines which Agent sub-class is used)
        Examples: "anthropic", "openai", "google", "azure"
        NOTE: Can be replaced by model_id reference to _models section
    model_family : str
        The overall model family - important when using a multi-model host
        Examples: "claude", "gpt", "gemini"
        NOTE: Can be replaced by model_id reference to _models section
    model_version : str
        The specific model+version
        Examples: "claude-3-5-haiku-latest", "gpt-4o", "gemini-2.5-flash"
        NOTE: Can be replaced by model_id reference to _models section
    prompt : str
        Prompt template name and version in format "name:version" or "name:latest"
        Examples: "assistant:v1", "assistant:2", "assistant:latest"

    Optional Fields:
    ---------------
    model_parameters : dict, optional
        Settings specific to the model (temperature, max_tokens, etc.)
        When using model_id, merges with model definition (agent config takes precedence)
    provider_auth : dict, optional
        Any authentication details needed for the host
        Examples: {"api_key": "sk-...", "project_id": "my-project"}
        When using model_id, merges with model definition (agent config takes precedence)
    custom_configs : dict, optional
        Additional custom configuration for this agent. Supports logging configuration:
        {
            "logging": {
                "enabled": True,           # Enable/disable LLM call logging
                "mode": "standard",        # "standard", "jsonl_file", "callback", "disabled"
                "file_path": "path.jsonl", # Required for jsonl_file mode
                "callback": func,          # Required for callback mode
                "level": "standard"        # "minimal" or "standard" detail level
            }
        }
    tools : list, optional
        FUTURE: List of all MCP tools available to the agent
    stream : bool, optional
        Enable token streaming for supported providers (default: False)
    memory_enabled : bool, optional
        Enable/disable chat history persistence (default: True)
    max_memory_tokens : int, optional
        Maximum tokens to keep in conversation memory (default: 8000)
    response_truncate_length : int, optional
        Truncate responses longer than this character count (default: 1000)

    Model Separation with _models section:
    -------------------------------------
    Configuration files can include a special "_models" section to define reusable
    model configurations. Agents can then reference these via "model_id" instead
    of defining model fields directly:

    {
        "_models": {
            "gpt4": {
                "model_provider": "openai",
                "model_family": "gpt",
                "model_version": "gpt-4o",
                "model_parameters": {"temperature": 0.7},
                "provider_auth": {"api_key": "sk-..."}
            }
        },
        "my_agent": {
            "model_id": "gpt4",          # References model definition
            "prompt": "assistant:v1",
            "model_parameters": {"temperature": 0.9}  # Overrides model default
        }
    }

    Usage:
    ------
    # Create from dictionary (direct model config)
    config = AgentConfig.from_dict({
        "agent_name": "my_assistant",
        "model_provider": "anthropic",
        "model_family": "claude",
        "model_version": "claude-3-5-haiku-latest",
        "prompt": "assistant:v1"
    })

    # Load from file with _models section
    configs = AgentConfig.load_from_file("agents.json")
    agent = Agent.create_from_config(configs["my_assistant"])

    # Enable LLM call logging to JSONL file
    config_with_logging = AgentConfig.from_dict({
        "agent_name": "my_assistant",
        "model_provider": "anthropic",
        "model_family": "claude",
        "model_version": "claude-3-5-haiku-latest",
        "prompt": "assistant:v1",
        "custom_configs": {
            "logging": {
                "enabled": True,
                "mode": "jsonl_file",
                "file_path": "./logs/agent_calls.jsonl",
                "level": "standard"
            }
        }
    })
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
    prompts_dir: Optional[str] = None  # Optional prompts directory override
    stream: bool = False  # Enable token streaming for supported providers
    # Memory configuration
    memory_enabled: bool = True  # Enable/disable chat history persistence
    max_memory_tokens: int = 8000  # Maximum tokens to keep in conversation memory
    response_truncate_length: int = 1000  # Truncate responses longer than this

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        """
        Create an AgentConfig instance from a dictionary.

        :param config_dict: Dictionary containing configuration parameters
        :return: AgentConfig instance
        :raises ValueError: If required keys are missing or config_dict is invalid
        """
        if not isinstance(config_dict, dict):
            raise ValueError("config_dict must be a dictionary")

        required_keys = [
            "agent_name",
            "model_provider",
            "model_family",
            "model_version",
            "prompt",
        ]
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
        config_data.setdefault("prompts_dir", None)
        config_data.setdefault("stream", False)
        config_data.setdefault("memory_enabled", True)
        config_data.setdefault("max_memory_tokens", 8000)
        config_data.setdefault("response_truncate_length", 1000)

        return cls(**config_data)

    def parse_prompt(self) -> tuple[str, str]:
        """
        Parse the prompt field into name and version components.

        :return: Tuple of (prompt_name, prompt_version)
        """
        if ":" not in self.prompt:
            raise ValueError(
                f"Invalid prompt format: '{self.prompt}'. Expected format: 'name:version' or 'name:latest'"
            )

        prompt_name, prompt_version = self.prompt.split(":", 1)
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
    def load_from_file(cls, file_path: Union[str, Path]) -> Dict[str, "AgentConfig"]:
        """
        Load agent configurations from a JSON or TOML file.

        The file should contain a dictionary where keys are agent names and
        values are agent configuration dictionaries. Optionally, a special
        "_models" key can define reusable model configurations that agents
        can reference via "model_id".

        :param file_path: Path to the configuration file
        :return: Dictionary mapping agent names to AgentConfig instances
        :raises FileNotFoundError: If the file doesn't exist
        :raises ValueError: If the file format is invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            if file_path.suffix.lower() == ".json":
                with open(file_path, "r") as f:
                    data = json.load(f)
            elif file_path.suffix.lower() in [".toml", ".tml"]:
                try:
                    import tomlkit

                    with open(file_path, "r") as f:
                        data = tomlkit.load(f)
                except ImportError:
                    raise ImportError(
                        "tomlkit package is required to load TOML files. Install with: pip install tomlkit"
                    )
            else:
                raise ValueError(
                    f"Unsupported file format: {file_path.suffix}. Supported formats: .json, .toml"
                )

        except (json.JSONDecodeError, Exception) as e:
            raise ValueError(f"Invalid configuration file format: {e}")

        if not isinstance(data, dict):
            raise ValueError(
                "Configuration file must contain a dictionary of agent configurations"
            )

        # Extract model definitions if present
        models = data.pop("_models", {})
        if models and not isinstance(models, dict):
            raise ValueError("_models section must be a dictionary")

        # Validate model definitions
        model_required_fields = ["model_provider", "model_family", "model_version"]
        for model_id, model_config in models.items():
            if not isinstance(model_config, dict):
                raise ValueError(
                    f"Model '{model_id}' configuration must be a dictionary"
                )
            for required_field in model_required_fields:
                if required_field not in model_config:
                    raise ValueError(
                        f"Model '{model_id}' missing required field: {required_field}"
                    )

        # Convert each agent config dictionary to AgentConfig instance
        agent_configs = {}
        for agent_name, config_dict in data.items():
            if not isinstance(config_dict, dict):
                raise ValueError(
                    f"Agent '{agent_name}' configuration must be a dictionary"
                )

            # Handle model_id reference if present
            if "model_id" in config_dict:
                model_id = config_dict.pop("model_id")
                if model_id not in models:
                    raise ValueError(
                        f"Agent '{agent_name}' references unknown model_id: '{model_id}'"
                    )

                # Check for conflicts between model_id and direct model fields
                model_fields = ["model_provider", "model_family", "model_version"]
                conflicts = [field for field in model_fields if field in config_dict]
                if conflicts:
                    raise ValueError(
                        f"Agent '{agent_name}' has model_id='{model_id}' but also defines: {', '.join(conflicts)}. Use either model_id or direct model fields, not both."
                    )

                # Merge model configuration into agent config
                model_config = models[model_id].copy()

                # Agent config can override model_parameters and provider_auth
                if "model_parameters" in config_dict:
                    # Merge model_parameters, with agent config taking precedence
                    agent_params = config_dict.pop("model_parameters")
                    model_params = model_config.get("model_parameters", {})
                    merged_params = {**model_params, **agent_params}
                    config_dict["model_parameters"] = merged_params

                if "provider_auth" in config_dict:
                    # Merge provider_auth, with agent config taking precedence
                    agent_auth = config_dict.pop("provider_auth")
                    model_auth = model_config.get("provider_auth", {})
                    merged_auth = {**model_auth, **agent_auth}
                    config_dict["provider_auth"] = merged_auth

                # Add the model config fields to agent config
                for key, value in model_config.items():
                    if key not in config_dict:  # Don't override if already set
                        config_dict[key] = value

            # Ensure agent_name matches the key
            config_dict["agent_name"] = agent_name
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
            "tools": self.tools,
            "prompts_dir": self.prompts_dir,
            "stream": self.stream,
        }

    def save_to_file(
        self, file_path: Union[str, Path], configs: Dict[str, "AgentConfig"] = None
    ) -> None:
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

        if file_path.suffix.lower() == ".json":
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        elif file_path.suffix.lower() in [".toml", ".tml"]:
            try:
                import tomlkit

                doc = tomlkit.document()
                for name, config_dict in data.items():
                    doc[name] = config_dict

                with open(file_path, "w") as f:
                    f.write(tomlkit.dumps(doc))
            except ImportError:
                raise ImportError(
                    "tomlkit package is required to save TOML files. Install with: pip install tomlkit"
                )
        else:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. Supported formats: .json, .toml"
            )


class Agent(metaclass=ABCMeta):
    """
    Base class for all agents.
    """

    # Registry for custom providers (class variable)
    _custom_providers: Dict[str, type] = {}

    def __init__(
        self,
        config: AgentConfig,
        logger: logging.Logger,
        prompts_dir: Optional[Union[str, Path]] = None,
    ):
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

        # Initialize agent call logger
        self.call_logger = self._setup_call_logger()

    def get_system_prompt(self) -> Optional[str]:
        """
        Load system prompt from prompt manager based on config.
        Caches the prompt after first load.

        :return: System prompt text or None if not found
        """
        if self._system_prompt is not None:
            return self._system_prompt

        # Parse prompt field (format: "name:version" or "name:latest")
        prompt_spec = self.config.prompt
        if ":" in prompt_spec:
            prompt_name, prompt_version = prompt_spec.split(":", 1)
        else:
            prompt_name, prompt_version = prompt_spec, "latest"

        # Handle "latest" version or None
        if prompt_version == "latest" or prompt_version is None:
            prompt_version = None

        self._system_prompt = self.prompt_manager.get_prompt(
            prompt_name, prompt_version
        )

        if self._system_prompt is None:
            self.logger.warning(
                f"System prompt not found: {prompt_name}:{prompt_version or 'latest'}"
            )

        return self._system_prompt

    @classmethod
    def register_provider(cls, provider_name: str, agent_class: type):
        """
        Register a custom provider agent class.

        :param provider_name: Name of the provider (e.g., "openai", "azure")
        :param agent_class: Agent class that extends Agent
        """
        if not issubclass(agent_class, Agent):
            raise ValueError("Agent class must extend Agent base class")

        cls._custom_providers[provider_name.lower()] = agent_class

    @classmethod
    def _discover_plugin_providers(cls) -> Dict[str, type]:
        """
        Discover agent providers via entry points.

        :return: Dictionary mapping provider names to agent classes
        """
        providers = {}

        if not ENTRY_POINTS_AVAILABLE:
            return providers

        try:
            # Look for entry points in the 'dsat.providers' group
            eps = entry_points()

            # Handle different versions of entry_points API
            if hasattr(eps, "select"):
                # Python 3.10+ style
                provider_eps = eps.select(group="dsat.providers")
            else:
                # Older style
                provider_eps = eps.get("dsat.providers", [])

            for ep in provider_eps:
                try:
                    agent_class = ep.load()
                    if issubclass(agent_class, Agent):
                        providers[ep.name.lower()] = agent_class
                    else:
                        logging.getLogger(__name__).warning(
                            f"Plugin provider '{ep.name}' does not extend Agent class"
                        )
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        f"Failed to load plugin provider '{ep.name}': {e}"
                    )

        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Error discovering plugin providers: {e}"
            )

        return providers

    @classmethod
    def get_available_providers(cls) -> Dict[str, str]:
        """
        Get all available providers (built-in and plugins).

        :return: Dictionary mapping provider names to their source
        """
        providers = {}

        # Built-in providers
        built_in = ["anthropic", "google", "ollama", "litellm"]
        for provider in built_in:
            providers[provider] = "built-in"

        # Custom registered providers
        for provider in cls._custom_providers:
            providers[provider] = "registered"

        # Plugin providers
        plugin_providers = cls._discover_plugin_providers()
        for provider in plugin_providers:
            providers[provider] = "plugin"

        return providers

    def _setup_call_logger(self) -> Optional[AgentCallLogger]:
        """
        Setup the agent call logger based on configuration and environment variables.

        :return: AgentCallLogger instance or None if logging disabled
        """
        # Check environment variables first (highest priority)
        env_enabled = os.getenv("DSAT_AGENT_LOGGING_ENABLED", "").lower() in (
            "true",
            "1",
            "yes",
        )
        env_mode = os.getenv("DSAT_AGENT_LOGGING_MODE", "").lower()
        env_file_path = os.getenv("DSAT_AGENT_LOGGING_FILE_PATH")
        env_level = os.getenv("DSAT_AGENT_LOGGING_LEVEL", "").lower()

        # Get logging config from agent config
        logging_config = self.config.custom_configs.get("logging", {})

        # Environment variables override config settings
        if env_enabled:
            logging_config = logging_config.copy()  # Don't modify original
            logging_config["enabled"] = True
            if env_mode:
                logging_config["mode"] = env_mode
            if env_file_path:
                logging_config["file_path"] = env_file_path
            if env_level:
                logging_config["level"] = env_level

        return AgentCallLogger.create_from_config(
            self.config.agent_name, logging_config
        )

    @abstractmethod
    def invoke(self, user_prompt: str, system_prompt: Optional[str] = None, 
              history: Optional[List["ConversationMessage"]] = None) -> str:
        """
        Send the prompts to the LLM and return the response.

        :param user_prompt: Specific user prompt
        :param system_prompt: Optional system prompt override. If None, loads from config via prompt manager.
        :param history: Optional conversation history for context
        :return: Text of response
        """
        pass

    @abstractmethod
    async def invoke_async(
        self, user_prompt: str, system_prompt: Optional[str] = None,
        history: Optional[List["ConversationMessage"]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Send the prompts to the LLM and return a streaming async generator of response tokens.

        :param user_prompt: Specific user prompt
        :param system_prompt: Optional system prompt override. If None, loads from config via prompt manager.
        :param history: Optional conversation history for context
        :return: AsyncGenerator yielding response text chunks
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
    def create(
        cls,
        config: AgentConfig,
        logger: logging.Logger = None,
        prompts_dir: Optional[Union[str, Path]] = None,
    ) -> "Agent":
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
        logger.info(
            f"Creating agent for provider: {provider} with model family: {config.model_family} and version: {config.model_version}"
        )

        # Check custom registered providers first
        if provider in cls._custom_providers:
            agent_class = cls._custom_providers[provider]
            return agent_class(config=config, logger=logger, prompts_dir=prompts_dir)
        else:
            logger.debug(f"No custom provider registered for: {provider}")

        # Check plugin providers
        plugin_providers = cls._discover_plugin_providers()
        if provider in plugin_providers:
            agent_class = plugin_providers[provider]
            return agent_class(config=config, logger=logger, prompts_dir=prompts_dir)
        else:
            logger.debug(f"No plugin provider found for: {provider}")

        # Fall back to built-in providers
        if provider == "anthropic":
            # Check if anthropic is available
            try:
                from .anthropic_agent import ClaudeLLMAgent, ANTHROPIC_AVAILABLE

                if not ANTHROPIC_AVAILABLE:
                    raise ImportError(
                        "anthropic package is required for Anthropic provider"
                    )
            except ImportError:
                raise ImportError(
                    "anthropic package is required for Anthropic provider. Install with: pip install anthropic"
                )

            # Get API key from provider_auth
            api_key = config.provider_auth.get("api_key")
            if not api_key:
                raise ValueError(
                    "api_key is required in provider_auth for Anthropic provider"
                )

            return ClaudeLLMAgent(
                config=config, api_key=api_key, logger=logger, prompts_dir=prompts_dir
            )

        elif provider == "google":
            # Check if vertex AI is available
            try:
                from .vertex_agent import GoogleVertexAIAgent, VERTEX_AI_AVAILABLE

                if not VERTEX_AI_AVAILABLE:
                    raise ImportError(
                        "google-cloud-aiplatform package is required for Google provider"
                    )
            except ImportError:
                raise ImportError(
                    "google-cloud-aiplatform package is required for Google provider. Install with: pip install google-cloud-aiplatform"
                )

            # Get required auth parameters
            project_id = config.provider_auth.get("project_id")
            location = config.provider_auth.get("location", "us-central1")

            if not project_id:
                raise ValueError(
                    "project_id is required in provider_auth for Google provider"
                )

            return GoogleVertexAIAgent(
                config=config,
                project_id=project_id,
                location=location,
                logger=logger,
                prompts_dir=prompts_dir,
            )

        elif provider == "ollama":
            # Check if requests is available
            try:
                from .ollama_agent import OllamaAgent, REQUESTS_AVAILABLE

                if not REQUESTS_AVAILABLE:
                    raise ImportError(
                        "requests package is required for Ollama provider"
                    )
            except ImportError:
                raise ImportError(
                    "requests package is required for Ollama provider. Install with: pip install requests"
                )

            # Get optional base URL
            base_url = config.provider_auth.get("base_url", "http://localhost:11434")

            return OllamaAgent(
                config=config, base_url=base_url, logger=logger, prompts_dir=prompts_dir
            )

        elif provider == "litellm":
            # Check if LiteLLM is available
            try:
                from .litellm_agent import LiteLLMAgent, LITELLM_AVAILABLE

                if not LITELLM_AVAILABLE:
                    raise ImportError(
                        "litellm package is required for LiteLLM provider"
                    )
            except ImportError:
                raise ImportError(
                    "litellm package is required for LiteLLM provider. Install with: pip install litellm"
                )

            return LiteLLMAgent(
                config=config, logger=logger, prompts_dir=prompts_dir
            )

        else:
            # Get available providers for error message
            available_providers = list(cls.get_available_providers().keys())
            raise ValueError(
                f"Unsupported provider: {provider}. Available providers: {', '.join(available_providers)}"
            )

    @classmethod
    def create_from_config(
        cls,
        config_file: Union[str, Path],
        agent_name: str,
        logger: logging.Logger = None,
        prompts_dir: Optional[Union[str, Path]] = None,
    ) -> "Agent":
        """
        Convenience method to create an agent from a config file and agent name.

        :param config_file: Path to JSON config file containing agent configurations
        :param agent_name: Name of the agent to load from the config file
        :param logger: Optional logger instance, will create default if None
        :param prompts_dir: Directory containing prompt TOML files. If None, defaults to ./prompts
        :return: Agent instance
        """
        configs = AgentConfig.load_from_file(config_file)
        if agent_name not in configs:
            raise ValueError(
                f"Agent '{agent_name}' not found in config file {config_file}"
            )

        return cls.create(configs[agent_name], logger, prompts_dir)

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        logger: logging.Logger = None,
        prompts_dir: Optional[Union[str, Path]] = None,
    ) -> "Agent":
        """
        Convenience method to create an agent from a configuration dictionary.

        :param config_dict: Dictionary containing agent configuration
        :param logger: Optional logger instance, will create default if None
        :param prompts_dir: Directory containing prompt TOML files. If None, defaults to ./prompts
        :return: Agent instance
        """
        config = AgentConfig.from_dict(config_dict)
        return cls.create(config, logger, prompts_dir)
