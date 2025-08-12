import logging
from .agent import Agent, AgentConfig

try:
    from anthropic import Anthropic, APIStatusError, APIConnectionError

    ANTHROPIC_AVAILABLE = True
except ImportError:
    Anthropic = None
    APIStatusError = None
    APIConnectionError = None
    ANTHROPIC_AVAILABLE = False



class ClaudeLLMAgent(Agent):
    """
    Anthropic Claude LLM agent.
    """

    def __init__(self, config: AgentConfig = None, api_key: str = None, model: str = None, logger: logging.Logger = None, prompts_dir = None):
        # Support both old API (api_key, model) and new API (config)
        if config is None:
            # Backward compatibility - create config from parameters
            if api_key is None or model is None:
                raise ValueError("Either config must be provided, or both api_key and model must be provided")
                
            config = AgentConfig(
                agent_name="claude",
                model_provider="anthropic",
                model_family="claude", 
                model_version=model,
                prompt="default:v1",
                provider_auth={"api_key": api_key}
            )
        else:
            # Use provided config, but allow api_key override for backward compatibility
            if api_key is not None:
                config.provider_auth["api_key"] = api_key
        
        # Use provided logger or create a default one
        if logger is None:
            logger = logging.getLogger(__name__)
        
        super().__init__(config, logger, prompts_dir)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required for ClaudeLLMAgent")
        
        # Get API key from config
        api_key_from_config = config.provider_auth.get("api_key")
        if not api_key_from_config:
            raise ValueError("api_key is required in provider_auth for ClaudeLLMAgent")
            
        self.client = Anthropic(api_key=api_key_from_config)

    def invoke(self, user_prompt: str, system_prompt: str = None) -> str:
        try:
            # Use model parameters from config, with defaults
            model_params = self.config.model_parameters or {}
            max_tokens = model_params.get("max_tokens", 4096)
            temperature = model_params.get("temperature", 0.0)
            
            # Use provided system prompt or load from prompt manager
            if system_prompt is None:
                system_prompt = self.get_system_prompt()
            
            response = self.client.messages.create(
                model=self.config.model_version,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            self.logger.debug(f"Claude raw response: {response.content}")
            if len(response.content) == 1:
                self.logger.debug(f".. response: {len(response.content[0].text)} bytes / {len(response.content[0].text.split())} words")
                return response.content[0].text
            else:
                return "ERROR - NO DATA"
        except (APIStatusError, APIConnectionError) as e:
            self.logger.error(f"Claude API error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in Claude agent: {str(e)}")
            raise

    @property
    def model(self) -> str:
        return self.config.model_version

