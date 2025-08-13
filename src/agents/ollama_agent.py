import logging
import requests
from .agent import Agent, AgentConfig
from .agent_logger import CallTimer

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class OllamaAgent(Agent):
    """
    Ollama LLM agent for local model interactions.
    """

    def __init__(self, config: AgentConfig, base_url: str = "http://localhost:11434", logger: logging.Logger = None, prompts_dir=None):
        # Use provided logger or create a default one
        if logger is None:
            logger = logging.getLogger(__name__)
        
        super().__init__(config, logger, prompts_dir)
        
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests package is required for OllamaAgent")
        
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/generate"

    def invoke(self, user_prompt: str, system_prompt: str = None) -> str:
        """
        Send the prompts to Ollama and return the response.
        
        :param user_prompt: Specific user prompt
        :param system_prompt: Optional system prompt override. If None, loads from config via prompt manager.
        :return: Text of response
        """
        # Use model parameters from config, with defaults
        model_params = self.config.model_parameters or {}
        temperature = model_params.get("temperature", 0.0)
        
        # Use provided system prompt or load from prompt manager
        if system_prompt is None:
            system_prompt = self.get_system_prompt()
        
        # Prepare the prompt - combine system and user prompts
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            full_prompt = user_prompt
        
        # Prepare request data for logging
        request_data = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "model_parameters": {
                "temperature": temperature
            }
        }
        
        # Prepare the request payload
        payload = {
            "model": self.config.model_version,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            with CallTimer() as timer:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                # Parse the response
                response_data = response.json()
            
            self.logger.debug(f"Ollama raw response: {response_data}")
            
            if "response" in response_data:
                response_text = response_data["response"]
                self.logger.debug(f".. response: {len(response_text)} bytes / {len(response_text.split())} words")
                
                # Prepare response data for logging
                response_log_data = {
                    "content": response_text,
                    "tokens_used": {
                        "input": response_data.get("eval_count"),
                        "output": response_data.get("prompt_eval_count")
                    }
                }
                
                # Log the LLM call if logger is configured
                if self.call_logger:
                    self.call_logger.log_llm_call(
                        request_data=request_data,
                        response_data=response_log_data,
                        duration_ms=timer.duration_ms,
                        model_provider=self.config.model_provider,
                        model_version=self.config.model_version
                    )
                
                return response_text
            else:
                error_response = "ERROR - NO DATA"
                
                # Log error case
                if self.call_logger:
                    self.call_logger.log_llm_call(
                        request_data=request_data,
                        response_data={"content": error_response, "error": "No response in API response"},
                        duration_ms=timer.duration_ms,
                        model_provider=self.config.model_provider,
                        model_version=self.config.model_version
                    )
                
                return error_response
                
        except requests.exceptions.RequestException as e:
            # Log API errors
            if self.call_logger:
                self.call_logger.log_llm_call(
                    request_data=request_data,
                    response_data={"error": str(e), "error_type": type(e).__name__},
                    duration_ms=timer.duration_ms if 'timer' in locals() else 0,
                    model_provider=self.config.model_provider,
                    model_version=self.config.model_version
                )
            
            self.logger.error(f"Ollama API error: {str(e)}")
            raise
        except Exception as e:
            # Log unexpected errors
            if self.call_logger:
                self.call_logger.log_llm_call(
                    request_data=request_data,
                    response_data={"error": str(e), "error_type": type(e).__name__},
                    duration_ms=timer.duration_ms if 'timer' in locals() else 0,
                    model_provider=self.config.model_provider,
                    model_version=self.config.model_version
                )
            
            self.logger.error(f"Unexpected error in Ollama agent: {str(e)}")
            raise

    @property
    def model(self) -> str:
        return self.config.model_version