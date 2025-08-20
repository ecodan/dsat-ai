import logging
import requests
import json
from typing import AsyncGenerator
from .agent import Agent, AgentConfig
from .agent_logger import CallTimer
from ..cli.memory import TokenCounter

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class OllamaAgent(Agent):
    """
    Ollama LLM agent for local model interactions.
    """

    def __init__(
        self,
        config: AgentConfig,
        base_url: str = "http://localhost:11434",
        logger: logging.Logger = None,
        prompts_dir=None,
    ):
        # Use provided logger or create a default one
        if logger is None:
            logger = logging.getLogger(__name__)

        super().__init__(config, logger, prompts_dir)

        if not REQUESTS_AVAILABLE:
            raise ImportError("requests package is required for OllamaAgent")

        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/generate"

    def invoke(self, user_prompt: str, system_prompt: str = None, history=None) -> str:
        """
        Send the prompts to Ollama and return the response.

        :param user_prompt: Specific user prompt
        :param system_prompt: Optional system prompt override. If None, loads from config via prompt manager.
        :param history: Optional conversation history for context
        :return: Text of response
        """
        # Use model parameters from config, with defaults
        model_params = self.config.model_parameters or {}
        temperature = model_params.get("temperature", 0.0)

        # Use provided system prompt or load from prompt manager
        if system_prompt is None:
            system_prompt = self.get_system_prompt()

        # Build conversation context with history
        full_prompt = self._build_conversation_context(system_prompt, history, user_prompt)

        # Prepare request data for logging
        request_data = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "history_length": len(history) if history else 0,
            "full_prompt_tokens": TokenCounter.estimate_tokens(full_prompt),
            "model_parameters": {"temperature": temperature},
        }

        # Prepare the request payload
        payload = {
            "model": self.config.model_version,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            with CallTimer() as timer:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()

                # Parse the response
                response_data = response.json()

            self.logger.debug(f"Ollama raw response: {response_data}")

            if "response" in response_data:
                response_text = response_data["response"]
                self.logger.debug(
                    f".. response: {len(response_text)} bytes / {len(response_text.split())} words"
                )

                # Prepare response data for logging
                response_log_data = {
                    "content": response_text,
                    "tokens_used": {
                        "input": response_data.get("eval_count"),
                        "output": response_data.get("prompt_eval_count"),
                    },
                }

                # Log the LLM call if logger is configured
                if self.call_logger:
                    self.call_logger.log_llm_call(
                        request_data=request_data,
                        response_data=response_log_data,
                        duration_ms=timer.duration_ms,
                        model_provider=self.config.model_provider,
                        model_version=self.config.model_version,
                    )

                return response_text
            else:
                error_response = "ERROR - NO DATA"

                # Log error case
                if self.call_logger:
                    self.call_logger.log_llm_call(
                        request_data=request_data,
                        response_data={
                            "content": error_response,
                            "error": "No response in API response",
                        },
                        duration_ms=timer.duration_ms,
                        model_provider=self.config.model_provider,
                        model_version=self.config.model_version,
                    )

                return error_response

        except requests.exceptions.RequestException as e:
            # Log API errors
            if self.call_logger:
                self.call_logger.log_llm_call(
                    request_data=request_data,
                    response_data={"error": str(e), "error_type": type(e).__name__},
                    duration_ms=timer.duration_ms if "timer" in locals() else 0,
                    model_provider=self.config.model_provider,
                    model_version=self.config.model_version,
                )

            self.logger.error(f"Ollama API error: {str(e)}")
            raise
        except Exception as e:
            # Log unexpected errors
            if self.call_logger:
                self.call_logger.log_llm_call(
                    request_data=request_data,
                    response_data={"error": str(e), "error_type": type(e).__name__},
                    duration_ms=timer.duration_ms if "timer" in locals() else 0,
                    model_provider=self.config.model_provider,
                    model_version=self.config.model_version,
                )

            self.logger.error(f"Unexpected error in Ollama agent: {str(e)}")
            raise

    async def invoke_async(
        self, user_prompt: str, system_prompt: str = None, history=None
    ) -> AsyncGenerator[str, None]:
        """
        Send the prompts to Ollama and return a streaming async generator of response tokens.

        :param user_prompt: Specific user prompt
        :param system_prompt: Optional system prompt override. If None, loads from config via prompt manager.
        :return: AsyncGenerator yielding response text chunks
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp package is required for async streaming in OllamaAgent. Install with: pip install aiohttp"
            )

        # Use model parameters from config, with defaults
        model_params = self.config.model_parameters or {}
        temperature = model_params.get("temperature", 0.0)

        # Use provided system prompt or load from prompt manager
        if system_prompt is None:
            system_prompt = self.get_system_prompt()

        # Build conversation context with history
        full_prompt = self._build_conversation_context(system_prompt, history, user_prompt)

        # Prepare request data for logging
        request_data = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "history_length": len(history) if history else 0,
            "full_prompt_tokens": TokenCounter.estimate_tokens(full_prompt),
            "model_parameters": {"temperature": temperature},
        }

        # Prepare the request payload
        payload = {
            "model": self.config.model_version,
            "prompt": full_prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }

        try:
            with CallTimer() as timer:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        response.raise_for_status()

                        # Collect chunks for logging
                        response_chunks = []

                        # Process streaming NDJSON response
                        async for line in response.content:
                            line = line.decode("utf-8").strip()
                            if not line:
                                continue

                            try:
                                chunk_data = json.loads(line)
                                if "response" in chunk_data:
                                    text_chunk = chunk_data["response"]
                                    response_chunks.append(text_chunk)
                                    yield text_chunk

                                # Check if streaming is done
                                if chunk_data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                # Skip malformed JSON lines
                                continue

                # After streaming is complete, log the full response
                full_response = "".join(response_chunks)
                self.logger.debug(
                    f"Ollama async response complete: {len(full_response)} bytes / {len(full_response.split())} words"
                )

                # Prepare response data for logging
                response_log_data = {
                    "content": full_response,
                    "tokens_used": {
                        "input": None,  # Token usage not available in streaming
                        "output": None,
                    },
                }

                # Log the LLM call if logger is configured
                if self.call_logger:
                    self.call_logger.log_llm_call(
                        request_data=request_data,
                        response_data=response_log_data,
                        duration_ms=timer.duration_ms,
                        model_provider=self.config.model_provider,
                        model_version=self.config.model_version,
                    )

        except aiohttp.ClientError as e:
            # Log API errors
            if self.call_logger:
                self.call_logger.log_llm_call(
                    request_data=request_data,
                    response_data={"error": str(e), "error_type": type(e).__name__},
                    duration_ms=timer.duration_ms if "timer" in locals() else 0,
                    model_provider=self.config.model_provider,
                    model_version=self.config.model_version,
                )

            self.logger.error(f"Ollama API error in streaming: {str(e)}")
            raise
        except Exception as e:
            # Log unexpected errors
            if self.call_logger:
                self.call_logger.log_llm_call(
                    request_data=request_data,
                    response_data={"error": str(e), "error_type": type(e).__name__},
                    duration_ms=timer.duration_ms if "timer" in locals() else 0,
                    model_provider=self.config.model_provider,
                    model_version=self.config.model_version,
                )

            self.logger.error(f"Unexpected error in Ollama async agent: {str(e)}")
            raise

    def _build_conversation_context(self, system_prompt, history, user_prompt):
        """
        Build conversation context for Ollama from system prompt, history, and current user prompt.
        
        :param system_prompt: System prompt string or None
        :param history: List of ConversationMessage objects or None
        :param user_prompt: Current user prompt string
        :return: Full conversation context as a string
        """
        context_parts = []
        
        # Add system prompt if provided
        if system_prompt:
            context_parts.append(system_prompt)
        
        # Add conversation history if provided
        if history:
            for msg in history:
                # Format each message clearly
                role_label = "Human" if msg.role == "user" else "Assistant"
                context_parts.append(f"{role_label}: {msg.content}")
        
        # Add current user prompt
        context_parts.append(f"Human: {user_prompt}")
        
        # Join all parts with double newlines for clear separation
        return "\n\n".join(context_parts)

    @property
    def model(self) -> str:
        return self.config.model_version
