import logging
from typing import AsyncGenerator
from .agent import Agent, AgentConfig
from .agent_logger import CallTimer

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

    def __init__(
        self,
        config: AgentConfig = None,
        api_key: str = None,
        model: str = None,
        logger: logging.Logger = None,
        prompts_dir=None,
    ):
        # Support both old API (api_key, model) and new API (config)
        if config is None:
            # Backward compatibility - create config from parameters
            if api_key is None or model is None:
                raise ValueError(
                    "Either config must be provided, or both api_key and model must be provided"
                )

            config = AgentConfig(
                agent_name="claude",
                model_provider="anthropic",
                model_family="claude",
                model_version=model,
                prompt="default:v1",
                provider_auth={"api_key": api_key},
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

    def invoke(self, user_prompt: str, system_prompt: str = None, history=None) -> str:
        # Use model parameters from config, with defaults
        model_params = self.config.model_parameters or {}
        max_tokens = model_params.get("max_tokens", 4096)
        temperature = model_params.get("temperature", 0.0)

        # Use provided system prompt or load from prompt manager
        if system_prompt is None:
            system_prompt = self.get_system_prompt()

        # Build messages array from history and current prompt
        messages = self._build_messages_from_history(history, user_prompt)

        # Prepare request data for logging
        request_data = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "model_parameters": {"max_tokens": max_tokens, "temperature": temperature},
            "history_length": len(history) if history else 0,
        }

        try:
            with CallTimer() as timer:
                response = self.client.messages.create(
                    model=self.config.model_version,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=messages,
                )

            self.logger.debug(f"Claude raw response: {response.content}")

            if len(response.content) == 1:
                response_text = response.content[0].text
                self.logger.debug(
                    f".. response: {len(response_text)} bytes / {len(response_text.split())} words"
                )

                # Prepare response data for logging
                response_data = {
                    "content": response_text,
                    "tokens_used": {
                        "input": (
                            getattr(response.usage, "input_tokens", None)
                            if hasattr(response, "usage")
                            else None
                        ),
                        "output": (
                            getattr(response.usage, "output_tokens", None)
                            if hasattr(response, "usage")
                            else None
                        ),
                    },
                }

                # Log the LLM call if logger is configured
                if self.call_logger:
                    self.call_logger.log_llm_call(
                        request_data=request_data,
                        response_data=response_data,
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
                            "error": "No content in response",
                        },
                        duration_ms=timer.duration_ms,
                        model_provider=self.config.model_provider,
                        model_version=self.config.model_version,
                    )

                return error_response

        except (APIStatusError, APIConnectionError) as e:
            # Log API errors
            if self.call_logger:
                self.call_logger.log_llm_call(
                    request_data=request_data,
                    response_data={"error": str(e), "error_type": type(e).__name__},
                    duration_ms=timer.duration_ms if "timer" in locals() else 0,
                    model_provider=self.config.model_provider,
                    model_version=self.config.model_version,
                )

            self.logger.error(f"Claude API error: {str(e)}")
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

            self.logger.error(f"Unexpected error in Claude agent: {str(e)}")
            raise

    async def invoke_async(
        self, user_prompt: str, system_prompt: str = None, history=None
    ) -> AsyncGenerator[str, None]:
        """
        Send the prompts to Claude and return a streaming async generator of response tokens.

        :param user_prompt: Specific user prompt
        :param system_prompt: Optional system prompt override. If None, loads from config via prompt manager.
        :return: AsyncGenerator yielding response text chunks
        """
        # Use model parameters from config, with defaults
        model_params = self.config.model_parameters or {}
        max_tokens = model_params.get("max_tokens", 4096)
        temperature = model_params.get("temperature", 0.0)

        # Use provided system prompt or load from prompt manager
        if system_prompt is None:
            system_prompt = self.get_system_prompt()

        # Build messages array from history and current prompt
        messages = self._build_messages_from_history(history, user_prompt)

        # Prepare request data for logging
        request_data = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "model_parameters": {"max_tokens": max_tokens, "temperature": temperature},
            "history_length": len(history) if history else 0,
        }

        try:
            with CallTimer() as timer:
                # Create streaming request
                stream = self.client.messages.create(
                    model=self.config.model_version,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=messages,
                    stream=True,
                )

                # Collect chunks for logging
                response_chunks = []

                # Yield tokens as they arrive
                for chunk in stream:
                    if (
                        chunk.type == "content_block_delta"
                        and hasattr(chunk, "delta")
                        and hasattr(chunk.delta, "text")
                    ):
                        text_chunk = chunk.delta.text
                        response_chunks.append(text_chunk)
                        yield text_chunk

                # After streaming is complete, log the full response
                full_response = "".join(response_chunks)
                self.logger.debug(
                    f"Claude async response complete: {len(full_response)} bytes / {len(full_response.split())} words"
                )

                # Prepare response data for logging
                response_data = {
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
                        response_data=response_data,
                        duration_ms=timer.duration_ms,
                        model_provider=self.config.model_provider,
                        model_version=self.config.model_version,
                    )

        except Exception as e:
            # Check if it's specifically an Anthropic API error
            is_api_error = False
            if (
                ANTHROPIC_AVAILABLE
                and APIStatusError is not None
                and APIConnectionError is not None
            ):
                is_api_error = isinstance(e, (APIStatusError, APIConnectionError))

            if is_api_error:
                # Log API errors
                if self.call_logger:
                    self.call_logger.log_llm_call(
                        request_data=request_data,
                        response_data={"error": str(e), "error_type": type(e).__name__},
                        duration_ms=timer.duration_ms if "timer" in locals() else 0,
                        model_provider=self.config.model_provider,
                        model_version=self.config.model_version,
                    )

                self.logger.error(f"Claude API error in streaming: {str(e)}")
                raise
            else:
                # Handle as general exception (including test cases)
                if self.call_logger:
                    self.call_logger.log_llm_call(
                        request_data=request_data,
                        response_data={"error": str(e), "error_type": type(e).__name__},
                        duration_ms=timer.duration_ms if "timer" in locals() else 0,
                        model_provider=self.config.model_provider,
                        model_version=self.config.model_version,
                    )

                self.logger.error(f"Unexpected error in Claude async agent: {str(e)}")
                raise

    def _build_messages_from_history(self, history, user_prompt):
        """
        Build Anthropic messages array from conversation history and current user prompt.
        
        :param history: List of ConversationMessage objects or None
        :param user_prompt: Current user prompt string
        :return: List of message dictionaries for Anthropic API
        """
        messages = []
        
        # Add conversation history if provided
        if history:
            for msg in history:
                # Only include user and assistant messages (skip system messages)
                if msg.role in ["user", "assistant"]:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
        
        # Add current user prompt
        messages.append({
            "role": "user", 
            "content": user_prompt
        })
        
        return messages

    @property
    def model(self) -> str:
        return self.config.model_version
