import logging
from typing import AsyncGenerator, List, Optional, TYPE_CHECKING
from .agent import Agent, AgentConfig
from .agent_logger import CallTimer

if TYPE_CHECKING:
    from ..cli.memory import ConversationMessage

try:
    import litellm
    from litellm import completion, acompletion

    LITELLM_AVAILABLE = True
except ImportError:
    litellm = None
    completion = None
    acompletion = None
    LITELLM_AVAILABLE = False


class LiteLLMAgent(Agent):
    """
    LiteLLM agent providing access to 100+ LLM providers through a unified interface.

    Supports all providers that LiteLLM supports including:
    - OpenAI, Anthropic, Google, Azure, AWS Bedrock
    - Cohere, HuggingFace, Ollama, Groq, Replicate
    - And many more through LiteLLM's unified API
    """

    def __init__(
        self,
        config: AgentConfig,
        logger: logging.Logger = None,
        prompts_dir=None,
    ):
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm package is required for LiteLLMAgent")

        # Use provided logger or create a default one
        if logger is None:
            logger = logging.getLogger(__name__)

        super().__init__(config, logger, prompts_dir)

        # Set up LiteLLM configuration
        self._setup_litellm_config()

    def _setup_litellm_config(self):
        """Setup LiteLLM configuration from agent config."""
        # Configure LiteLLM with any provider-specific auth
        provider_auth = self.config.provider_auth

        if provider_auth:
            # Set environment variables that LiteLLM expects
            import os

            # Common API keys that LiteLLM looks for
            if "api_key" in provider_auth:
                # This could be OpenAI, Anthropic, etc.
                api_key = provider_auth["api_key"]

                # Determine provider from model_version format (provider/model)
                if "/" in self.config.model_version:
                    provider_prefix = self.config.model_version.split("/")[0].lower()

                    if provider_prefix in ["openai", "gpt"]:
                        os.environ["OPENAI_API_KEY"] = api_key
                    elif provider_prefix in ["anthropic", "claude"]:
                        os.environ["ANTHROPIC_API_KEY"] = api_key
                    elif provider_prefix in ["cohere"]:
                        os.environ["COHERE_API_KEY"] = api_key
                    elif provider_prefix in ["huggingface", "hf"]:
                        os.environ["HUGGINGFACE_API_KEY"] = api_key
                    # Add more providers as needed

            # Google/Vertex AI specific
            if "project_id" in provider_auth:
                os.environ["VERTEXAI_PROJECT"] = provider_auth["project_id"]
            if "location" in provider_auth:
                os.environ["VERTEXAI_LOCATION"] = provider_auth["location"]

            # AWS Bedrock specific
            if "aws_access_key_id" in provider_auth:
                os.environ["AWS_ACCESS_KEY_ID"] = provider_auth["aws_access_key_id"]
            if "aws_secret_access_key" in provider_auth:
                os.environ["AWS_SECRET_ACCESS_KEY"] = provider_auth[
                    "aws_secret_access_key"
                ]
            if "aws_region" in provider_auth:
                os.environ["AWS_DEFAULT_REGION"] = provider_auth["aws_region"]

    def invoke(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List["ConversationMessage"]] = None,
    ) -> str:
        """
        Send the prompts to the LLM via LiteLLM and return the response.

        :param user_prompt: Specific user prompt
        :param system_prompt: Optional system prompt override. If None, loads from config via prompt manager.
        :param history: Optional conversation history for context
        :return: Text of response
        """
        # Use model parameters from config, with defaults
        model_params = self.config.model_parameters or {}
        max_tokens = model_params.get("max_tokens", 4096)
        temperature = model_params.get("temperature", 0.0)

        # Use provided system prompt or load from prompt manager
        if system_prompt is None:
            system_prompt = self.get_system_prompt()

        # Build messages array from history and current prompt
        messages = self._build_messages_from_history(
            history, user_prompt, system_prompt
        )

        # Prepare request data for logging
        request_data = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "model_parameters": {"max_tokens": max_tokens, "temperature": temperature},
            "history_length": len(history) if history else 0,
            "model": self.config.model_version,
        }

        try:
            with CallTimer() as timer:
                response = completion(
                    model=self.config.model_version,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **{
                        k: v
                        for k, v in model_params.items()
                        if k not in ["max_tokens", "temperature"]
                    },
                )

            self.logger.debug(f"LiteLLM raw response: {response}")

            if response and response.choices and len(response.choices) > 0:
                response_text = response.choices[0].message.content
                self.logger.debug(
                    f".. response: {len(response_text)} bytes / {len(response_text.split())} words"
                )

                # Prepare response data for logging
                response_data = {
                    "content": response_text,
                    "tokens_used": {
                        "input": (
                            getattr(response.usage, "prompt_tokens", None)
                            if hasattr(response, "usage")
                            else None
                        ),
                        "output": (
                            getattr(response.usage, "completion_tokens", None)
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
                        model_provider="litellm",
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
                        model_provider="litellm",
                        model_version=self.config.model_version,
                    )

                return error_response

        except Exception as e:
            # Log unexpected errors
            if self.call_logger:
                self.call_logger.log_llm_call(
                    request_data=request_data,
                    response_data={"error": str(e), "error_type": type(e).__name__},
                    duration_ms=timer.duration_ms if "timer" in locals() else 0,
                    model_provider="litellm",
                    model_version=self.config.model_version,
                )

            self.logger.error(f"LiteLLM error: {str(e)}")
            raise

    async def invoke_async(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List["ConversationMessage"]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Send the prompts to the LLM via LiteLLM and return a streaming async generator of response tokens.

        :param user_prompt: Specific user prompt
        :param system_prompt: Optional system prompt override. If None, loads from config via prompt manager.
        :param history: Optional conversation history for context
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
        messages = self._build_messages_from_history(
            history, user_prompt, system_prompt
        )

        # Prepare request data for logging
        request_data = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "model_parameters": {"max_tokens": max_tokens, "temperature": temperature},
            "history_length": len(history) if history else 0,
            "model": self.config.model_version,
        }

        try:
            with CallTimer() as timer:
                # Create streaming request
                response = await acompletion(
                    model=self.config.model_version,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    **{
                        k: v
                        for k, v in model_params.items()
                        if k not in ["max_tokens", "temperature"]
                    },
                )

                # Collect chunks for logging
                response_chunks = []

                # Yield tokens as they arrive
                async for chunk in response:
                    if chunk and chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            text_chunk = delta.content
                            response_chunks.append(text_chunk)
                            yield text_chunk

                # After streaming is complete, log the full response
                full_response = "".join(response_chunks)
                self.logger.debug(
                    f"LiteLLM async response complete: {len(full_response)} bytes / {len(full_response.split())} words"
                )

                # Prepare response data for logging
                response_data = {
                    "content": full_response,
                    "tokens_used": {
                        "input": None,  # Token usage not always available in streaming
                        "output": None,
                    },
                }

                # Log the LLM call if logger is configured
                if self.call_logger:
                    self.call_logger.log_llm_call(
                        request_data=request_data,
                        response_data=response_data,
                        duration_ms=timer.duration_ms,
                        model_provider="litellm",
                        model_version=self.config.model_version,
                    )

        except Exception as e:
            # Log unexpected errors
            if self.call_logger:
                self.call_logger.log_llm_call(
                    request_data=request_data,
                    response_data={"error": str(e), "error_type": type(e).__name__},
                    duration_ms=timer.duration_ms if "timer" in locals() else 0,
                    model_provider="litellm",
                    model_version=self.config.model_version,
                )

            self.logger.error(f"LiteLLM async error: {str(e)}")
            raise

    def _build_messages_from_history(self, history, user_prompt, system_prompt=None):
        """
        Build LiteLLM messages array from conversation history and current user prompt.

        :param history: List of ConversationMessage objects or None
        :param user_prompt: Current user prompt string
        :param system_prompt: System prompt string or None
        :return: List of message dictionaries for LiteLLM API
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history if provided
        if history:
            for msg in history:
                # Include all message types (system, user, assistant)
                messages.append({"role": msg.role, "content": msg.content})

        # Add current user prompt
        messages.append({"role": "user", "content": user_prompt})

        return messages

    @property
    def model(self) -> str:
        """Return the model name."""
        return self.config.model_version
