import logging
from typing import AsyncGenerator
from .agent import Agent, AgentConfig
from .agent_logger import CallTimer

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel

    VERTEX_AI_AVAILABLE = True
except ImportError:
    vertexai = None
    GenerativeModel = None
    VERTEX_AI_AVAILABLE = False


class GoogleVertexAIAgent(Agent):
    """
    Google Vertex AI LLM agent.
    """

    def __init__(
        self,
        config: AgentConfig = None,
        project_id: str = None,
        location: str = None,
        model: str = None,
        logger: logging.Logger = None,
        prompts_dir=None,
    ):
        # Support both old API (project_id, location, model) and new API (config)
        if config is None:
            # Backward compatibility - create config from parameters
            if project_id is None or model is None:
                raise ValueError(
                    "Either config must be provided, or both project_id and model must be provided"
                )

            config = AgentConfig(
                agent_name="vertex",
                model_provider="google",
                model_family="gemini",
                model_version=model,
                prompt="default:v1",
                provider_auth={
                    "project_id": project_id,
                    "location": location or "us-central1",
                },
            )
        else:
            # Use provided config, but allow parameter overrides for backward compatibility
            if project_id is not None:
                config.provider_auth["project_id"] = project_id
            if location is not None:
                config.provider_auth["location"] = location

        # Use provided logger or create a default one
        if logger is None:
            logger = logging.getLogger(__name__)

        super().__init__(config, logger, prompts_dir)

        if not VERTEX_AI_AVAILABLE:
            raise ImportError(
                "google-cloud-aiplatform package is required for Google Vertex AI support"
            )

        # Get auth parameters from config
        project_id_from_config = config.provider_auth.get("project_id")
        location_from_config = config.provider_auth.get("location", "us-central1")

        if not project_id_from_config:
            raise ValueError(
                "project_id is required in provider_auth for GoogleVertexAIAgent"
            )

        vertexai.init(project=project_id_from_config, location=location_from_config)
        self.client = GenerativeModel(config.model_version)

    def invoke(self, user_prompt: str, system_prompt: str = None, history=None) -> str:
        # Use model parameters from config, with defaults
        model_params = self.config.model_parameters or {}
        temperature = model_params.get("temperature", 0.3)
        max_output_tokens = model_params.get("max_output_tokens", 20000)

        # Use provided system prompt or load from prompt manager
        if system_prompt is None:
            system_prompt = self.get_system_prompt()

        # Build conversation context with history
        full_prompt = self._build_conversation_context(system_prompt, history, user_prompt)

        # Prepare request data for logging
        request_data = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "full_prompt": full_prompt,
            "history_length": len(history) if history else 0,
            "model_parameters": {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        }

        try:
            with CallTimer() as timer:
                response = self.client.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                    },
                )

            self.logger.debug(f"Vertex AI raw response: {response.text}")
            self.logger.debug(
                f".. response: {len(response.text)} bytes / {len(response.text.split())} words"
            )

            # Prepare response data for logging
            response_data = {
                "content": response.text,
                "tokens_used": {
                    "input": (
                        getattr(response.usage_metadata, "prompt_token_count", None)
                        if hasattr(response, "usage_metadata")
                        else None
                    ),
                    "output": (
                        getattr(response.usage_metadata, "candidates_token_count", None)
                        if hasattr(response, "usage_metadata")
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

            return response.text

        except Exception as e:
            # Log errors
            if self.call_logger:
                self.call_logger.log_llm_call(
                    request_data=request_data,
                    response_data={"error": str(e), "error_type": type(e).__name__},
                    duration_ms=timer.duration_ms if "timer" in locals() else 0,
                    model_provider=self.config.model_provider,
                    model_version=self.config.model_version,
                )

            self.logger.error(f"Vertex AI API error: {str(e)}")
            raise

    async def invoke_async(
        self, user_prompt: str, system_prompt: str = None, history=None
    ) -> AsyncGenerator[str, None]:
        """
        Send the prompts to Vertex AI and return a streaming async generator of response tokens.

        :param user_prompt: Specific user prompt
        :param system_prompt: Optional system prompt override. If None, loads from config via prompt manager.
        :return: AsyncGenerator yielding response text chunks
        """
        # Use model parameters from config, with defaults
        model_params = self.config.model_parameters or {}
        temperature = model_params.get("temperature", 0.3)
        max_output_tokens = model_params.get("max_output_tokens", 20000)

        # Use provided system prompt or load from prompt manager
        if system_prompt is None:
            system_prompt = self.get_system_prompt()

        # Build conversation context with history
        full_prompt = self._build_conversation_context(system_prompt, history, user_prompt)

        # Prepare request data for logging
        request_data = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "full_prompt": full_prompt,
            "history_length": len(history) if history else 0,
            "model_parameters": {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        }

        try:
            with CallTimer() as timer:
                # Generate streaming content
                response_stream = self.client.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                    },
                    stream=True,
                )

                # Collect chunks for logging
                response_chunks = []

                # Yield tokens as they arrive
                for chunk in response_stream:
                    if hasattr(chunk, "text") and chunk.text:
                        text_chunk = chunk.text
                        response_chunks.append(text_chunk)
                        yield text_chunk

            # After streaming is complete, log the full response
            full_response = "".join(response_chunks)
            self.logger.debug(
                f"Vertex AI async response complete: {len(full_response)} bytes / {len(full_response.split())} words"
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
            # Log errors
            if self.call_logger:
                self.call_logger.log_llm_call(
                    request_data=request_data,
                    response_data={"error": str(e), "error_type": type(e).__name__},
                    duration_ms=timer.duration_ms if "timer" in locals() else 0,
                    model_provider=self.config.model_provider,
                    model_version=self.config.model_version,
                )

            self.logger.error(f"Vertex AI API error in streaming: {str(e)}")
            raise

    def _build_conversation_context(self, system_prompt, history, user_prompt):
        """
        Build conversation context for Vertex AI from system prompt, history, and current user prompt.
        
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
