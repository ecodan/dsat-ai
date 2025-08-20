"""
EchoAgent - A simple test agent that echoes inputs for testing purposes.
"""

import logging
import asyncio
from pathlib import Path
from typing import Optional, Union, AsyncGenerator
from dsat.agents.agent import Agent, AgentConfig


class EchoAgent(Agent):
    """
    Simple test agent that echoes back the user prompt with some formatting.
    
    This agent doesn't require any external API keys or services, making it
    perfect for testing agent functionality without mocking.
    """

    def __init__(self, config: AgentConfig, logger: logging.Logger = None, prompts_dir: Optional[Union[str, Path]] = None):
        if logger is None:
            logger = logging.getLogger(__name__)
        super().__init__(config, logger, prompts_dir)
        self.call_count = 0

    def invoke(self, user_prompt: str, system_prompt: str = None) -> str:
        """Echo back the user prompt with system context."""
        self.call_count += 1
        
        # Use provided system prompt or load from prompt manager
        if system_prompt is None:
            system_prompt = self.get_system_prompt()
        
        # Log the interaction for testing
        self.logger.debug(f"EchoAgent call #{self.call_count}")
        self.logger.debug(f"System prompt: {system_prompt}")
        self.logger.debug(f"User prompt: {user_prompt}")
        
        # Create formatted response
        if system_prompt:
            response = f"[{system_prompt.strip()}] Echo: {user_prompt}"
        else:
            response = f"Echo: {user_prompt}"
        
        # Log the LLM call if logger is configured
        if self.call_logger:
            self.call_logger.log_llm_call(
                request_data={
                    "user_prompt": user_prompt,
                    "system_prompt": system_prompt,
                    "model_parameters": self.config.model_parameters or {}
                },
                response_data={
                    "content": response,
                    "tokens_used": {
                        "input": len(user_prompt.split()),
                        "output": len(response.split())
                    }
                },
                duration_ms=10.0,  # Simulate fast response
                model_provider=self.config.model_provider,
                model_version=self.config.model_version
            )
        
        return response

    async def invoke_async(self, user_prompt: str, system_prompt: str = None) -> AsyncGenerator[str, None]:
        """
        Echo back the user prompt with system context, yielding characters one by one.
        
        :param user_prompt: Specific user prompt
        :param system_prompt: Optional system prompt override. If None, loads from config via prompt manager.
        :return: AsyncGenerator yielding response text chunks
        """
        self.call_count += 1
        
        # Use provided system prompt or load from prompt manager
        if system_prompt is None:
            system_prompt = self.get_system_prompt()
        
        # Log the interaction for testing
        self.logger.debug(f"EchoAgent async call #{self.call_count}")
        self.logger.debug(f"System prompt: {system_prompt}")
        self.logger.debug(f"User prompt: {user_prompt}")
        
        # Create formatted response
        if system_prompt:
            response = f"[{system_prompt.strip()}] Echo: {user_prompt}"
        else:
            response = f"Echo: {user_prompt}"
        
        # Yield response character by character with small delays to simulate streaming
        for char in response:
            yield char
            await asyncio.sleep(0.01)  # Small delay to simulate realistic streaming
        
        # Log the LLM call if logger is configured
        if self.call_logger:
            self.call_logger.log_llm_call(
                request_data={
                    "user_prompt": user_prompt,
                    "system_prompt": system_prompt,
                    "model_parameters": self.config.model_parameters or {}
                },
                response_data={
                    "content": response,
                    "tokens_used": {
                        "input": len(user_prompt.split()),
                        "output": len(response.split())
                    }
                },
                duration_ms=len(response) * 10.0,  # Simulate streaming duration
                model_provider=self.config.model_provider,
                model_version=self.config.model_version
            )

    @property
    def model(self) -> str:
        """Return the configured model version."""
        return self.config.model_version


def create_echo_agent_config(agent_name: str = "echo_test", prompt: str = "echo:v1") -> AgentConfig:
    """Create a standard EchoAgent configuration for testing."""
    return AgentConfig(
        agent_name=agent_name,
        model_provider="echo",
        model_family="test",
        model_version="echo-v1",
        prompt=prompt,
        model_parameters={"temperature": 0.0, "max_tokens": 1000},
        provider_auth={},  # No auth needed for echo agent
        custom_configs={"test_agent": True}
    )


# Register EchoAgent with the Agent factory
def register_echo_agent():
    """Register EchoAgent with the Agent.create factory method."""
    from dsat.agents.agent import Agent
    
    # Store original create method
    original_create = Agent.create
    
    @classmethod
    def enhanced_create(cls, config: AgentConfig, logger: logging.Logger = None, prompts_dir: Optional[Union[str, Path]] = None) -> 'Agent':
        """Enhanced create method that supports echo provider."""
        if config.model_provider.lower() == "echo":
            return EchoAgent(config, logger, prompts_dir)
        else:
            return original_create(config, logger, prompts_dir)
    
    # Replace the create method
    Agent.create = enhanced_create


# Auto-register when imported
register_echo_agent()