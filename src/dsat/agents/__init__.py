"""
Agents module for creating and managing LLM agents.

This module provides a framework for creating agents with different LLM providers,
including agent configuration, factory methods, and prompt versioning.
"""

from .agent import Agent, AgentConfig
from .prompts import PromptManager
from .agent_logger import AgentCallLogger, LoggingMode, CallTimer

# Core exports that are always available
__all__ = [
    "Agent",
    "AgentConfig",
    "PromptManager",
    "AgentCallLogger",
    "LoggingMode",
    "CallTimer",
]

# Optional scryptorum integration - only available if scryptorum is installed
try:
    from .experiment import AgentExperiment, AgentRun

    __all__.extend(["AgentExperiment", "AgentRun"])
except ImportError:
    AgentExperiment = None
    AgentRun = None

# Optional agent imports - only available if dependencies are installed
try:
    from .anthropic_agent import ClaudeLLMAgent, ANTHROPIC_AVAILABLE

    if ANTHROPIC_AVAILABLE:
        __all__.append("ClaudeLLMAgent")
except ImportError:
    ClaudeLLMAgent = None
    ANTHROPIC_AVAILABLE = False

try:
    from .vertex_agent import GoogleVertexAIAgent, VERTEX_AI_AVAILABLE

    if VERTEX_AI_AVAILABLE:
        __all__.append("GoogleVertexAIAgent")
except ImportError:
    GoogleVertexAIAgent = None
    VERTEX_AI_AVAILABLE = False

try:
    from .litellm_agent import LiteLLMAgent, LITELLM_AVAILABLE

    if LITELLM_AVAILABLE:
        __all__.append("LiteLLMAgent")
except ImportError:
    LiteLLMAgent = None
    LITELLM_AVAILABLE = False
