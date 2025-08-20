"""
DSAT - Dan's Simple Agent Toolkit

A comprehensive Python toolkit for building LLM applications and running experiments.
Consists of two main frameworks:

1. Agents Framework: Multi-provider LLM agent system
2. Scryptorum Framework: Experiment tracking and management
"""

__version__ = "0.1.0"

# Import main modules for convenience
from .agents import Agent, AgentConfig
from .scryptorum import experiment, metric, timer, llm_call

__all__ = [
    "Agent",
    "AgentConfig",
    "experiment",
    "metric",
    "timer",
    "llm_call",
]
