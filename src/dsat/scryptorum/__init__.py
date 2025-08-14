"""
Scryptorum - Modern LLM Experiment Framework

A lightweight, annotation-driven framework for running and tracking LLM experiments.
"""

from .core.experiment import Experiment
from .core.runs import Run, RunType
from .core.decorators import experiment, metric, timer, llm_call, set_default_run_type
from .execution.runner import Runner

__version__ = "0.1.0"
__all__ = [
    "Experiment",
    "Run",
    "RunType",
    "experiment",
    "metric",
    "timer",
    "llm_call",
    "set_default_run_type",
    "Runner",
]
