"""
Agent call logging system for LLM interactions.

This module provides flexible logging for agent LLM calls that integrates well with
host applications. Supports multiple output modes:
- Standard Python logger (default)
- Dedicated JSONL file
- Custom callback function
- Disabled (no logging)
"""

import json
import logging
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import uuid


class LoggingMode(Enum):
    """Agent logging output modes."""

    STANDARD = "standard"  # Use Python logger with structured data
    JSONL_FILE = "jsonl_file"  # Write to dedicated JSONL file
    CALLBACK = "callback"  # Call custom function
    DISABLED = "disabled"  # No logging


class AgentCallLogger:
    """
    Logger for agent LLM calls with multiple output modes.

    This class handles logging of LLM interactions in a way that's friendly
    to host applications using this package.
    """

    def __init__(
        self,
        agent_name: str,
        mode: LoggingMode = LoggingMode.STANDARD,
        file_path: Optional[Union[str, Path]] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        level: str = "standard",
    ):
        """
        Initialize the agent call logger.

        Args:
            agent_name: Name of the agent for logger hierarchy
            mode: Logging output mode
            file_path: Path for JSONL file mode (required if mode is JSONL_FILE)
            callback: Custom logging function for callback mode
            level: Logging detail level ("minimal" or "standard")
        """
        self.agent_name = agent_name
        self.mode = mode
        self.level = level
        self._lock = threading.RLock()

        # Set up Python logger for standard mode and warnings
        self.logger = logging.getLogger(f"dsat.agents.{agent_name}")

        if mode == LoggingMode.STANDARD:
            # Standard Python logger mode - host app controls routing
            pass

        elif mode == LoggingMode.JSONL_FILE:
            if not file_path:
                raise ValueError("file_path is required for JSONL_FILE mode")
            self.file_path = Path(file_path)
            # Ensure parent directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

        elif mode == LoggingMode.CALLBACK:
            if not callback:
                raise ValueError("callback function is required for CALLBACK mode")
            self.callback = callback

        elif mode == LoggingMode.DISABLED:
            # No setup needed for disabled mode
            pass

    def log_llm_call(
        self,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        duration_ms: float,
        model_provider: str,
        model_version: str,
        **metadata,
    ) -> None:
        """
        Log an LLM call with request/response details.

        Args:
            request_data: Request details (prompts, parameters)
            response_data: Response details (content, tokens, etc.)
            duration_ms: Call duration in milliseconds
            model_provider: Provider name (e.g., "anthropic", "google")
            model_version: Model version (e.g., "claude-3-5-haiku-latest")
            **metadata: Additional metadata to include
        """
        if self.mode == LoggingMode.DISABLED:
            return

        # Generate unique call ID
        call_id = str(uuid.uuid4())

        # Build log data structure
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "call_id": call_id,
            "agent_name": self.agent_name,
            "model_provider": model_provider,
            "model_version": model_version,
            "duration_ms": duration_ms,
            **metadata,
        }

        # Add request/response data based on detail level
        if self.level == "minimal":
            log_data.update(
                {
                    "request": {
                        "user_prompt_length": len(request_data.get("user_prompt", "")),
                        "system_prompt_length": len(
                            request_data.get("system_prompt", "")
                        ),
                        "model_parameters": request_data.get("model_parameters", {}),
                    },
                    "response": {
                        "content_length": len(response_data.get("content", "")),
                        "tokens_used": response_data.get("tokens_used", {}),
                    },
                }
            )
        else:  # standard level
            log_data.update({"request": request_data, "response": response_data})

        # Route to appropriate output
        try:
            if self.mode == LoggingMode.STANDARD:
                self._log_to_standard_logger(log_data)
            elif self.mode == LoggingMode.JSONL_FILE:
                self._log_to_jsonl_file(log_data)
            elif self.mode == LoggingMode.CALLBACK:
                self._log_to_callback(log_data)
        except Exception as e:
            # Log error to standard logger and continue
            self.logger.error(f"Failed to log LLM call: {e}")

    def _log_to_standard_logger(self, log_data: Dict[str, Any]) -> None:
        """Log to standard Python logger with structured data."""
        # Create a summary message
        duration = log_data["duration_ms"]
        model = log_data["model_version"]
        call_id = log_data["call_id"][:8]  # Short ID for readability

        message = (
            f"LLM call completed in {duration:.1f}ms (model={model}, call_id={call_id})"
        )

        # Include full structured data in extra for structured logging handlers
        self.logger.info(message, extra={"llm_call_data": log_data})

    def _log_to_jsonl_file(self, log_data: Dict[str, Any]) -> None:
        """Log to dedicated JSONL file with thread safety."""
        with self._lock:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data) + "\n")

    def _log_to_callback(self, log_data: Dict[str, Any]) -> None:
        """Log via custom callback function."""
        self.callback(log_data)

    def log_warning(self, message: str) -> None:
        """Log a warning message via standard logger."""
        self.logger.warning(message)

    @classmethod
    def create_from_config(
        cls, agent_name: str, logging_config: Dict[str, Any]
    ) -> Optional["AgentCallLogger"]:
        """
        Create logger from configuration dictionary.

        Args:
            agent_name: Name of the agent
            logging_config: Configuration dictionary with logging settings

        Returns:
            AgentCallLogger instance or None if logging disabled
        """
        if not logging_config.get("enabled", False):
            return None

        mode_str = logging_config.get("mode", "standard").lower()
        try:
            mode = LoggingMode(mode_str)
        except ValueError:
            # Fall back to standard mode with warning
            logger = logging.getLogger(f"dsat.agents.{agent_name}")
            logger.warning(
                f"Invalid logging mode '{mode_str}', falling back to 'standard'"
            )
            mode = LoggingMode.STANDARD

        kwargs = {
            "agent_name": agent_name,
            "mode": mode,
            "level": logging_config.get("level", "standard"),
        }

        if mode == LoggingMode.JSONL_FILE:
            file_path = logging_config.get("file_path")
            if not file_path:
                logger = logging.getLogger(f"dsat.agents.{agent_name}")
                logger.warning(
                    "JSONL file logging enabled but no file_path specified. "
                    "LLM calls will not be logged. Set logging.file_path in agent config."
                )
                return None
            kwargs["file_path"] = file_path

        elif mode == LoggingMode.CALLBACK:
            callback = logging_config.get("callback")
            if not callback:
                logger = logging.getLogger(f"dsat.agents.{agent_name}")
                logger.warning(
                    "Callback logging enabled but no callback function specified. "
                    "LLM calls will not be logged. Set logging.callback in agent config."
                )
                return None
            kwargs["callback"] = callback

        return cls(**kwargs)


class CallTimer:
    """Context manager for timing LLM calls."""

    def __init__(self):
        self.start_time = None
        self.duration_ms = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            self.duration_ms = (time.perf_counter() - self.start_time) * 1000
